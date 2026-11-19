"""Email verification and one-time account recovery for the lab portal."""
import json
import os
import re
import secrets
import smtplib
import ssl
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage
from time import sleep
from urllib.parse import urlsplit

from . import accounts

VALID_FOR_MINUTES = 30
EMAIL_RESPONSE = '如果可以处理这项请求，验证邮件将发送到该邮箱。请检查收件箱和垃圾邮件。'
INVALID_LINK = '链接无效、已用过或已过期，请重新申请。'
_EMAIL = re.compile(r"[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+", re.ASCII)
_TOKEN = re.compile(r'[A-Za-z0-9_-]{43}', re.ASCII)


def normalize_email(value):
    email = value.strip().lower()
    if len(email) > 254 or not _EMAIL.fullmatch(email):
        raise ValueError('请输入有效的邮箱地址。')
    return email


class TooManyEmails(ValueError):
    pass


class WrongCurrentPassword(ValueError):
    pass


class SMTPMailer:
    def __init__(self, host, port, sender, origin, username, password, security):
        self.host, self.port, self.sender, self.origin = host, port, sender, origin
        self.username, self.password, self.security = username, password, security

    @classmethod
    def configured(cls):
        host, sender, origin = (os.environ.get('NEUROEI_' + k, '').strip()
                                for k in ('SMTP_HOST', 'MAIL_FROM', 'PUBLIC_ORIGIN'))
        if not all((host, sender, origin)):
            return None
        url = urlsplit(origin)
        if url.scheme != 'https' or not url.hostname or url.username or url.path not in ('', '/') or url.query or url.fragment:
            raise ValueError('NEUROEI_PUBLIC_ORIGIN must be a trusted HTTPS origin')
        security = os.environ.get('NEUROEI_SMTP_SECURITY', 'starttls')
        if security not in ('starttls', 'ssl'):
            raise ValueError('NEUROEI_SMTP_SECURITY must be starttls or ssl')
        username = os.environ.get('NEUROEI_SMTP_USERNAME', '')
        password = os.environ.get('NEUROEI_SMTP_PASSWORD', '')
        if bool(username) != bool(password):
            raise ValueError('Configure both SMTP username and password when authentication is required')
        port = int(os.environ.get('NEUROEI_SMTP_PORT', '465' if security == 'ssl' else '587'))
        if not 1 <= port <= 65535:
            raise ValueError('Invalid SMTP port')
        return cls(host, port, normalize_email(sender), origin.rstrip('/'), username, password, security)

    def send_daily_report(self, email, subject, body):
        message = EmailMessage()
        message['From'], message['To'], message['Subject'] = self.sender, normalize_email(email), subject
        message.set_content(body)
        context = ssl.create_default_context()
        transport = (smtplib.SMTP_SSL(self.host, self.port, timeout=8, context=context)
                     if self.security == 'ssl' else smtplib.SMTP(self.host, self.port, timeout=8))
        with transport as smtp:
            if self.security == 'starttls':
                smtp.starttls(context=context)
            if self.username:
                smtp.login(self.username, self.password)
            smtp.send_message(message)

    def __call__(self, email, action, token, username=None):
        label = {'register': '注册 ResearchPulse', 'reset': '重置 ResearchPulse 密码', 'bind': '绑定 ResearchPulse 邮箱'}[action]
        message = EmailMessage()
        message['From'], message['To'], message['Subject'] = self.sender, email, label
        account_line = f'用户名：{username}\n\n' if username else ''
        message.set_content(f'{account_line}请打开链接完成操作（{VALID_FOR_MINUTES} 分钟内有效）：\n'
                            f'{self.origin}/account/{action}#{token}\n\n'
                            '若你没有发起这项请求，可以忽略这封邮件。')
        context = ssl.create_default_context()
        transport = (smtplib.SMTP_SSL(self.host, self.port, timeout=8, context=context)
                     if self.security == 'ssl' else smtplib.SMTP(self.host, self.port, timeout=8))
        with transport as smtp:
            if self.security == 'starttls':
                smtp.starttls(context=context)
            if self.username:
                smtp.login(self.username, self.password)
            smtp.send_message(message)


def deliver(mailer, email, action, token, username):
    """Retry a temporary SMTP failure using the same verification link."""
    for delay in (1, 3, None):
        try:
            mailer(email, action, token, username)
            return
        except smtplib.SMTPRecipientsRefused as exc:
            if delay is None or not any(400 <= code < 500 for code, _ in exc.recipients.values()):
                raise
        except smtplib.SMTPResponseException as exc:
            if delay is None or not 400 <= exc.smtp_code < 500:
                raise
        except OSError:
            if delay is None:
                raise
        sleep(delay)


def _by_email(store, email):
    return next((a for a in store.list('account') if a.get('verified_email') == email), None)


def issue(store, action, email, *, account=None, password=None):
    email = normalize_email(email)
    token = secrets.token_urlsafe(32)
    clock = datetime.now(UTC)
    with store.transaction() as db:
        db.execute('BEGIN IMMEDIATE')
        owner = _by_email(store, email)
        if action == 'bind':
            current = accounts.find(store, account['user_id']) if account else None
            if not current or current['disabled'] or not accounts._matches(current['password_hash'], password or ''):
                raise WrongCurrentPassword('当前密码不正确。')
            if owner:
                raise ValueError('此邮箱无法绑定，请检查账号设置或使用其他邮箱。')
            account = current
        elif action == 'register':
            if owner:
                return None
        elif action == 'reset':
            if not owner or owner['disabled']:
                return None
            account = owner
        else:
            raise ValueError('未知邮箱操作。')
        db.execute("DELETE FROM objects WHERE kind='account_mail_token' AND json_extract(body,'$.expires_at')<=?",
                   (clock.isoformat(),))
        recent = clock - timedelta(minutes=1)
        quota, sent = db.execute(
            "SELECT count(*), coalesce(sum(json_extract(body,'$.email')=?),0) FROM objects "
            "WHERE kind='account_mail_token' AND json_extract(body,'$.sent_at')>?",
            (email, recent.isoformat())).fetchone()
        # Limit across recipients as well as per address on a shared lab server.
        if sent or quota >= 20:
            raise TooManyEmails('发送过于频繁，请一分钟后重试。')
        record = {'action': action, 'email': email, 'sent_at': clock.isoformat(),
                  'expires_at': (clock + timedelta(minutes=VALID_FOR_MINUTES)).isoformat()}
        if account:
            record.update(user_id=account['user_id'], credential=accounts._token_digest(account['password_hash']),
                          previous_email=account.get('verified_email'))
        db.execute('INSERT INTO objects VALUES (?,?,?)',
                   ('account_mail_token', accounts._token_digest(token), json.dumps(record)))
    return token, account['username'] if account and action == 'reset' else None


def complete(store, action, token, *, username='', display_name='', password='', account=None):
    if not _TOKEN.fullmatch(token):
        raise ValueError(INVALID_LINK)
    key = accounts._token_digest(token)
    with store.transaction() as db:
        db.execute('BEGIN IMMEDIATE')
        record = accounts._optional(store, 'account_mail_token', key)
        if (not record or record['action'] != action or record.get('used')
                or record['expires_at'] <= datetime.now(UTC).isoformat()):
            raise ValueError(INVALID_LINK)
        email = record['email']
        owner = _by_email(store, email)
        if action == 'register' and not owner:
            accounts.create(store, username, display_name, password, db=db, email=email)
        elif action in ('reset', 'bind'):
            current = accounts.find(store, record['user_id'])
            if (not current or current['disabled'] or current.get('verified_email') != record['previous_email']
                    or accounts._token_digest(current['password_hash']) != record['credential']):
                raise ValueError(INVALID_LINK)
            if action == 'reset' and owner and owner['user_id'] == current['user_id']:
                accounts._require_password(password)
                accounts.replace_password(db, current, accounts._hash(password))
            elif action == 'bind' and account and account['user_id'] == current['user_id'] and not owner:
                current['verified_email'] = email
                db.execute("UPDATE objects SET body=? WHERE kind='account' AND id=?",
                           (json.dumps(current), current['user_id']))
            else:
                raise ValueError(INVALID_LINK)
        else:
            raise ValueError(INVALID_LINK)
        # Retain the send timestamp until expiration so consuming a link does not
        # reset the per-address cooldown.
        db.execute("UPDATE objects SET body=? WHERE kind='account_mail_token' AND id=?",
                   (json.dumps({**record, 'used': True}), key))
