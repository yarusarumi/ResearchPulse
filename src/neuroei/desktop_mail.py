"""Profile-local SMTP settings for sending a report to the configured sender."""
import json
import re
from pathlib import Path
from smtplib import SMTPAuthenticationError, SMTPException
from tempfile import NamedTemporaryFile

from .account_email import SMTPMailer, normalize_email
from .desktop_models import protect_secret, unprotect_secret
from .feishu_user_auth import DESKTOP_ORIGIN

_FILE = 'mail-settings.json'


def _read(profile):
    try:
        return json.loads((profile / _FILE).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {}


def settings(profile):
    saved = _read(profile)
    return {'configured': bool(saved), 'address': saved.get('address', ''),
            'host': saved.get('host', ''), 'port': saved.get('port', 465),
            'security': saved.get('security', 'ssl')}


def save(profile, payload):
    if payload == {'remove': True}:
        (profile / _FILE).unlink(missing_ok=True)
        return settings(profile)
    if (set(payload) != {'address', 'host', 'port', 'security', 'authorization_code'}
            or not all(isinstance(payload[k], str) for k in ('address', 'host', 'security', 'authorization_code'))):
        raise ValueError('请填写邮箱和 SMTP 设置。')
    address = normalize_email(payload['address'])
    host = payload['host'].strip().lower()
    port, security = payload['port'], payload['security']
    code = payload['authorization_code'].strip()
    if not re.fullmatch(r'[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?', host):
        raise ValueError('请输入 SMTP 服务器域名，不含协议、路径或端口。')
    if type(port) is not int or not 1 <= port <= 65535 or security not in ('ssl', 'starttls'):
        raise ValueError('请核对 SMTP 端口和加密方式。')
    connection = {'address': address, 'host': host, 'port': port, 'security': security}
    previous = _read(profile)
    if len(code) > 512 or (not code and any(previous.get(key) != value for key, value in connection.items())):
        raise ValueError('请重新填写该邮箱的 SMTP 授权码。')
    sealed = protect_secret(code) if code else previous['authorization_code']
    profile.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode='w', encoding='utf-8', dir=profile, delete=False) as pending:
        temporary = Path(pending.name)
        json.dump({**connection, 'authorization_code': sealed}, pending)
    try:
        temporary.replace(profile / _FILE)
    finally:
        temporary.unlink(missing_ok=True)
    return settings(profile)


def mailer(profile, origin=DESKTOP_ORIGIN):
    saved = _read(profile)
    if not saved:
        return None
    address = saved['address']
    return SMTPMailer(saved['host'], saved['port'], address, origin, address,
                      unprotect_secret(saved['authorization_code']), saved['security'])


def send_test(profile, origin):
    smtp = mailer(profile, origin)
    if not smtp:
        raise ValueError('请先保存邮件设置。')
    try:
        smtp.send_daily_report(smtp.sender, 'ResearchPulse 测试邮件',
                               'ResearchPulse 已通过您配置的邮箱向同一个地址提交测试邮件。')
    except SMTPAuthenticationError:
        raise ValueError('SMTP 登录失败。请确认邮箱已开启 SMTP 服务，并核对授权码。') from None
    except (OSError, SMTPException):
        raise ValueError('测试邮件发送失败。请核对网络、SMTP 服务器、端口和加密方式。') from None
    return {'sent': True}
