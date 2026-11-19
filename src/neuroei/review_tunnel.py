"""Minimal foreground SSH tunnel for the review console.

The tunnel binds only to the local loopback and forwards raw TCP bytes to the
fixed review-console address through a fixed SSH account.  It deliberately has
no HTTP awareness: headers, origins, cookies and CSRF tokens pass through
unchanged, so the remote application keeps enforcing its own CSRF checks.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import socket
import socketserver
import sys
import threading
from typing import Any

import paramiko

LOCAL_HOST = '127.0.0.1'
SSH_HOST = 'researchpulse-host.example.invalid'
SSH_PORT = 22
SSH_USER = 'researchpulse'
TARGET_HOST = '127.0.0.1'
TARGET_PORT = 8768
DEFAULT_PORT = 8768
CONNECT_TIMEOUT = 15
KEEPALIVE_SECONDS = 30
BUFFER_SIZE = 64 * 1024


def validate_port(value: object) -> int:
    """Return a real int in 1..65535; booleans and strings are not ports."""
    if type(value) is not int:
        raise ValueError('port must be an integer')
    if not 1 <= value <= 65535:
        raise ValueError('port must be between 1 and 65535')
    return value


def _port_argument(value: str) -> int:
    if not value.isdigit():
        raise argparse.ArgumentTypeError('port must be an integer')
    try:
        port = validate_port(int(value, 10))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return port


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='python -m neuroei.review_tunnel',
        description='Forward a local TCP port to the fixed review console over SSH.',
    )
    parser.add_argument(
        '--port',
        type=_port_argument,
        default=DEFAULT_PORT,
        help=f'local loopback port (default: {DEFAULT_PORT})',
    )
    return parser.parse_args(argv)


def _connect(password: str) -> tuple[paramiko.SSHClient, Any]:
    """Open the fixed SSH transport and return it with its owning client."""
    client = paramiko.SSHClient()
    try:
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        client.connect(
            hostname=SSH_HOST,
            port=SSH_PORT,
            username=SSH_USER,
            password=password,
            timeout=CONNECT_TIMEOUT,
            banner_timeout=CONNECT_TIMEOUT,
            auth_timeout=CONNECT_TIMEOUT,
            look_for_keys=False,
            allow_agent=False,
        )
        transport = client.get_transport()
        if transport is None:
            raise RuntimeError('SSH transport is unavailable')
        transport.set_keepalive(KEEPALIVE_SECONDS)
    except BaseException:
        # main() never receives the local client, so every initialization
        # failure must close it here before propagating the original error.
        _close_quietly(client)
        raise
    return client, transport


def _close_quietly(resource: Any) -> None:
    with contextlib.suppress(Exception):
        resource.close()


def _shutdown_socket(sock: Any) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    _close_quietly(sock)


def _relay(client: Any, channel: Any) -> None:
    """Copy bytes both ways until either side closes, then close both sides."""
    stop = threading.Event()

    def pump(source: Any, destination: Any) -> None:
        try:
            while not stop.is_set():
                data = source.recv(BUFFER_SIZE)
                if not data:
                    break
                destination.sendall(data)
        except Exception:  # noqa: BLE001 - close both sides on any socket/channel failure
            stop.set()
        finally:
            stop.set()
            _shutdown_socket(client)
            _close_quietly(channel)

    threads = [
        threading.Thread(target=pump, args=(client, channel), daemon=True),
        threading.Thread(target=pump, args=(channel, client), daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        transport = self.server.transport
        try:
            peer_host, peer_port = self.request.getpeername()
        except OSError:
            peer_host, peer_port = LOCAL_HOST, 0
        try:
            channel = transport.open_channel(
                'direct-tcpip',
                (TARGET_HOST, TARGET_PORT),
                (peer_host, peer_port),
            )
        except Exception:  # noqa: BLE001 - a channel failure must not escape the client thread
            if not transport.is_active():
                self.server.disconnected.set()
                self.server.shutdown()
            _shutdown_socket(self.request)
            return
        try:
            _relay(self.request, channel)
        finally:
            _close_quietly(channel)
            _shutdown_socket(self.request)


class _TunnelServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = False
    daemon_threads = True

    def __init__(self, address: tuple[str, int], handler: type, transport: Any):
        self.transport = transport
        self.disconnected = threading.Event()
        self._monitor_stop = threading.Event()
        self._monitor = threading.Thread(
            target=self._watch_transport,
            name='review-tunnel-ssh-monitor',
            daemon=True,
        )
        super().__init__(address, handler)

    def serve_forever(self, poll_interval: float = 0.5) -> None:
        self._monitor.start()
        try:
            super().serve_forever(poll_interval)
        finally:
            # BaseServer.shutdown() waits only for __is_shut_down, which
            # BaseServer.serve_forever() sets before returning.  Setting the
            # monitor stop flag first therefore cannot deadlock this join.
            self._monitor_stop.set()
            self._monitor.join(timeout=2.0)

    def _watch_transport(self) -> None:
        while not self._monitor_stop.is_set():
            try:
                active = self.transport.is_active()
            except Exception:  # noqa: BLE001 - a dead transport is a clean shutdown
                active = False
            if not active:
                self.disconnected.set()
                if not self._monitor_stop.is_set():
                    self.shutdown()
                return
            self._monitor_stop.wait(0.5)


def _safe_banner(port: int) -> None:
    print(f'Review tunnel ready: http://{LOCAL_HOST}:{port}/')
    print(f'Forwarding raw TCP to {TARGET_HOST}:{TARGET_PORT} via {SSH_USER}@{SSH_HOST}')


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    password = os.environ.get('NEUROEI_SSH_PASSWORD')
    if not password:
        print('error: NEUROEI_SSH_PASSWORD is not set', file=sys.stderr)
        return 2

    try:
        client, transport = _connect(password)
    except Exception as exc:  # noqa: BLE001 - CLI must fail closed without a traceback
        print(f'error: SSH connection failed ({type(exc).__name__})', file=sys.stderr)
        return 1

    try:
        try:
            server = _TunnelServer((LOCAL_HOST, args.port), _Handler, transport)
        except OSError:
            print(f'error: cannot bind {LOCAL_HOST}:{args.port}', file=sys.stderr)
            return 1

        try:
            _safe_banner(args.port)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
        except Exception as exc:  # noqa: BLE001 - CLI must fail closed without a traceback
            print(f'error: tunnel failed ({type(exc).__name__})', file=sys.stderr)
            return 1
        finally:
            server.server_close()

        if server.disconnected.is_set():
            print('error: SSH connection lost; tunnel stopped', file=sys.stderr)
            return 1
        return 0
    finally:
        _close_quietly(transport)
        _close_quietly(client)


if __name__ == '__main__':
    raise SystemExit(main())
