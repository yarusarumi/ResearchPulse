"""Manual research workspace launcher with an explicit data directory."""
import argparse
import getpass
from pathlib import Path

from . import accounts
from .portal import make_server
from .store import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', required=True, type=Path)
    parser.add_argument('--port', type=int, default=8878)
    parser.add_argument('--init-admin', metavar='USERNAME', help='Create the first administrator; password entered privately')
    parser.add_argument('--display-name', default='研究管理员')
    args = parser.parse_args()
    if not (args.store / 'neuroei.sqlite3').exists() and not args.init_admin:
        parser.error('Initialize a new workspace explicitly with --init-admin')
    store = Store(args.store.resolve())
    if args.init_admin:
        if accounts.accounts(store):
            parser.error('This workspace already has accounts; use the existing administrator')
        password = getpass.getpass('管理员密码（至少10位）: ')
        if password != getpass.getpass('再次输入密码: '):
            parser.error('两次密码不一致，未创建账号')
        accounts.create(store, args.init_admin, args.display_name, password, 'admin')
    server = make_server(store, args.port, legacy_execution_enabled=False)
    print(f'Workspace: http://{server.server_address[0]}:{server.server_address[1]}/', flush=True)
    print(f'Data: {store.root}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        store.close()


if __name__ == '__main__':
    main()
