"""Operator entry for authorized journal batches; no Feishu access or implicit calls."""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import accounts
from .config import LLMConfig
from .llm import FlashClient
from .project_understanding import FlashUnderstandingExtractor, JournalBatch
from .recommendation_pipeline import progress
from .store import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', required=True, type=Path)
    parser.add_argument('--batch', required=True, type=Path, help='Authorized exported JournalBatch JSON')
    parser.add_argument('--user-id', required=True)
    parser.add_argument('--project-id', required=True)
    parser.add_argument('--as-of', default=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
    parser.add_argument('--bootstrap', action='store_true')
    parser.add_argument('--validate-only', action='store_true', help='Validate the exported batch without recording an event')
    parser.add_argument('--max-model-calls', type=int, default=0, choices=range(101),
                        help='Zero checks updates and records status without models; positive enables paid understanding, no retries')
    args = parser.parse_args()
    if not (args.store / 'neuroei.sqlite3').is_file():
        parser.error('Existing account workspace required')
    if args.max_model_calls and not args.validate_only and not os.environ.get('DEEPSEEK_API_KEY'):
        parser.error('Explicit paid understanding requires DEEPSEEK_API_KEY')
    store = Store(args.store)
    client = None
    try:
        owner = accounts.find(store, args.user_id)
        if not owner or owner['disabled']:
            parser.error('An enabled account is required')
        allowed = accounts.projects_for(store, owner)
        if allowed is not None and args.project_id not in allowed:
            parser.error('Account has no access to this project')
        batch = JournalBatch.model_validate(json.loads(args.batch.read_text(encoding='utf-8')))
        if any((r.user_id, r.project_id) != (args.user_id, args.project_id) for r in batch.records):
            parser.error('Journal user/project scope mismatch')
        if args.validate_only:
            print(json.dumps({'status': 'validated_only', 'journal_records': len(batch.records), 'model_calls': 0}))
            return
        if args.max_model_calls:
            client = FlashClient(LLMConfig(max_calls_per_run=args.max_model_calls, request_attempts=1,
                                 schema_repair_attempts=0, max_tokens=32768), store,
                                 'understanding:'+datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        outcome = progress(store, batch, extractor=FlashUnderstandingExtractor(client) if client else None, sources=None,
                           user_id=args.user_id, project_id=args.project_id, projects=[args.project_id],
                           as_of=args.as_of, bootstrap=args.bootstrap, max_calls=args.max_model_calls)
        result = outcome['understanding']
        print(json.dumps({'status': result['status'], 'reason': result['reason'],
                          'review_queue_received': len(outcome.get('deliveries', [])),
                          'event_id': outcome.get('event', {}).get('event_id') if outcome.get('event') else None,
                          'model_calls': client.calls if client else 0}, ensure_ascii=False))
    finally:
        if client:
            client.close()
        store.close()


if __name__ == '__main__':
    main()
