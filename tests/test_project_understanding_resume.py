"""Recover a durable import checkpoint using the next normal project analysis request."""
from contextlib import closing

import pytest

from neuroei.project_understanding import JournalBatch, identity, understand
from neuroei.store import Store


FIRST = '2026-11-19T12:00:00+08:00'
LATER = '2026-11-20T12:00:00+08:00'


def journal_batch(stamp, first_revision='1'):
    return JournalBatch.model_validate({'records': [
        {'user_id': 'owner', 'project_id': 'project', 'source_id': source,
         'record_date': '2026-11-18', 'available_at': stamp, 'synced_at': stamp,
         'blocks': [{'document_id': source, 'block_id': source + '-text',
                     'revision': first_revision if source == 'journal-a' else '1',
                     'text': '采集样本。补做对照后，再次记录：采集样本。'}]}
        for source in ('journal-a', 'journal-b', 'journal-c')
    ], 'source_runs': [{'source_id': 'feishu-folder:example', 'status': 'read'}]})


class Extractor:
    audit_ref = 'fixture-audit'
    extraction_settings = {'model': 'fixture'}

    def __init__(self, interrupt=None):
        self.calls = []
        self.interrupt = interrupt

    def plan_chunks(self, records, prior_items, chunk_chars):
        return [[record] for record in records]

    def extract(self, payload):
        journal, = payload['journals']
        source = journal['source_id']
        self.calls.append(source)
        self.provider_audit = {'status': 'fixture', 'source': source}
        if source == self.interrupt:
            raise SystemExit('simulate process interruption before saving current batch')
        if source == 'journal-b':
            assert payload['prior_items'][0]['source_refs'][0]['source_id'] == 'journal-a'
        return {'items': [{'id':'fact', 'kind':'method', 'text':'完成采集',
            'epistemic':'recorded', 'conditions':[], 'occurred_on':None,
            'citations':[{'source_id':source, 'block_id':source+'-text', 'quote':'采集样本。'}]}],
            'needs':[], 'reviews':[{'source_id':source,'decision':'research_change','reason':'记录采集步骤'}],
            'warnings':[]}

    def citation_location(self, item_id, citation, source):
        text = source.blocks[0].text
        start = text.rindex(citation.quote)
        return {'quote_char_start': start, 'quote_char_end': start + len(citation.quote)}


@pytest.mark.parametrize('first_revision, expected_calls', [
    ('1', ['journal-b', 'journal-c']),
    ('2', ['journal-a', 'journal-b', 'journal-c']),
])
def test_resumes_only_valid_batches_and_keeps_exact_citations(tmp_path, first_revision, expected_calls):
    def run(store, extractor, stamp, revision):
        return understand(store, extractor, journal_batch(stamp, revision), user_id='owner',
                          project_id='project', as_of=stamp, imported_at=stamp)

    with closing(Store(tmp_path)) as store:
        original = Extractor(interrupt='journal-b')
        with pytest.raises(SystemExit):
            run(store, original, FIRST, '1')
        assert original.calls == ['journal-a', 'journal-b']
        assert not store.items('understanding_head')
        (run_id, resume), = store.items('understanding_resume')
        first_artifact, = resume['batches'].values()
        assert first_artifact['artifact'] == run_id + ':0'

    with closing(Store(tmp_path)) as reopened:
        extractor = Extractor()
        result = run(reopened, extractor, LATER, first_revision)
        assert result['status'] == 'completed' and result['run_id'] == run_id
        assert extractor.calls == expected_calls and result['model_calls'] == len(expected_calls)
        assert result.get('resumed_batches', 0) == 3 - len(expected_calls)
        assert len(result['provider_audit_refs']) == 3
        head = reopened.get('understanding_head', identity(['owner', 'project']))
        refs = [item['source_refs'][0] for item in reopened.get('understanding_state', head['key'])['items']]
        assert [ref['source_id'] for ref in refs] == ['journal-a', 'journal-b', 'journal-c']
        assert refs[0]['quote_char_start'] == len('采集样本。补做对照后，再次记录：')
        assert refs[0]['locator']['revision'] == first_revision


def test_fully_saved_batches_can_finish_after_repeated_interruption(tmp_path):
    def run(store, extractor, should_stop=None):
        return understand(store, extractor, journal_batch(FIRST), user_id='owner',
                          project_id='project', as_of=FIRST, imported_at=FIRST, should_stop=should_stop)

    def interrupt_before_publication():
        checkpoints = iter((False, False, False))

        def stop():
            try:
                return next(checkpoints)
            except StopIteration:
                raise SystemExit('simulate interruption before publishing project state') from None

        return stop

    with closing(Store(tmp_path)) as store:
        first = Extractor()
        second = Extractor()
        for extractor in (first, second):
            with pytest.raises(SystemExit):
                run(store, extractor, interrupt_before_publication())
        assert len(first.calls) == 3 and not second.calls
        result = run(store, Extractor())
        assert result['status'] == 'completed'
        assert result['resumed_batches'] == 3 and result['model_calls'] == 0
