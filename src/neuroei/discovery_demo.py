"""python -m neuroei.discovery_demo --request ... [--live --scratch DIR]."""
import argparse
import json
from pathlib import Path

from .discovery import discover
from .discovery_sources import DiscoverySources
from .models import PaperRecord, SourceResult


class FixtureSource:
    """Synthetic transport fixture; never use as scientific-quality evidence."""
    def collect(self, name, query, since, until):
        return SourceResult(source=name, query=query, since=since, until=until, status='ok', papers=[
            PaperRecord(paper_id='arxiv:2303.04137', version='v1', title='Diffusion Policy',
                        abstract='', published='2023-03-07T18:00:00Z', sources=['arxiv'],
                        url='https://arxiv.org/abs/2303.04137')])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--live', action='store_true', help='free public metadata; CPU ranking only with explicit --model-path')
    parser.add_argument('--scratch', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model-path', type=Path, help='explicit already provisioned CPU model; no downloads')
    parser.add_argument('--model-revision')
    parser.add_argument('--aliases', type=Path)
    parser.add_argument('--embedding-audit-ref')
    parser.add_argument('--abstract-review', type=Path)
    args = parser.parse_args()
    if args.live and not args.scratch:
        parser.error('--live requires an isolated --scratch directory for arXiv coordination')
    if args.model_path and (not args.live or not args.model_revision or not args.aliases or not args.embedding_audit_ref):
        parser.error('--model-path requires --live, --model-revision, --aliases, --embedding-audit-ref')
    request = json.loads(Path(args.request).read_text(encoding='utf-8'))
    sources = DiscoverySources(max_results=20) if args.live else FixtureSource()
    if args.live:
        sources.coordination_dir = args.scratch
    ranker = None
    if args.model_path:
        import torch

        from .discovery_paperflow import PaperFlowSemanticRanker
        from .readers import bge_cpu
        torch.set_num_threads(2)
        encoder = bge_cpu(str(args.model_path.resolve(strict=True)))
        encoder.max_seq_length = 1024
        ranker = PaperFlowSemanticRanker(encoder, json.loads(args.aliases.read_text(encoding='utf-8')),
            model_revision=args.model_revision, embedding_audit_ref=args.embedding_audit_ref, max_calls=1)
    review = json.loads(args.abstract_review.read_text(encoding='utf-8')) if args.abstract_review else None
    try:
        result = discover(request, sources, source_names=('arxiv',), ranker=ranker, abstract_review=review)
        output = result.model_dump(mode='json')
        output['audit']['verification_kind'] = ('real_public_sources_cpu_semantic' if ranker else
            'real_public_sources_no_model' if args.live else 'synthetic_transport_no_model')
        encoded = json.dumps(output, ensure_ascii=False, indent=2)
        if args.output:
            args.output.write_text(encoded, encoding='utf-8')
        else:
            print(encoded)
    finally:
        if args.live:
            sources.close()


if __name__ == '__main__':
    main()
