"""Frozen, explicitly bounded official proceedings scope, checked 2026-11-09."""
from zoneinfo import ZoneInfo

from .discovery_contract import RecommendationRequest

CATALOG_VERSION = 'official-proceedings-20261109-v1'
CATALOG = (
    ('official-rss:22', 2026, 'RSS', 'https://www.roboticsproceedings.org/rss22/'),
    ('official-rss:21', 2025, 'RSS', 'https://www.roboticsproceedings.org/rss21/'),
    ('official-rss:20', 2024, 'RSS', 'https://www.roboticsproceedings.org/rss20/'),
    ('official-rss:19', 2023, 'RSS', 'https://www.roboticsproceedings.org/rss19/'),
    ('pmlr:306', 2026, 'ICML', 'https://proceedings.mlr.press/v306/'),
    ('pmlr:267', 2025, 'ICML', 'https://proceedings.mlr.press/v267/'),
    ('pmlr:235', 2024, 'ICML', 'https://proceedings.mlr.press/v235/'),
    ('pmlr:202', 2023, 'ICML', 'https://proceedings.mlr.press/v202/'),
    ('neurips:2025', 2025, 'NeurIPS', 'https://proceedings.neurips.cc/paper_files/paper/2025/vol38-main-conference'),
    ('neurips:2024', 2024, 'NeurIPS', 'https://proceedings.neurips.cc/paper_files/paper/2024'),
    ('neurips:2023', 2023, 'NeurIPS', 'https://proceedings.neurips.cc/paper_files/paper/2023'),
)


def plan_official_sources(request):
    """Latest two known volumes per venue; bootstrap also includes 2023 anchors.

    No inferred 2026 NeurIPS, no all-history claim, no unknown paper date -> daily.
    """
    request = RecommendationRequest.model_validate(request)
    year = request.as_of.astimezone(ZoneInfo('Asia/Shanghai')).year
    chosen = []
    warnings = ['bounded_title_discovery_not_complete_venue_coverage']
    for venue in ('RSS', 'ICML', 'NeurIPS'):
        available = [row for row in CATALOG if row[2] == venue and row[1] <= year]
        selected = available[:2]
        if request.mode == 'direction_bootstrap':
            selected += [row for row in available if row[1] == 2023 and row not in selected]
        chosen.extend(selected)
    if request.mode == 'direction_daily':
        chosen = []
        warnings.append('official_proceedings_daily_publication_timestamp_unavailable')
    if year > 2026:
        warnings.append('official_catalog_requires_refresh')
    entries = [{'source': s, 'year': y, 'venue': v, 'index_url': u} for s, y, v, u in chosen]
    return {'schema_version': '1', 'catalog_version': CATALOG_VERSION,
        'request_id': request.request_id, 'user_id': request.user_id,
        'as_of': request.as_of.isoformat(), 'mode': request.mode,
        'source_names': [e['source'] for e in entries], 'volumes': entries,
        'queries': list(request.queries[:24]), 'detail_limit_per_volume': 5,
        'index_byte_limit': 20_000_000, 'detail_byte_limit': 1_000_000,
        'max_http_attempts': len(entries)*6, 'coverage': 'unknown', 'warnings': warnings}
