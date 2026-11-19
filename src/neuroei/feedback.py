"""Separate explicit interest labels from novelty, applicability and report corrections."""
from datetime import datetime

from .models import FeedbackEvent


def feedback_channels(events: list[FeedbackEvent], *, project_id: str, user_id: str, cutoff: datetime):
    if cutoff.tzinfo is None:
        raise ValueError('Feedback cutoff needs timezone')
    eligible = [e for e in events if e.project_id == project_id and e.user_id == user_id and e.occurred_at < cutoff]
    eligible.sort(key=lambda e: e.occurred_at)
    explicit = {}
    other = {name: [] for name in ('later', 'known', 'method_unsuitable', 'fact_correction', 'report_quality')}
    for event in eligible:
        if event.action in ('relevant', 'irrelevant'):
            previous = explicit.get(event.paper_id)
            if previous and previous.occurred_at == event.occurred_at and previous.action != event.action:
                raise ValueError('Conflicting same-time interest feedback requires review')
            explicit[event.paper_id] = event
        else:
            other[event.action].append(event.model_dump(mode='json'))
    return {'project_id': project_id, 'user_id': user_id, 'cutoff_exclusive': cutoff.isoformat(),
            'eligible_event_count': len(eligible),
            'positive_paper_ids': sorted(pid for pid, e in explicit.items() if e.action == 'relevant'),
            'negative_paper_ids': sorted(pid for pid, e in explicit.items() if e.action == 'irrelevant'),
            'other_channels': other,
            'policy': 'Only explicit relevant/irrelevant affect topic interest; unselected papers produce no event',
            'upstream_update_applied': False}
