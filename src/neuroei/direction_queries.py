"""Bounded translation of explicitly public topic text; private descriptions stay private."""
import hashlib
import json
import re
from typing import Annotated

from pydantic import Field, StringConstraints, field_validator

from .config import StrictModel
from .directions import optional
from .models import now

SEARCH_POLICY_VERSION = 'direction-search-v4'  # v4: strict bioRxiv terms/dates and English shortlist queries
Query = Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=120)]


class QueryPlanningFailed(ValueError):
    pass


class DirectionChanged(ValueError):
    pass


class EnglishQueries(StrictModel):
    queries: list[Query] = Field(min_length=1, max_length=3)

    @field_validator('queries')
    @classmethod
    def public_english_phrases(cls, values):
        for value in values:
            if not value.isascii() or not re.search(r'[a-zA-Z]{2}', value):
                raise ValueError('English public search terms required')
            if len(value.split()) > 6 or re.search(r'[^a-zA-Z0-9 \-()/]', value):
                raise ValueError('Use short plain search phrases')
        return list(dict.fromkeys(values))


def resolve_direction_queries(client, store, direction):
    # Only text already declared for public search is used. Neither description,
    # research questions, lab facts nor user identity are included in the prompt.
    provided = direction['public_queries'] or [direction['title']]
    basis = {'policy': SEARCH_POLICY_VERSION, 'public_text': provided}
    digest = hashlib.sha256(json.dumps(basis, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]
    key = direction['direction_id'] + ':' + digest
    cached = optional(store, 'direction_query_plan', key)
    if cached:
        return {**cached, 'reused': True}
    if all(text.isascii() and re.search(r'[a-zA-Z]{2}', text) for text in provided):
        queries = list(dict.fromkeys(provided))
        origin = 'explicit_public_queries' if direction['public_queries'] else 'english_direction_title'
    else:
        prompt = {'public_topic_text': provided, 'instruction':
            'Translate the provided public research topics into 1-3 concise English academic search phrases. '
            'Use 2-4 words where possible, at most 6 words per phrase. Include a broad core phrase and '
            'at most two genuinely related variants. Preserve the topic, avoid invented assumptions, '
            'institution names, dates, private background or exact-paper titles. '
            'Plain ASCII phrases only; no boolean operators or field syntax. Return JSON queries.'}
        try:
            queries = client.structured(json.dumps(prompt, ensure_ascii=False), EnglishQueries,
                                        '公开检索词规划').queries
        except DirectionChanged:
            raise
        except (RuntimeError, ValueError, TypeError):
            raise QueryPlanningFailed('Could not generate valid public English queries') from None
        origin = 'translated_public_topic'
    record = {'plan_id': key, 'policy_version': SEARCH_POLICY_VERSION, 'public_text': provided,
              'queries': queries, 'origin': origin, 'created_at': now(),
              'private_description_used': False, 'private_questions_used': False}
    store.put('direction_query_plan', key, record, immutable=True)
    return {**record, 'reused': False}


def search_result_summary(sources, candidate_count, recommendation_count):
    failed = [s for s in sources if s['status'] == 'failed']
    incomplete = [s for s in sources if s['status'] in ('partial', 'failed')]
    if sources and len(failed) == len(sources):
        return 'search_failed', '所有来源检索均失败；未进入论文分析，不能据此判断没有相关研究。'
    if not candidate_count:
        if incomplete:
            return 'search_incomplete', '检索覆盖不完整且没有取得有效候选；不能判断本方向没有新进展。'
        return 'no_candidates', '本次来源及日期窗口内未取得候选；不代表整个领域没有相关论文。'
    if not recommendation_count:
        return 'no_recommendations', f'取得{candidate_count}篇候选，摘要筛选后暂无达到相关性阈值的推荐；未做全文分析。'
    return 'recommendations_ready', f'取得{candidate_count}篇候选，保留{recommendation_count}篇推荐；全文完成情况见阅读状态。'


ERROR_MESSAGES = {
    'QueryPlanningFailed': '英文检索词生成失败，尚未完成有效检索；请查看失败记录。',
    'ModelFailure': '模型调用未完成，请检查模型服务及凭据；这不表示没有相关论文。',
    'DirectionChanged': '方向已暂停或修改，已在安全检查点停止。',
    'NoEnabledSources': '尚未启用公开来源，未进行检索。',
}
