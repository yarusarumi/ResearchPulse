"""Source-bound daily/weekly scientific synthesis from already checked claims."""
import hashlib
import json
import re
from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from .config import StrictModel
from .evidence_selection import setup_signals
from .experiment_design import ExperimentDesign, proposal_issues, render_design

NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
BASIS_FIELDS = {'hypothesis_id', 'hypothesis_prediction', 'paper_fact_ref', 'counter_result', 'comparable_conditions'}


class ChallengeBasis(StrictModel):
    """Structured, checkable counter-evidence for a project-hypothesis challenge."""
    hypothesis_id: NonBlankStr
    hypothesis_prediction: NonBlankStr
    paper_fact_ref: NonBlankStr
    counter_result: NonBlankStr
    comparable_conditions: NonBlankStr


class Insight(StrictModel):
    category: Literal['conditional_result', 'open_question', 'transferable_method', 'hypothesis_challenge', 'experiment_proposal']
    text: str
    claim_refs: list[str] = Field(min_length=1)
    question_ids: list[str] = Field(min_length=1)
    conditions_and_limits: str
    hypothesis_ids: list[str] = Field(default_factory=list)
    challenge_basis: ChallengeBasis | None = None
    experiment_scope: dict[str, str] = Field(default_factory=dict)
    method_steps: list[str] = Field(default_factory=list, max_length=8)
    experiment_design: ExperimentDesign | None = None


class Synthesis(StrictModel):
    insights: list[Insight] = Field(max_length=12)
    evidence_gaps: list[str]


def _gap_key(text):
    """Normalize a limitation sentence for exact-duplicate suppression only."""
    return re.sub(r"[\s，。；：、,.;:!?！？（）()【】\[\]“”\"'`]+", "", str(text or "")).casefold()


def _gap_stated(gap, insights):
    """A gap already written into an insight's conditions or scope need not be repeated."""
    key = _gap_key(str(gap or '').rstrip('。.'))
    if not key:
        return True
    for item in insights:
        fields = [item.get('conditions_and_limits', '')]
        fields.extend((item.get('experiment_scope') or {}).values())
        if any(key in _gap_key(field) for field in fields):
            return True
    return False


def method_consistency_issues(item):
    """Catch explicit known/unknown contradictions, not arbitrary scientific claims."""
    steps = '\n'.join(item.get('method_steps') or [])
    limits = item.get('conditions_and_limits', '')
    for sentence in re.split(r'[。；;]', limits):
        if not re.search(r'未读取|未知|未取得', sentence):
            continue
        for parameter in ('基函数中心', '标准差', 'Σ_w'):
            if parameter in sentence and re.search(re.escape(parameter) + r'.{0,10}(?:为|=|取|设)', steps):
                return ['方法步骤给出了限制中标为未读取的参数；须核对来源并统一说明。']
    return []


def synthesize(client, profile, entries, period, *, context_kind='project', compact=False):
    from .synthesis_audit import evidence_context
    claims, reading_context = {}, []
    for entry in entries:
        a = entry['analysis']
        source_context = evidence_context(a)
        for evidence in a.get('evidence', []):
            if evidence.get('evidence_id') in source_context:
                source_context[evidence['evidence_id']]['file'] = evidence.get('file', '')
        paper = entry['paper']
        setup_spans = []
        for evidence in a.get('evidence', []):
            text = evidence.get('text') or ''
            signals = setup_signals(text)
            if signals:
                setup_spans.append({'block_id': evidence.get('block_id') or evidence.get('evidence_id'),
                                    'section': evidence.get('section', ''), 'signals': signals,
                                    'text': text[:900]})
        reading_context.append({'paper_id': paper.paper_id, 'version': paper.version,
            'analysis_date': entry.get('analysis_date'), 'reading_reused': entry.get('reading_reused'),
            'published': paper.published, 'updated': paper.updated,
            'available_blocks': a.get('available_blocks'), 'selected_blocks': a.get('selected_blocks'),
            'selected_tables': sum(' | ' in e.get('text', '') for e in a.get('evidence', [])),
            'unanswered': a.get('unanswered', []), 'reading_limits': a.get('limitations', []),
            'setup_spans': setup_spans[:6],
            'accepted_claims': len(a['accepted_claims']), 'rejected_claims': a.get('rejected_count', 0)})
        for item in a['accepted_claims']:
            ref = a['run_id']+'#claim-'+str(item['verification']['claim_index'])
            claims[ref] = {'claim': item['claim'], 'paper_id': a['paper_id'], 'version': a['paper_version'],
                           'paper_title': paper.title, 'url': paper.url,
                           'published': paper.published, 'updated': paper.updated,
                           'analysis_date': entry.get('analysis_date'), 'reading_reused': entry.get('reading_reused'),
                           'source_context': [source_context[eid] for eid in item['claim'].get('evidence_ids', [])
                                              if eid in source_context],
                           'source_spans': item.get('grounding', {}).get('supports', [])}
    if not claims:
        return {'insights': [], 'evidence_gaps': ['没有可用的已检查全文主张，不能生成科研判断。'], 'claim_lookup': {}}
    question_ids = {f.id for f in profile.facts if f.kind=='question'}
    payload = {'period': period, 'project_facts': [f.model_dump() for f in profile.facts], 'claims': claims,
        'current_reading_context': reading_context,
        'instruction': '以中文写研究情报。所有条目是待审阅的综合判断，必须引用给定claim_refs和项目question_ids。'
        '回答哪些问题仅在论文实验条件内取得进展，哪些仍开放，方法如何迁移，哪些证据可能挑战项目假设。'
        '论文内条件进展不等于本项目问题已解决；项目问题在项目数据/任务复现前仍开放，不得把“论文里有效”说成“项目已解决”。'
        '不同任务/数据/基线结果不能直接拼成矛盾。没有项目假设证据时明确不足。'
        '每条conditional_result只讨论一个实验；experiment_scope须分别写experiment、dataset、'
        'replicates、uncertainty、budget_control，不能从另一实验借条件。先读current_reading_context中的'
        'setup_spans与所引source_context原文：若已给出训练种子数、数据量、独立重复、误差口径、基线或预算，'
        '必须写入对应字段并注明原文口径；只有确实未取得的字段才写本次未读取，并指明缺少哪一项。'
        '不得写论文未报告；不得把种子数等计数直接解释成统计显著。source_context的原表caption优先于历史摘要和unanswered。'
        '训练种子数与按任务bootstrap的样本数/标准误必须分别解释。'
        'hypothesis_challenge必须填写hypothesis_ids，且只能引用kind=hypothesis的项目事实；研究目标/问题不是既有假设。'
        '假设挑战至少引用一条kind=paper_fact的自动检查保留主张，不能把模型自己的实验建议或类比单独当作反证；'
        '还必须填写challenge_basis五个非空字段：hypothesis_id在本条hypothesis_ids与项目真实假设中，'
        'paper_fact_ref在本条claim_refs且对应kind=paper_fact，hypothesis_prediction与counter_result具体到引用事实/适用条件，'
        'comparable_conditions说明两者可比条件。没有匹配条件下的反证只写open_question或evidence_gaps，不得为凑类别或8条强行写挑战。'
        'open_question仅说明本次证据尚不能回答，不能断言整个领域尚无人解决。'
        '论文发表/更新日期以给定元数据为准；新见、新读与新发表分别叙述。'
        'reading_reused=true的论文必须写沿用既有分析，不能写本次读取；analysis_date未知就写原阅读日期未知。'
        '方法步骤与限制逐项一致：已在步骤给出的参数不能同时写未读取；来源不支持的参数从步骤删去并明确未知。'
        'previous_progress中的缺口仅是历史待复核记录，不得原样当作本期事实；'
        'accepted_claims只表示自动检查保留，未经过逐项人工核对，不得称已核验、已证实或独立确认。'
        '已取得结果表却无自动检查保留的定量主张时，应说定量解读尚未通过自动检查，不能声称未取得结果表或论文没有该结果。'
        '相同的方法启发与其2×2实验建议应合并或精简，避免同义重复。'
        '最多8条；实验建议最多2条，包含对照、指标、失败判据和前提；不启动实验。'
        '实验建议须预先定义主要指标、最小实际有意义差异、配对单位和差值区间；'
        '区分支持改善、证据不足、支持无实际收益。不能用两组CI是否重叠、方差覆盖差异判定失败。'
        '训练种子为独立重复时，差值区间按配对种子汇总抽样或分层bootstrap（先种子再episode）；'
        '不能把同一种子内episode视作独立训练重复。仅固定模型的episode区间必须明确不外推到训练随机性。'
        '等重建质量与等计算预算必须是两组独立对照，不把二者同时强行固定。'
        'experiment_proposal必须填写experiment_design：primary_metric只选一个主要指标，'
        'improvement_direction为higher_is_better或lower_is_better，minimum_effect未获用户设置时写待使用者预注册，'
        'pairing_unit说明训练重复与评估样本的配对层级，uncertainty_method说明差值区间的抽样单位，'
        'comparison_regime只能选一种比较口径；不能根据论文数值捏造使用者的收益阈值。'
        '实验建议必须先说明要检验哪个问题；primary_control 的身份只能来自本条所引来源的实验条件或方法证据，'
        '来源未说明时写未知，不得用通用配方（如同一生成器、随机示范、已有示范）代替来源。'
        '数据量、示范数量、训练预算等匹配约束与对照身份分开写，匹配约束不构成另一个主对照；'
        '只能有一个 primary_control，其余对照写入 secondary_controls 并说明各自检验的问题，'
        '正文不得把另一个对照写成主比较。'
        '主要指标的名称、定义与改善方向必须与该篇来源一致，来源未给出定义时写未知，不得套用其他论文的口径。'
        '实验建议text只写问题、对照和动机，不另写判读规则，判定由结构化设计统一生成。'
        '固定updates/训练步数仅属于equal_training_updates，不等于equal_compute_budget；'
        '等计算预算需按FLOPs或固定硬件耗时定义。'
        'transferable_method用method_steps给出中文注释的高层伪代码：输入、训练目标、冻结模块和输出；'
        '每步受所引主张支持，未知实现细节明确未知，不虚构超参或代码。'
        '自定义指标必须在 method_steps 中按来源写明：输入与动作/数据的对齐方式、被比较对象、'
        '逐项或逐维的计算方式、聚合与加权方式；来源未给出的实现细节写未知，不得按其他论文的默认实现补全。'
        '任何一致性分数都不能单独证明学到忠实动力学。'
        '不补写新的数值，不称整个领域问题已解决，不将历史项目资料说成当前运行状态；阅读摘要不足不能补成全文结论。'}
    if context_kind == 'direction':
        payload['instruction'] += ('当前含使用者自主设置的关注方向；只有project:前缀条目来自关联项目的已启用档案，'
            '其他设置不是科研观察。关联项目条目保留来源日期和历史属性，不代表当前实验状态；围绕方向梳理方法路线、'
            '实际证据、争议及下一步跟进。结合previous_progress区分新见材料、已有材料的深化与重复内容，'
            '没有新证据时明确说明；不能把个人兴趣当科研假设或声称已改变项目结论。'
            'new_paper_ids只表示历史候选记录中未见的版本；first_recommended_paper_ids只表示本方向首次推荐，'
            '不能混称首次发现或新发表。evidence_gaps只写具体阅读缺口，不重复模板免责声明，'
            '也不重复conditions_and_limits或experiment_scope中已写明的同一缺口；同一限制只写一次，放在最相关的位置。'
            '不输出kind=hypothesis等实现字段，不凭空讨论本项目数据/项目假设；'
            '若未声明研究条件，迁移建议写若你的研究具备这些条件。')
    if compact:
        payload['instruction'] += ('上次输出截断。本次精简恢复：最多2条综合判断，不写实验建议或伪代码；'
            '每条只讨论一个事实，正文与限制各不超过80字，证据缺口最多2条。'
            '保留完整JSON结构、真实引用和必要实验条件；不能省略条件来凑短。')
    result = client.structured(json.dumps(payload, ensure_ascii=False), Synthesis, '跨论文科研情报综合')
    for item in result.insights:
        if not set(item.claim_refs)<=claims.keys() or not set(item.question_ids)<=question_ids:
            raise ValueError('Synthesis referenced nonexistent claim/project question')
    from .synthesis_audit import absence_scope_issues, audit_scope, lrc_method_issues
    clean, rejected = [], []
    for item in result.model_dump()['insights']:
        contexts = [c for ref in item['claim_refs'] for c in claims[ref].get('source_context', [])]
        reasons = audit_scope(item['text'] + ' ' + item['conditions_and_limits'], contexts)
        reasons += absence_scope_issues(item['text']) + absence_scope_issues(item['conditions_and_limits'])
        for field in item['experiment_scope'].values():
            reasons += audit_scope(str(field), contexts) + absence_scope_issues(str(field))
        reasons += proposal_issues(item)
        if item['category'] == 'transferable_method':
            reasons += lrc_method_issues(item)
            reasons += method_consistency_issues(item)
        reasons = list(dict.fromkeys(reasons))
        if item['category'] == 'conditional_result' and not all(
                item['experiment_scope'].get(k) for k in
                ('experiment', 'dataset', 'replicates', 'uncertainty', 'budget_control')):
            reasons.append('实验条件未分项定位')
        if reasons:
            rejected.append({'insight': item, 'reason': '；'.join(reasons)})
        else:
            clean.append(item)
    return gate_synthesis({**result.model_dump(), 'insights': clean, 'excluded_insights': rejected,
                           'claim_lookup': claims},profile)


def _valid_challenge_basis(item, hypotheses, claim_lookup):
    basis = item.get('challenge_basis')
    if not isinstance(basis, dict) or set(basis) != BASIS_FIELDS:
        return False
    if any(not isinstance(basis[key], str) or not basis[key].strip() for key in BASIS_FIELDS):
        return False
    if basis['hypothesis_id'] not in item.get('hypothesis_ids', []) or basis['hypothesis_id'] not in hypotheses:
        return False
    ref = basis['paper_fact_ref']
    if ref not in item.get('claim_refs', []):
        return False
    return claim_lookup.get(ref, {}).get('claim', {}).get('kind') == 'paper_fact'


def gate_synthesis(result,profile):
    """Goals and unanswered questions must never become fabricated project hypotheses, and a
    hypothesis_challenge needs a structured comparable counter-evidence basis, not just absence of support."""
    hypotheses={f.id for f in profile.facts if f.kind=='hypothesis'}
    kept,excluded=[],list(result.get('excluded_insights',[]))
    for item in result['insights']:
        reason = None
        if item['category']=='hypothesis_challenge':
            if not item.get('hypothesis_ids') or not set(item['hypothesis_ids'])<=hypotheses:
                reason = 'No explicit sourced project hypothesis identity'
            elif not any(
                    result.get('claim_lookup',{}).get(ref,{}).get('claim',{}).get('kind')=='paper_fact'
                    for ref in item.get('claim_refs',[])):
                reason = 'No primary paper fact supports the hypothesis challenge'
            elif not _valid_challenge_basis(item, hypotheses, result.get('claim_lookup',{})):
                reason = 'No comparable counter-evidence basis for the hypothesis challenge'
        if reason:
            excluded.append({'insight':item,'reason':reason})
        else:
            kept.append(item)
    gaps=list(result['evidence_gaps'])
    # The same limitation often appears in conditions, experiment_scope and evidence_gaps.
    # Suppress only exact normalized duplicates; do not merge semantically different gaps.
    conditions_blob = '\u0001'.join(
        '\u0001'.join([_gap_key(item.get('conditions_and_limits', ''))] +
                      [_gap_key(value) for value in (item.get('experiment_scope') or {}).values()])
        for item in kept)
    unique_gaps, seen = [], set()
    for gap in gaps:
        key = _gap_key(gap)
        if not key or key in seen:
            continue
        seen.add(key)
        if key in conditions_blob:
            continue
        unique_gaps.append(gap)
    gaps = unique_gaps
    if any(e['reason']=='No explicit sourced project hypothesis identity' for e in excluded):
        note='尚未配置可核对的项目假设，或本条未对应已配置的假设；相关条目已排除，研究目标和问题不等于已确认假设。'
        if note not in gaps:
            gaps.append(note)
    if any(e['reason']=='No primary paper fact supports the hypothesis challenge' for e in excluded):
        note='未提供自动检查保留的论文事实支持相关假设挑战；模型自己的实验建议或类比不能单独作为反证。'
        if note not in gaps:
            gaps.append(note)
    if any(e['reason']=='No comparable counter-evidence basis for the hypothesis challenge' for e in excluded):
        note=('未提供完整的假设预测、论文事实、相反结果和可比实验条件；'
              '缺乏支持不等于反证，相关假设挑战已排除。')
        if note not in gaps:
            gaps.append(note)
    return {**result,'insights':kept,'excluded_insights':excluded,'evidence_gaps':gaps,
            'question_titles': {f.id: f.text for f in profile.facts if f.kind == 'question'},
            'has_project_hypotheses': bool(hypotheses),
            'verification':'source_identity_and_hypothesis_identity_checked_semantic_review_required'}


def render_synthesis(result, *, context_kind='project', profile=None):
    names = {'conditional_result':'在限定条件下的进展','open_question':'仍开放的问题',
             'transferable_method':'可借鉴方法','hypothesis_challenge':'可能挑战项目假设的证据','experiment_proposal':'待讨论实验'}
    content = ('\n## 关注方向研究进展（待审阅）\n\n' if context_kind == 'direction'
               else '\n## 项目科研情报综合（待审阅推断）\n\n')
    # Store serializes dictionaries with sorted keys: numbering must survive reloads.
    references = sorted(result['claim_lookup'])
    numbers = {ref: index + 1 for index, ref in enumerate(references)}
    titles = ({f.id: f.text for f in profile.facts if f.kind == 'question'} if profile is not None
              else result.get('question_titles', {}))
    has_hypotheses = (any(f.kind == 'hypothesis' for f in profile.facts) if profile is not None
                      else result.get('has_project_hypotheses'))
    if has_hypotheses is False:
        content += '当前未配置可核对的项目假设，因此不评估论文是否推翻项目假设；仍可围绕关注问题整理证据。\n\n'
    if not result['insights']:
        content += '- 本期没有通过自动检查的综合条目；保留记录与阅读报告供人工判断，不据此推断没有进展。\n\n'
    for item in result['insights']:
        content += f"- **{names[item['category']]}**：{item['text']}\n\n  条件与限制：{item['conditions_and_limits']}\n\n"
        scope = item.get('experiment_scope', {})
        if scope:
            labels = {'experiment':'实验／表格', 'dataset':'数据', 'replicates':'独立重复',
                      'uncertainty':'误差口径', 'budget_control':'对照条件'}
            conditions_key = _gap_key(item.get('conditions_and_limits', ''))
            parts = [f'{labels[k]}：{scope[k]}' for k in labels
                     if scope.get(k) and _gap_key(scope[k]) not in conditions_key]
            if parts:
                content += '  实验条件：'+'；'.join(parts)+'。\n\n'
        if item.get('method_steps'):
            content += '```text\n# 依据所引正文整理的高层流程，未经代码复现\n'
            content += '\n'.join(item['method_steps'])+'\n```\n\n'
        if item['category'] == 'experiment_proposal':
            if item.get('experiment_design'):
                content += render_design(item['experiment_design'])+'\n\n'
            else:
                content += '> 旧版建议缺少结构化实验设计，待补全主要指标、配对与差值区间；不能据此直接判定成败。\n\n'
        content += ('  关注问题：' if context_kind == 'direction' else '  项目问题：')+'；'.join(
            titles.get(qid, '未找到对应的问题标题') for qid in item['question_ids'])+'；依据：'
        content += '；'.join('[查看证据 '+str(numbers[ref])+'](#claim-'+hashlib.sha256(ref.encode()).hexdigest()[:12]+')'
                            for ref in item['claim_refs'])+'。\n\n'
        content += '<details><summary>追溯详情</summary>\n\n问题编号：' + ', '.join(
            '`' + qid + '`' for qid in item['question_ids']) + '\n\n</details>\n\n'
    gaps = [gap for gap in result['evidence_gaps'] if not _gap_stated(gap, result['insights'])]
    if gaps:
        content += '\n'.join('- 证据缺口：'+gap for gap in gaps)+'\n'
    content += ('\n### 综合判断的证据核查\n\n' if references else '')
    for ref in references:
        source = result['claim_lookup'][ref]
        content += '\n### claim-'+hashlib.sha256(ref.encode()).hexdigest()[:12]+'\n\n'
        content += source['paper_title']+'\n\n'+source['claim']['text']+'\n\n'
        spans = source.get('source_spans', [])
        for span in spans:
            content += '> '+span['quote'].replace('\n', '\n> ')+'\n\n'
        contexts = source.get('source_context', [])
        for context in contexts:
            anchor = ('page=' + str(context['page']) if context.get('page')
                      else context.get('block_id', ''))
            target = context.get('file') or source['url'].replace('/abs/', '/html/').replace('http:', 'https:')
            content += '[打开原文定位]('+target.split('#')[0]+'#'+anchor+')\n\n'
        if not contexts:
            content += '[打开官方论文]('+source['url']+')\n\n'
        content += '<details><summary>追溯详情</summary>\n\n追溯编号：`'+ref+'`\n\n</details>\n\n'
    return content
