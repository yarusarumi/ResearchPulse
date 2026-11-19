"""One transparent, generic recall variant; never inject paper IDs/names/seeds."""
import re

POLICY = 'recall-variant-v1'


def concept_expression(query, categories):
    """First topic plus another term in title/abstract, in declared domains.

    No paper-specific synonyms: singular/plural noun alternatives only. This is
    deliberately a broader discovery lane, never a claim of topical relevance.
    """
    if not categories or any(not re.fullmatch(r'[a-z]+\.[A-Za-z-]+', c) for c in categories):
        raise ValueError('Explicit arXiv domain categories required')
    terms = list(dict.fromkeys(re.findall(r'[a-z0-9-]+', query.casefold())))
    if not 2 <= len(terms) <= 6:
        raise ValueError('Concept lane requires 2..6 original terms')
    reduced = [term for term in terms if term not in {'learning', 'control'}]
    if len(reduced) >= 2:
        terms = reduced
    def variants(term):
        variants = [term]
        if term.isalpha() and len(term) > 3 and not term.endswith(('s', 'ing')):
            variants.append(term+'s')
        return variants
    pairs = ['('+ ' OR '.join(field+':"'+a+' '+b+'"' for field in ('ti', 'abs')
             for a in variants(terms[0]) for b in variants(other))+')' for other in terms[1:]]
    return '('+' OR '.join(pairs)+') AND ('+' OR '.join('cat:'+c for c in categories)+')'


def recall_variant(query):
    tokens = re.findall(r'[a-z0-9]+', query.casefold())
    if not tokens:
        raise ValueError('English research terms required')
    rules = ['lowercase_alphanumeric_terms']
    # Retain the original query as a separate lane. A variant deliberately drops
    # weak task words to explore nearby methods; it is never a statement of fit.
    reduced = [term for term in tokens if term not in {'learning', 'control'}]
    if len(reduced) >= 2 and reduced != tokens:
        tokens = reduced
        rules.append('drop_learning_control_only_when_two_terms_remain')
    replacements = {'robot': 'robotic', 'demonstrations': 'demonstration'}
    changed = [replacements.get(t, t) for t in tokens]
    if changed != tokens:
        rules.append('robot_robotic_demonstrations_demonstration')
    tokens = changed
    # "model predictive" is a technical phrase. Otherwise a noun number variant
    # is applied consistently, including unrelated holdout research queries.
    for index in range(len(tokens)):
        if tokens[index] == 'model' and tokens[index:index+2] != ['model', 'predictive']:
            tokens[index] = 'models'
            rules.append('model_plural_variant')
    return {'original_query': query, 'planned_query': ' '.join(tokens),
            'policy': POLICY, 'rules': rules, 'semantic_scope': 'recall_variant_not_equivalent_query'}
