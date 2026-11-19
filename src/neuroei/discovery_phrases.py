"""Bounded surface-form evidence, not synonym expansion or applicability proof."""
import re

PHRASE_POLICY = 'surface-evidence-v1'
# Explicit regular noun inflections only; no stemming of entities or technical verbs.
NOUNS = {'transformer', 'neuron', 'robot', 'horizon'}
NEGATION = re.compile(r"\b(?:no|non|not|never|without|neither|nor|cannot|lack|lacks|lacking|exclude|excludes|excluded|excluding|except)\b|n['’]t\b", re.IGNORECASE)
SEPARATOR = r'[\s\-‐‑–]+'


def phrase_evidence(title, abstract, phrase):
    """Return the first ordered contiguous match in one non-negated sentence.

    Permit one inserted 'with' for >=3-token phrases and four named noun plurals.
    Explicit connectors, numbers, acronyms and all other tokens stay unchanged.
    Negation anywhere in the sentence causes conservative abstention.
    """
    tokens = re.split(SEPARATOR, phrase.strip())
    if not tokens or any(not re.fullmatch(r'[A-Za-z0-9]+', t) for t in tokens):
        return None
    if len(tokens) == 1 and tokens[0].isupper():
        return None  # A bare acronym does not establish which expansion is meant.
    parts = []
    for token in tokens:
        singular = token.removesuffix('s')
        if token.isdigit():
            parts.append(r'(?<!\d[.,/\-])'+token+r'(?![.,/\-–]\d)')
        elif token.islower() and singular in NOUNS:
            parts.append('(?i:'+re.escape(singular)+'s?)')
        elif token.isupper() and len(token) > 1:
            parts.append(re.escape(token))
        else:
            parts.append('(?i:'+re.escape(token)+')')
    patterns = [SEPARATOR.join(parts)]
    if len(tokens) >= 3 and not any(t.casefold() == 'with' for t in tokens):
        for gap in range(1, len(parts)):
            patterns.append(SEPARATOR.join(parts[:gap])+SEPARATOR+'(?i:with)'+SEPARATOR+SEPARATOR.join(parts[gap:]))
    pattern = re.compile(r'(?<![\w])(?:'+'|'.join(patterns)+r')(?![\w])')
    for scope, text in (('title', title), ('abstract', abstract or '')):
        # Preserve offsets while protecting common abbreviation and decimal dots.
        segmented = re.sub(r'\b(?:e\.g\.|i\.e\.|etc\.|vs\.|fig\.|dr\.|et al\.)',
                           lambda m: m[0].replace('.', '\u2060'), text, flags=re.IGNORECASE)
        for sentence in re.finditer(r'(?:[^.!?\n]|(?<=\d)\.(?=\d))+(?:[.!?]|$)', segmented):
            original = text[sentence.start():sentence.end()]
            if NEGATION.search(original):
                continue
            hit = pattern.search(original)
            if hit:
                start, end = sentence.start()+hit.start(), sentence.start()+hit.end()
                return {'scope': scope, 'phrase': phrase, 'matched_text': text[start:end],
                        'start': start, 'end': end, 'text': original,
                        'sentence_start': sentence.start(), 'sentence_end': sentence.end(),
                        'match_policy': PHRASE_POLICY}
    return None
