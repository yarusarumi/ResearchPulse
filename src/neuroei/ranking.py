"""Deterministic ranking/evaluation primitives; no embeddings, labels or model calls invented."""
import math

from rank_bm25 import BM25Okapi

from .research import tokenize


def cosine(left, right):
    if not left or len(left) != len(right) or not all(math.isfinite(x) for x in [*left, *right]):
        raise ValueError("Embedding dimensions/values invalid")
    ln = math.sqrt(sum(x*x for x in left))
    rn = math.sqrt(sum(x*x for x in right))
    if not ln or not rn:
        raise ValueError("Zero embedding is not semantic evidence")
    return sum(a*b for a, b in zip(left, right)) / (ln*rn)


def hybrid_recall(papers, queries, paper_vectors, query_vectors, *, limit=30, rrf_k=60):
    if not papers or not queries or len(papers) != len(paper_vectors) or len(queries) != len(query_vectors):
        raise ValueError("Missing/misaligned hybrid retrieval inputs")
    ids = [p.paper_id for p in papers]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate retrieval paper IDs")
    bm25 = BM25Okapi([tokenize(p.title + " " + p.abstract) for p in papers])
    lexical = bm25.get_scores(tokenize(" ".join(queries))).tolist()
    semantic = [max(cosine(v, q) for q in query_vectors) for v in paper_vectors]
    lists = [sorted(range(len(ids)), key=lambda i: (-values[i], ids[i])) for values in (lexical, semantic)]
    ranks = [{index: rank for rank, index in enumerate(order, 1)} for order in lists]
    rows = [{"paper_id": pid, "bm25": float(lexical[i]), "bge_cosine_max": semantic[i],
             "bm25_rank": ranks[0][i], "bge_rank": ranks[1][i],
             "rrf_score": sum(1/(rrf_k + rank[i]) for rank in ranks)} for i, pid in enumerate(ids)]
    rows.sort(key=lambda x: (-x["rrf_score"], x["paper_id"]))
    return rows[:limit], rows


def relevance_metrics(ranked_ids, labels, *, k=5):
    """All candidate labels must be explicit. Recall concerns only this fixed labelled pool."""
    if type(k) is not int or k < 1:
        raise ValueError("Cutoff k must be a true integer of at least 1")
    if not isinstance(labels, dict):
        raise ValueError("Relevance labels must be a mapping of paper IDs")  # noqa: TRY004
    if not isinstance(ranked_ids, (list, tuple)) or not ranked_ids:
        raise ValueError("Ranking candidates must be a non-empty list")
    ids = list(ranked_ids)
    if any(type(pid) is not str or not pid for pid in ids):
        raise ValueError("Ranking paper IDs must be non-empty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate ranking paper IDs")
    # Validate every value first: an explicit null must not hide bool/str/float labels.
    for value in labels.values():
        if value is not None and (type(value) is not int or not 0 <= value <= 3):
            raise ValueError("Relevance labels must be null or integers 0–3")
    if not set(ids) <= labels.keys():
        raise ValueError("Ranking lists paper IDs outside the labelled pool")
    if any(value is None for value in labels.values()):
        return {"status": "awaiting_human_labels", "metrics": None,
                "labelled_count": sum(value is not None for value in labels.values()),
                "candidate_count": len(labels)}
    top = ids[:k]
    relevant = sum(value >= 2 for value in labels.values())
    hits = sum(labels[pid] >= 2 for pid in top)
    dcg = sum((2**labels[pid]-1)/math.log2(i+2) for i, pid in enumerate(top))
    ideal = sum((2**value-1)/math.log2(i+2) for i, value in enumerate(sorted(labels.values(), reverse=True)[:k]))
    return {"status": "labelled_pool_only", "labelled_count": len(labels), "candidate_count": len(labels),
            "metrics": {f"precision_at_{k}": hits/k, f"recall_at_{k}_within_labelled_pool": hits/relevant if relevant else None,
                        f"ndcg_at_{k}": dcg/ideal if ideal else None},
            "short_list_policy": "Precision denominator remains k; unfilled recommendation slots count as misses"}
