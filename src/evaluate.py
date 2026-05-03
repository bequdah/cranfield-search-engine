import math
from typing import List, Dict, Set


def compute_ap(retrieved_doc_ids: List[int], relevant_doc_ids: Set[int]) -> float:
    """
    Compute Average Precision (AP) for a single query.
    """
    if not relevant_doc_ids:
        return 0.0

    hits = 0
    sum_precisions = 0.0

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            hits += 1
            sum_precisions += hits / rank

    return sum_precisions / len(relevant_doc_ids)


def compute_map(all_retrieved: Dict[int, List[int]], qrels: List[dict]) -> float:
    """
    Compute Mean Average Precision (MAP) using binary relevance.

    Cranfield project rule:
    relevance 1,2,3,4 => relevant
    missing or -1      => non-relevant
    """
    relevant_by_query: Dict[int, Set[int]] = {}

    for qrel in qrels:
        q_id = qrel["query_id"]
        d_id = qrel["doc_id"]
        rel = qrel["relevance"]

        if rel in {1, 2, 3, 4}:
            if q_id not in relevant_by_query:
                relevant_by_query[q_id] = set()
            relevant_by_query[q_id].add(d_id)

    aps = []
    all_query_ids = set(all_retrieved.keys()) | set(relevant_by_query.keys())

    for q_id in all_query_ids:
        relevant_docs = relevant_by_query.get(q_id, set())
        retrieved = all_retrieved.get(q_id, [])
        aps.append(compute_ap(retrieved, relevant_docs))

    return sum(aps) / len(aps) if aps else 0.0


def convert_relevance(raw_rel: int) -> int:
    """
    Convert Cranfield graded relevance for NDCG:
    1 -> 4
    2 -> 3
    3 -> 2
    4 -> 1
    others -> 0
    """
    mapping = {1: 4, 2: 3, 3: 2, 4: 1}
    return mapping.get(raw_rel, 0)


def compute_dcg(relevances: List[int]) -> float:
    """
    Compute DCG using:
    DCG = sum(rel_i / log2(i + 1)), with rank starting at 1
    """
    dcg = 0.0
    for i, rel in enumerate(relevances, start=1):
        dcg += rel / math.log2(i + 1)
    return dcg


def compute_ndcg(all_retrieved: Dict[int, List[int]], qrels: List[dict], k: int = 10) -> float:
    """
    Compute mean NDCG@k using graded relevance.
    """
    true_rel_by_query: Dict[int, Dict[int, int]] = {}

    for qrel in qrels:
        q_id = qrel["query_id"]
        d_id = qrel["doc_id"]
        raw_rel = qrel["relevance"]

        if q_id not in true_rel_by_query:
            true_rel_by_query[q_id] = {}

        true_rel_by_query[q_id][d_id] = convert_relevance(raw_rel)

    ndcg_scores = []
    all_query_ids = set(all_retrieved.keys()) | set(true_rel_by_query.keys())

    for q_id in all_query_ids:
        doc_relevances = true_rel_by_query.get(q_id, {})
        retrieved = all_retrieved.get(q_id, [])[:k]

        retrieved_rels = [doc_relevances.get(doc_id, 0) for doc_id in retrieved]
        dcg = compute_dcg(retrieved_rels)

        ideal_rels = sorted([r for r in doc_relevances.values() if r > 0], reverse=True)[:k]
        idcg = compute_dcg(ideal_rels)

        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0


if __name__ == "__main__":
    sample_qrels = [
        {"query_id": 1, "doc_id": 10, "relevance": 1},
        {"query_id": 1, "doc_id": 20, "relevance": 2},
        {"query_id": 1, "doc_id": 30, "relevance": 4},
    ]

    sample_retrieved = {
        1: [20, 10, 99, 88]
    }

    print("MAP:", compute_map(sample_retrieved, sample_qrels))
    print("NDCG@10:", compute_ndcg(sample_retrieved, sample_qrels, k=10))
