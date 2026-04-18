import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def compute_log_tf(term_count: int) -> float:
    """
    SMART 'l' weighting:
    1 + log10(tf) if tf > 0, else 0
    """
    return 1.0 + math.log10(term_count) if term_count > 0 else 0.0


def compute_idf(N: int, df: int) -> float:
    """
    SMART 't' weighting:
    idf = log10(N / df)
    """
    return math.log10(N / df) if df > 0 else 0.0


class VectorSpaceModel:
    """
    Vector Space Model using SMART weighting: lnc.ltc

    Documents: lnc
        l = logarithmic tf
        n = no idf
        c = cosine normalization

    Queries: ltc
        l = logarithmic tf
        t = idf
        c = cosine normalization
    """

    def __init__(self) -> None:
        # term -> {doc_id: normalized_document_weight}
        self.inverted_index: Dict[str, Dict[int, float]] = defaultdict(dict)

        # doc_id -> L2 norm before normalization
        self.doc_lengths: Dict[int, float] = {}

        # term -> idf
        self.idf_dict: Dict[str, float] = {}

        # total number of documents
        self.N: int = 0

    def build_index(self, docs: Dict[int, List[str]]) -> None:
        """
        Build the inverted index for documents using lnc weighting.

        Input:
            docs = {doc_id: [token1, token2, ...]}
        """
        # Reset state
        self.inverted_index.clear()
        self.doc_lengths.clear()
        self.idf_dict.clear()
        self.N = len(docs)

        # Step 1: Count terms per document and document frequency
        doc_term_counts: Dict[int, Counter] = {}
        df_counts: Dict[str, int] = defaultdict(int)

        for doc_id, tokens in docs.items():
            term_counts = Counter(tokens)
            doc_term_counts[doc_id] = term_counts

            for term in term_counts:
                df_counts[term] += 1

        # Step 2: Compute IDF for queries (ltc)
        for term, df in df_counts.items():
            self.idf_dict[term] = compute_idf(self.N, df)

        # Step 3: Build document vectors with lnc
        # document weight = log-tf only, then cosine normalize
        for doc_id, term_counts in doc_term_counts.items():
            raw_doc_weights: Dict[str, float] = {}
            sum_squares = 0.0

            for term, count in term_counts.items():
                weight = compute_log_tf(count)   # l
                raw_doc_weights[term] = weight
                sum_squares += weight ** 2

            doc_norm = math.sqrt(sum_squares)
            self.doc_lengths[doc_id] = doc_norm

            if doc_norm == 0.0:
                continue

            # cosine normalization: c
            for term, weight in raw_doc_weights.items():
                normalized_weight = weight / doc_norm
                self.inverted_index[term][doc_id] = normalized_weight

    def get_top_k(self, query_tokens: List[str], k: int = 10) -> List[Tuple[int, float]]:
        """
        Retrieve top-k documents using cosine similarity with lnc.ltc.

        Query weighting = ltc:
            l = log tf
            t = idf
            c = cosine normalization

        Since document vectors are already normalized in the index,
        and query vector is normalized here, cosine similarity reduces
        to a dot product over shared terms.
        """
        if not query_tokens:
            return []

        # Step 1: Query term counts
        query_term_counts = Counter(query_tokens)

        # Step 2: Query weights with ltc
        query_weights: Dict[str, float] = {}
        sum_squares = 0.0

        for term, count in query_term_counts.items():
            if term not in self.idf_dict:
                continue

            log_tf = compute_log_tf(count)           # l
            idf = self.idf_dict[term]                # t
            weight = log_tf * idf
            query_weights[term] = weight
            sum_squares += weight ** 2

        query_norm = math.sqrt(sum_squares)
        if query_norm == 0.0:
            return []

        # Step 3: Normalize query vector (c)
        for term in query_weights:
            query_weights[term] /= query_norm

        # Step 4: Accumulate cosine scores
        scores: Dict[int, float] = defaultdict(float)

        for term, q_weight in query_weights.items():
            postings = self.inverted_index.get(term, {})
            for doc_id, d_weight in postings.items():
                scores[doc_id] += q_weight * d_weight

        # Step 5: Sort Results
        ranked_results = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return ranked_results[:k]


if __name__ == "__main__":
    # Tiny example
    docs = {
        1: ["car", "insurance", "auto", "insurance"],
        2: ["best", "car", "repair"],
        3: ["insurance", "claim", "policy", "car"],
    }

    query = ["best", "car", "insurance"]

    vsm = VectorSpaceModel()
    vsm.build_index(docs)
    results = vsm.get_top_k(query, k=3)

    print("Top results for query:", query)
    for doc_id, score in results:
        print(f"Doc {doc_id}: {score:.4f}")
