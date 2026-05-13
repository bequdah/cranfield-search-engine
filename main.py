import time
from pathlib import Path

from src.parser import parse_cran_docs, parse_cran_queries, parse_cran_qrels
from src.preprocess import preprocess_collection
from src.vsm import VectorSpaceModel
from src.evaluate import compute_map, compute_ndcg


def main():
    print("\n" + "=" * 50)
    print("   Cranfield Information Retrieval System   ")
    print("         Vector Space Model (lnc.ltc)        ")
    print("=" * 50)

    # Paths
    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir / "dataset"

    docs_file = dataset_dir / "cran.all.1400"
    queries_file = dataset_dir / "cran.qry"
    qrels_file = dataset_dir / "cranqrel"

    # [1] Parsing
    print("\n[1] Parsing dataset...")
    start = time.time()

    try:
        raw_docs = parse_cran_docs(docs_file)
        raw_queries = parse_cran_queries(queries_file)
        qrels = parse_cran_qrels(qrels_file)
    except FileNotFoundError as e:
        print(f"[ERROR] Dataset not found: {e}")
        return

    print(f"    Documents: {len(raw_docs)}")
    print(f"    Queries:   {len(raw_queries)}")
    print(f"    Qrels:     {len(qrels)}")
    print(f"    Time: {time.time() - start:.2f}s")

    # [2] Preprocessing
    print("\n[2] Preprocessing...")
    start = time.time()

    clean_docs = preprocess_collection(raw_docs)
    clean_queries = preprocess_collection(raw_queries)

    print(f"    Time: {time.time() - start:.2f}s")

    # [3] Indexing
    print("\n[3] Building VSM (lnc.ltc)...")
    start = time.time()

    vsm = VectorSpaceModel()
    vsm.build_index(clean_docs)

    print(f"    Vocabulary size: {len(vsm.inverted_index)}")
    print(f"    Time: {time.time() - start:.2f}s")

    # [4] Retrieval
    print("\n[4] Retrieving documents...")
    start = time.time()

    K = 10
    all_retrieved = {}

    for q_id, tokens in clean_queries.items():
        # Retrieve all documents for accurate MAP calculation
        all_results = vsm.get_scores(tokens)
        all_retrieved[q_id] = [doc_id for doc_id, _ in all_results]

    print(f"    Time: {time.time() - start:.2f}s")

    # [5] Evaluation
    print("\n[5] Evaluating...")
    start = time.time()

    map_score = compute_map(all_retrieved, qrels)
    ndcg_score = compute_ndcg(all_retrieved, qrels, k=K)

    print(f"    Time: {time.time() - start:.4f}s")

    # Final Results
    print("\n" + "=" * 50)
    print("              FINAL RESULTS              ")
    print("=" * 50)
    print(f"MAP      : {map_score:.4f}")
    print(f"NDCG@{K} : {ndcg_score:.4f}")
    print("=" * 50)

    # --- Specific Query Test (Requested by User) ---
    target_q_id = 7
    if target_q_id in raw_queries:
        print(f"\n>>> Results for Query ID: {target_q_id}")
        print(f"Query: {raw_queries[target_q_id]}")
        
        # Ground Truth
        gt_docs = [q['doc_id'] for q in qrels if q['query_id'] == target_q_id and q['relevance'] > 0]
        print(f"\nGround Truth (Relevance > 0):")
        print(gt_docs)
        
        # Top-K Results
        top_k_results = all_retrieved[target_q_id][:K]
        print(f"\nTop-{K} Results from Model:")
        print(top_k_results)
        
        # Matches
        matches = set(gt_docs).intersection(set(top_k_results))
        print(f"\nMatches in Top-{K}: {list(matches)}")
        print(f"Precision@{K}: {len(matches)/K:.2f}")
    # -----------------------------------------------


if __name__ == "__main__":
    main()
