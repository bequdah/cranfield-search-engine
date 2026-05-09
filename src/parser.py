from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def parse_cran_docs(file_path: str | Path) -> Dict[int, str]:
    """
    Parse Cranfield documents from cran.all.1400.

    Uses ONLY:
      - .T => title
      - .W => abstract

    Returns:
        {doc_id: "title + abstract"}
    """
    file_path = Path(file_path)
    docs: Dict[int, str] = {}

    current_id: int | None = None
    current_field: str | None = None
    title_lines: List[str] = []
    abstract_lines: List[str] = []

    def save_current_doc() -> None:
        """Save the currently parsed document into docs."""
        if current_id is None:
            return

        title = " ".join(line for line in title_lines if line).strip()
        abstract = " ".join(line for line in abstract_lines if line).strip()
        full_text = f"{title} {title} {abstract}".strip()

        docs[current_id] = full_text

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            # New document
            if line.startswith(".I"):
                save_current_doc()

                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid document ID line: {line!r}")

                current_id = int(parts[1])
                current_field = None
                title_lines = []
                abstract_lines = []
                continue

            # Field marker — in SMART format these are always alone on a line.
            # Exact match prevents content lines starting with '.' (e.g. '.5 mach')
            # from being mistakenly treated as field markers.
            if line.strip() in {".T", ".W", ".A", ".B", ".X"}:
                current_field = line.strip()
                continue

            # Content lines
            stripped = line.strip()
            if current_field == ".T":
                title_lines.append(stripped)
            elif current_field == ".W":
                abstract_lines.append(stripped)
            # Ignore all other fields (.A, .B, etc.)

    # Save last document
    save_current_doc()
    return docs


def parse_cran_queries(file_path: str | Path) -> Dict[int, str]:
    """
    Parse Cranfield queries from cran.qry.

    Uses ONLY:
      - .W => query text

    Returns:
        {query_id: query_text}
    """
    file_path = Path(file_path)
    queries: Dict[int, str] = {}

    current_id: int | None = None
    current_field: str | None = None
    query_lines: List[str] = []
    
    # Cranfield query IDs in qrels are sequential (1 to 225), NOT the values in .I tags!
    sequential_id_counter = 0

    def save_current_query() -> None:
        """Save the currently parsed query into queries."""
        if current_id is None:
            return
        queries[current_id] = " ".join(line for line in query_lines if line).strip()

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            # New query
            if line.startswith(".I"):
                save_current_query()
                sequential_id_counter += 1
                current_id = sequential_id_counter
                current_field = None
                query_lines = []
                continue

            # Field marker — in SMART format these are always alone on a line.
            # Exact match prevents content lines starting with '.' (e.g. '.5 mach')
            # from being mistakenly treated as field markers.
            if line.strip() in {".T", ".W", ".A", ".B", ".X"}:
                current_field = line.strip()
                continue

            # Content lines
            if current_field == ".W":
                query_lines.append(line.strip())

    # Save last query
    save_current_query()
    return queries


def parse_cran_qrels(file_path: str | Path) -> List[dict]:
    """
    Parse Cranfield qrels from cranqrel.

    Expected format per line:
        query_id  doc_id  relevance

    Example:
        1 184 2

    Returns:
        [
            {"query_id": 1, "doc_id": 184, "relevance": 2},
            ...
        ]
    """
    file_path = Path(file_path)
    qrels: List[dict] = []

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if not line:
                continue

            parts = line.split()
            if len(parts) < 3:
                raise ValueError(
                    f"Invalid qrel format at line {line_number}: {raw_line!r}"
                )

            query_id, doc_id, relevance = int(parts[0]), int(parts[1]), int(parts[2])

            qrels.append(
                {
                    "query_id": query_id,
                    "doc_id": doc_id,
                    "relevance": relevance,
                }
            )

    return qrels


if __name__ == "__main__":
    docs_path = Path("../dataset/cran.all.1400")
    queries_path = Path("../dataset/cran.qry")
    qrels_path = Path("../dataset/cranqrel")

    if docs_path.exists():
        docs = parse_cran_docs(docs_path)
        print(f"Parsed {len(docs)} documents.")
        first_doc_id = next(iter(docs))
        print(f"Sample doc [{first_doc_id}]: {docs[first_doc_id][:300]}...\n")

    if queries_path.exists():
        queries = parse_cran_queries(queries_path)
        print(f"Parsed {len(queries)} queries.")
        first_query_id = next(iter(queries))
        print(f"Sample query [{first_query_id}]: {queries[first_query_id]}\n")

    if qrels_path.exists():
        qrels = parse_cran_qrels(qrels_path)
        print(f"Parsed {len(qrels)} qrels.")
        print(f"Sample qrel: {qrels[0]}")
