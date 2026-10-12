
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import TextPreprocessor
from inverted_index import InvertedIndex
from search import RankedSearch


def load_queries(path):
    queries = {}

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            queries[row["_id"]] = row["text"]

    return queries


def load_qrels(path):
    relevance = defaultdict(set)

    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")

        for row in reader:
            query_id = row["query-id"]
            doc_id = row["corpus-id"]
            score = int(row["score"])

            if score > 0:
                relevance[query_id].add(doc_id)

    return relevance


def precision_at_k(results, relevant_docs, k):
    """Relevant retrieved documents divided by k."""
    if k <= 0:
        raise ValueError("k must be positive")

    hits = sum(
        1 for doc_id, _ in results[:k]
        if doc_id in relevant_docs
    )

    return hits / k


def recall_at_k(results, relevant_docs, k):
    """Relevant retrieved documents divided by all known relevant docs."""
    if not relevant_docs:
        return 0.0

    hits = sum(
        1 for doc_id, _ in results[:k]
        if doc_id in relevant_docs
    )

    return hits / len(relevant_docs)


def average_precision_at_k(results, relevant_docs, k):
    """Average precision truncated at rank k (AP@k)."""
    if not relevant_docs or k <= 0:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for rank, (doc_id, _) in enumerate(results[:k], start=1):
        if doc_id in relevant_docs:
            hits += 1
            precision_sum += hits / rank

    # Cutoff-aware denominator for AP@k
    denominator = min(len(relevant_docs), k)

    return precision_sum / denominator


def main():
    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "data" / "scifact"

    corpus_path = data_dir / "corpus.jsonl"
    queries_path = data_dir / "queries.jsonl"
    qrels_path = data_dir / "qrels" / "test.tsv"

    processor = TextPreprocessor(
        remove_stopwords=True,
        normalization="stem"
    )

    print("Building inverted index...")
    index = InvertedIndex(processor)
    index.build(corpus_path)

    engine = RankedSearch(index)

    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)

    # Evaluate queries that have test relevance judgments.
    evaluated_ids = sorted(set(queries) & set(qrels))

    if not evaluated_ids:
        print("No queries with relevance judgments were found.")
        return

    precision_values = []
    recall_values = []
    ap_values = []

    print("Evaluating", len(evaluated_ids), "queries...")

    for query_id in evaluated_ids:
        query_text = queries[query_id]
        relevant_docs = qrels[query_id]

        # Retrieve 100 results for both top-10 and top-100 metrics.
        results = engine.search(query_text, top_k=100)

        precision_values.append(
            precision_at_k(results, relevant_docs, 10)
        )
        recall_values.append(
            recall_at_k(results, relevant_docs, 10)
        )
        ap_values.append(
            average_precision_at_k(results, relevant_docs, 100)
        )

    count = len(evaluated_ids)

    print("\nEvaluation complete")
    print("Queries evaluated:", count)
    print(f"Precision@10: {sum(precision_values) / count:.4f}")
    print(f"Recall@10:    {sum(recall_values) / count:.4f}")
    print(f"MAP@100:      {sum(ap_values) / count:.4f}")


if __name__ == "__main__":
    main()
