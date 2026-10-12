
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "scifact"

corpus_file = DATA_DIR / "corpus.jsonl"
queries_file = DATA_DIR / "queries.jsonl"
qrels_file = DATA_DIR / "qrels" / "test.tsv"

# Inspect one document
with corpus_file.open("r", encoding="utf-8") as file:
    first_document = json.loads(next(file))

print("FIRST DOCUMENT")
print(first_document)
print()

# Inspect one query
with queries_file.open("r", encoding="utf-8") as file:
    first_query = json.loads(next(file))

print("FIRST QUERY")
print(first_query)
print()

# Count documents and queries
with corpus_file.open("r", encoding="utf-8") as file:
    document_count = sum(1 for _ in file)

with queries_file.open("r", encoding="utf-8") as file:
    query_count = sum(1 for _ in file)

print("TOTAL DOCUMENTS:", document_count)
print("TOTAL QUERIES:", query_count)

# Display the relevance-judgment columns
with qrels_file.open("r", encoding="utf-8") as file:
    for _ in range(4):
        line = file.readline()
        if not line:
            break
        print(line.rstrip())
