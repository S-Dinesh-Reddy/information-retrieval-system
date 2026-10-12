
import json
import sys
from pathlib import Path
from collections import defaultdict

# Allow imports from the src directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import TextPreprocessor


class InvertedIndex:
    def __init__(self, preprocessor=None):
        self.index = defaultdict(set)
        self.documents = {}
        self.preprocessor = preprocessor or TextPreprocessor(
            remove_stopwords=True,
            normalization="stem"
        )

    def build(self, corpus_path):
        """Build an inverted index from a JSONL document collection."""
        with open(corpus_path, "r", encoding="utf-8") as file:
            for line in file:
                document = json.loads(line)

                doc_id = document["_id"]
                title = document.get("title", "")
                text = document.get("text", "")

                # Include title and body when indexing
                full_text = title + " " + text
                tokens = self.preprocessor.preprocess(full_text)

                self.documents[doc_id] = {
                    "title": title,
                    "text": text
                }

                for token in set(tokens):
                    self.index[token].add(doc_id)

        return self

    def search(self, query):
        """Return documents containing every query term."""
        query_tokens = self.preprocessor.preprocess(query)

        if not query_tokens:
            return []

        matching_docs = None

        for token in set(query_tokens):
            doc_ids = self.index.get(token, set())

            if matching_docs is None:
                matching_docs = set(doc_ids)
            else:
                matching_docs &= doc_ids

        return sorted(matching_docs or set())


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    corpus_path = project_dir / "data" / "scifact" / "corpus.jsonl"

    processor = TextPreprocessor(
        remove_stopwords=True,
        normalization="stem"
    )

    search_index = InvertedIndex(processor)
    search_index.build(corpus_path)

    print("Documents indexed:", len(search_index.documents))
    print("Unique terms indexed:", len(search_index.index))

    query = input("\nEnter a search query: ")
    results = search_index.search(query)

    print("\nMatching documents:", len(results))

    for doc_id in results[:10]:
        document = search_index.documents[doc_id]
        print(f"\nDocument ID: {doc_id}")
        print("Title:", document["title"])
        print("Text:", document["text"][:300], "...")
