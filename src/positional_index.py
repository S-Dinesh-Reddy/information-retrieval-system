
import json
import sys
from pathlib import Path
from collections import defaultdict

# Allow imports from the src directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import TextPreprocessor


class PositionalIndex:
    def __init__(self, preprocessor=None):
        # term -> document ID -> list of token positions
        self.index = defaultdict(lambda: defaultdict(list))
        self.documents = {}

        self.preprocessor = preprocessor or TextPreprocessor(
            remove_stopwords=False,
            normalization="none"
        )

    def build(self, corpus_path):
        """Build a positional index from a JSONL document collection."""
        with open(corpus_path, "r", encoding="utf-8") as file:
            for line in file:
                document = json.loads(line)

                doc_id = document["_id"]
                title = document.get("title", "")
                text = document.get("text", "")

                # Keep the same text representation for indexing and display
                full_text = (title + " " + text).strip()
                tokens = self.preprocessor.preprocess(full_text)

                self.documents[doc_id] = {
                    "title": title,
                    "text": text
                }

                for position, token in enumerate(tokens):
                    self.index[token][doc_id].append(position)

        return self

    def search_phrase(self, phrase):
        """Return documents containing the exact normalized token sequence."""
        query_tokens = self.preprocessor.preprocess(phrase)

        if not query_tokens:
            return []

        # Candidate documents must contain the first query term
        candidates = set(self.index.get(query_tokens[0], {}).keys())

        for token in query_tokens[1:]:
            candidates &= set(self.index.get(token, {}).keys())

        results = []

        for doc_id in candidates:
            first_positions = self.index[query_tokens[0]][doc_id]

            for start in first_positions:
                # Check whether every subsequent term is at the next position
                if all(
                    start + offset in self.index[token][doc_id]
                    for offset, token in enumerate(query_tokens[1:], start=1)
                ):
                    results.append(doc_id)
                    break

        return sorted(results)

    def search_term(self, term):
        """Return documents containing a single normalized term."""
        tokens = self.preprocessor.preprocess(term)

        if len(tokens) != 1:
            raise ValueError("Please provide a single term.")

        return sorted(self.index.get(tokens[0], {}).keys())


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    corpus_path = project_dir / "data" / "scifact" / "corpus.jsonl"

    processor = TextPreprocessor(
        remove_stopwords=False,
        normalization="none"
    )

    positional_index = PositionalIndex(processor)
    positional_index.build(corpus_path)

    print("Documents indexed:", len(positional_index.documents))
    print("Unique terms indexed:", len(positional_index.index))

    phrase = input("\nEnter a phrase to search for: ")
    results = positional_index.search_phrase(phrase)

    print("\nMatching documents:", len(results))

    for doc_id in results[:10]:
        document = positional_index.documents[doc_id]
        print(f"\nDocument ID: {doc_id}")
        print("Title:", document["title"])
        print("Text:", document["text"][:300], "...")
