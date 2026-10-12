
import math
import sys
from pathlib import Path
from collections import Counter

# Allow imports from the src directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inverted_index import InvertedIndex
from preprocessing import TextPreprocessor


class RankedSearch:
    def __init__(self, inverted_index):
        self.ir_index = inverted_index
        self.documents = inverted_index.documents
        self.document_count = len(self.documents)

        # Store document lengths and term frequencies
        self.term_frequencies = {}
        self.document_lengths = {}

        # Build statistics from the same preprocessing configuration
        for doc_id, document in self.documents.items():
            text = document["title"] + " " + document["text"]
            tokens = self.ir_index.preprocessor.preprocess(text)

            self.term_frequencies[doc_id] = Counter(tokens)
            self.document_lengths[doc_id] = len(tokens)

        # Average document length for BM25 is not needed for TF-IDF,
        # but average length can be useful for later extensions.
        self.document_frequency = Counter()

        for frequencies in self.term_frequencies.values():
            for term in frequencies:
                self.document_frequency[term] += 1

    def search(self, query, top_k=10):
        """Return matching documents ranked by TF-IDF score."""
        if top_k <= 0:
            return []

        query_tokens = self.ir_index.preprocessor.preprocess(query)

        if not query_tokens:
            return []

        query_terms = set(query_tokens)
        scores = Counter()

        for term in query_terms:
            df = self.document_frequency.get(term, 0)

            if df == 0:
                continue

            # Smoothed inverse document frequency
            idf = math.log(
                (self.document_count + 1) / (df + 1)
            ) + 1

            # Use the inverted index to find candidate documents
            candidate_docs = self.ir_index.index.get(term, set())

            for doc_id in candidate_docs:
                tf = self.term_frequencies[doc_id][term]
                tf_weight = 1 + math.log(tf)
                scores[doc_id] += tf_weight * idf

        # Higher scores first; document ID breaks ties deterministically
        ranked_results = sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0])
        )

        return ranked_results[:top_k]


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    corpus_path = project_dir / "data" / "scifact" / "corpus.jsonl"

    processor = TextPreprocessor(
        remove_stopwords=True,
        normalization="stem"
    )

    index = InvertedIndex(processor)
    index.build(corpus_path)

    search_engine = RankedSearch(index)

    print("Documents indexed:", len(index.documents))
    print("Unique terms indexed:", len(index.index))

    query = input("\nEnter your search query: ")
    results = search_engine.search(query, top_k=10)

    print(f"\nTop {len(results)} ranked results:")

    for rank, (doc_id, score) in enumerate(results, start=1):
        document = search_engine.documents[doc_id]

        print(f"\n{rank}. Document ID: {doc_id}")
        print(f"   TF-IDF score: {score:.4f}")
        print(f"   Title: {document['title']}")
        print(f"   Text: {document['text'][:250]}...")
