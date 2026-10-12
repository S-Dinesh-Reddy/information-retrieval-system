
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from preprocessing import TextPreprocessor
from inverted_index import InvertedIndex
from positional_index import PositionalIndex
from evaluation import (
    precision_at_k,
    recall_at_k,
    average_precision_at_k,
)


class TestInformationRetrieval(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.corpus_path = Path(self.temp_dir.name) / "corpus.jsonl"

        documents = [
            {
                "_id": "D1",
                "title": "Apple research",
                "text": "apple mango apple",
                "metadata": {},
            },
            {
                "_id": "D2",
                "title": "Mango research",
                "text": "mango banana",
                "metadata": {},
            },
            {
                "_id": "D3",
                "title": "Fruit research",
                "text": "apple banana",
                "metadata": {},
            },
        ]

        with self.corpus_path.open("w", encoding="utf-8") as file:
            for document in documents:
                file.write(json.dumps(document) + "\n")

        self.processor = TextPreprocessor(
            remove_stopwords=False,
            normalization="none",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_preprocessing(self):
        tokens = self.processor.preprocess("USA U.S.A. SCIENCE")
        self.assertEqual(tokens, ["usa", "usa", "science"])

    def test_inverted_index(self):
        index = InvertedIndex(self.processor)
        index.build(self.corpus_path)

        self.assertEqual(len(index.documents), 3)
        self.assertEqual(index.search("apple mango"), ["D1"])
        self.assertEqual(index.search("apple banana"), ["D3"])
        self.assertEqual(index.search("unknownterm"), [])

    def test_positional_index(self):
        index = PositionalIndex(self.processor)
        index.build(self.corpus_path)

        self.assertEqual(index.search_phrase("apple mango"), ["D1"])
        self.assertEqual(index.search_phrase("mango banana"), ["D2"])        
        self.assertEqual(index.search_phrase("apple"), ["D1", "D3"])

    def test_precision_at_k(self):
        results = [("D1", 3.0), ("D2", 2.0), ("D3", 1.0)]
        relevant = {"D1", "D3"}

        self.assertAlmostEqual(
            precision_at_k(results, relevant, 2), 0.5
        )

    def test_recall_at_k(self):
        results = [("D1", 3.0), ("D2", 2.0), ("D3", 1.0)]
        relevant = {"D1", "D3"}

        self.assertAlmostEqual(
            recall_at_k(results, relevant, 2), 0.5
        )

    def test_average_precision_at_k(self):
        results = [("D1", 3.0), ("D2", 2.0), ("D3", 1.0)]
        relevant = {"D1", "D3"}

        # Relevant documents at ranks 1 and 3:
        # AP@3 = (1/1 + 2/3) / 2
        self.assertAlmostEqual(
            average_precision_at_k(results, relevant, 3),
            5 / 6,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
