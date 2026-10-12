# Information Retrieval System

A Python-based Information Retrieval (IR) system built using the **SciFact dataset** from the BEIR benchmark. The project implements text preprocessing, inverted indexing, positional indexing, ranked retrieval using TF-IDF, and standard retrieval evaluation metrics.

## Features

- **Text preprocessing:** Lowercasing, tokenization, stopword removal, stemming, and lemmatization.
- **Inverted index:** Maps terms to documents for efficient Boolean AND retrieval.
- **Positional index:** Stores token positions and supports phrase queries.
- **Ranked retrieval:** Uses a TF-IDF-based scoring approach to rank documents by relevance to a query.
- **Evaluation:** Computes Precision@K, Recall@K, and Mean Average Precision (MAP).
- **Automated tests:** Unit tests cover preprocessing, indexing, phrase search, and evaluation metrics.

## Dataset

This project uses the English SciFact dataset distributed through the BEIR benchmark.

The dataset includes scientific-document records, queries, and relevance judgments. Dataset files are downloaded locally and excluded from Git using `.gitignore`.

Dataset source: [BEIR Benchmark](https://github.com/beir-cellar/beir)

## Project Structure

```text
information-retrieval-system/
├── data/
│   └── scifact/              # Downloaded dataset (not tracked by Git)
├── src/
│   ├── download_dataset.py   # Dataset download
│   ├── inspect_dataset.py    # Dataset inspection
│   ├── preprocessing.py      # Text preprocessing
│   ├── inverted_index.py    # Inverted index
│   ├── positional_index.py  # Positional index and phrase search
│   ├── search.py             # TF-IDF ranked retrieval
│   └── evaluation.py         # Retrieval evaluation
├── tests/
│   └── test_ir.py            # Unit tests
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3
- pip

## Setup

Clone the repository and move into the project directory:

```bash
git clone <your-repository-url>
cd information-retrieval-system
```

Create and activate a virtual environment.

**Windows PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Download the dataset:

```powershell
python src/download_dataset.py
```

If required by the preprocessing configuration, download the NLTK WordNet data:

```powershell
python -m nltk.downloader wordnet
```

## Running the System

Build and inspect the inverted index:

```powershell
python src/inverted_index.py
```

Run positional-index phrase-search examples:

```powershell
python src/positional_index.py
```

Run ranked retrieval examples:

```powershell
python src/search.py
```

Evaluate retrieval on the available test queries and relevance judgments:

```powershell
python src/evaluation.py
```

## Testing

Run the unit tests from the project root:

```powershell
python -m unittest discover -s tests -v
```

The current test suite contains six tests covering preprocessing, inverted indexing, positional phrase search, and evaluation metrics.

## Preliminary Evaluation

An initial evaluation run over 300 test queries produced the following results:

| Metric | Score |
|---|---:|
| Precision@10 | 0.0853 |
| Recall@10 | 0.7819 |
| MAP@100 | 0.5934 |

These are preliminary results from the current implementation. Results may vary if the implementation, preprocessing configuration, dataset, or evaluation conventions change.

## Limitations

- Ranked retrieval uses a simple TF-IDF-based scoring baseline.
- The inverted index supports AND-based Boolean retrieval rather than a full Boolean query language.
- Phrase matching uses the token sequence formed from each document's title and body.
- Retrieval quality depends on preprocessing choices and the relevance judgments available in the dataset.

## Future Improvements

- Add configurable Boolean operators.
- Improve phrase-search handling across title and body boundaries.
- Compare TF-IDF with BM25.
- Add more evaluation metrics and retrieval experiments.
- Provide an interactive search interface.

## License and Attribution

This project uses the SciFact dataset distributed through BEIR. Refer to the dataset's source repository and accompanying documentation for its terms of use and attribution requirements. The project's own code license should be specified separately if the code is distributed under an open-source license.
