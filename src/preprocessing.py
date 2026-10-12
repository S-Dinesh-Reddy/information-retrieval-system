
import re
import nltk

from nltk.stem import PorterStemmer, WordNetLemmatizer


# Common English stop words.
# This list can be expanded as needed.
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "am", "be", "been", "being", "and", "or", "but",
    "of", "in", "on", "at", "to", "for", "from",
    "by", "with", "as", "it", "this", "that",
    "these", "those", "has", "have", "had", "do",
    "does", "did", "not"
}


# Normalize selected abbreviations before tokenization.
ABBREVIATIONS = {
    "usa": "usa",
    "u.s.a.": "usa",
    "u.s.a": "usa",
    "uk": "uk",
    "u.k.": "uk",
    "u.k": "uk",
    "u.s.": "us",
    "u.s": "us",
}


class TextPreprocessor:
    def __init__(
        self,
        remove_stopwords=True,
        normalization="stem",
        lowercase=True
    ):
        """
        normalization options:
            "none"   - no stemming or lemmatization
            "stem"   - use Porter stemming
            "lemma"  - use WordNet lemmatization
        """
        if normalization not in {"none", "stem", "lemma"}:
            raise ValueError(
                "normalization must be 'none', 'stem', or 'lemma'"
            )

        self.remove_stopwords = remove_stopwords
        self.normalization = normalization
        self.lowercase = lowercase

        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def normalize_abbreviations(self, text):
        """Convert selected dotted abbreviations to standard forms."""
        for abbreviation, replacement in ABBREVIATIONS.items():
            pattern = (
                r"(?<![A-Za-z0-9])"
                + re.escape(abbreviation)
                + r"(?![A-Za-z0-9])"
            )
            text = re.sub(
                pattern,
                replacement,
                text,
                flags=re.IGNORECASE
            )

        return text

    def tokenize(self, text):
        """Normalize case and split text into word tokens."""
        if self.lowercase:
            text = text.lower()

        text = self.normalize_abbreviations(text)

        # Keep letters and numbers; punctuation separates tokens.
        return re.findall(r"[a-z0-9]+", text)

    def preprocess(self, text):
        """Apply tokenization, stop-word removal and normalization."""
        tokens = self.tokenize(text)

        if self.remove_stopwords:
            tokens = [
                token for token in tokens
                if token not in STOP_WORDS
            ]

        if self.normalization == "stem":
            tokens = [
                self.stemmer.stem(token)
                for token in tokens
            ]

        elif self.normalization == "lemma":
            tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
            ]

        return tokens


if __name__ == "__main__":
    sample = (
        "The SCIENCE of the USA and U.S.A. "
        "studies biological systems, studying "
        "cells and their functions."
    )

    print("Original text:")
    print(sample)

    basic = TextPreprocessor(
        remove_stopwords=False,
        normalization="none"
    )

    stemming = TextPreprocessor(
        remove_stopwords=True,
        normalization="stem"
    )

    lemmatizing = TextPreprocessor(
        remove_stopwords=True,
        normalization="lemma"
    )

    print("\nBasic tokens:")
    print(basic.preprocess(sample))

    print("\nWith stemming:")
    print(stemming.preprocess(sample))

    print("\nWith lemmatization:")
    try:
        print(lemmatizing.preprocess(sample))
    except LookupError:
        print("WordNet resource missing.")
        print("Run: python -m nltk.downloader wordnet")
        raise
