from __future__ import annotations

import string
from pathlib import Path
from typing import Dict, List

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def ensure_nltk_resources() -> None:
    """
    Ensure required NLTK resources are available.
    """
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
    ]

    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


ensure_nltk_resources()

STOP_WORDS = set(stopwords.words("english"))
PUNCT = set(string.punctuation)
STEMMER = PorterStemmer()


def preprocess_text(text: str) -> List[str]:
    """
    Apply standard IR preprocessing:
    1. Lowercasing
    2. Tokenization
    3. Stopword removal
    4. Punctuation/noise removal
    5. Stemming

    Returns:
        A list of cleaned, stemmed tokens.
    """
    text = text.lower()
    tokens = word_tokenize(text)

    cleaned_tokens: List[str] = []

    for token in tokens:
        # Skip pure punctuation tokens
        if token in PUNCT:
            continue

        # Skip tokens made entirely of punctuation
        if all(char in PUNCT for char in token):
            continue

        # Skip stopwords
        if token in STOP_WORDS:
            continue

        # Keep only tokens that contain at least one alphanumeric character
        if not any(char.isalnum() for char in token):
            continue

        # STEMMING RE-ENABLED
        stemmed = STEMMER.stem(token)
        cleaned_tokens.append(stemmed)

    return cleaned_tokens


def preprocess_collection(collection: Dict[int, str]) -> Dict[int, List[str]]:
    """
    Apply preprocessing to a collection of documents or queries.

    Args:
        collection: {id: raw_text}

    Returns:
        {id: [cleaned_tokens]}
    """
    return {item_id: preprocess_text(text) for item_id, text in collection.items()}


if __name__ == "__main__":
    sample_text = "The quick brown foxes are jumping over the lazy dogs."
    print("Original:", sample_text)
    print("Preprocessed:", preprocess_text(sample_text))
