from __future__ import annotations

import string
import re
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
        ("tokenizers/punkt_tab", "punkt_tab"),
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
    2. Punctuation removal (replaced with space)
    3. Tokenization
    4. Stopword removal
    5. Stemming

    Returns:
        A list of cleaned, stemmed tokens.
    """
    text = text.lower()
    
    # Replace punctuation with spaces to separate hyphenated words correctly
    text = re.sub(r'[^a-z0-9]', ' ', text)
    
    tokens = word_tokenize(text)

    cleaned_tokens: List[str] = []

    for token in tokens:
        # Skip stopwords
        if token in STOP_WORDS:
            continue

        # Skip pure digits (optional but standard for Cranfield)
        if token.isdigit():
            continue

        # Stemming
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
