"""Text processing utilities."""

import re
from typing import List


def normalize_text(text: str) -> str:
    """
    Normalize text by removing extra whitespace and special characters.

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        # Move start position for next chunk (with overlap)
        start = end - overlap

        # Break if we've covered the entire text
        if start >= len(text):
            break

    return chunks
