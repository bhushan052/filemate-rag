import re

from app.services.embeddings import (
    create_embedding
)
from app.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_NAME,
    qdrant
)


def split_sentences(text: str) -> list[str]:
    """
    Split document text into sentences.
    """

    # Remove extra spaces and new lines
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    # Split after ., ! or ?
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    # Remove empty sentences
    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    return sentences


def create_chunks(
    text: str,
    sentences_per_chunk: int = 3,
    overlap: int = 1
) -> list[str]:
    """
    Create sentence-based chunks with overlap.

    Example:

    Sentence 1
    Sentence 2
    Sentence 3
    Sentence 4
    Sentence 5

    Result:

    Chunk 1 = S1 + S2 + S3
    Chunk 2 = S3 + S4 + S5
    """

    if not text:
        return []

    if sentences_per_chunk <= 0:
        raise ValueError(
            "sentences_per_chunk must be greater than 0"
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative"
        )

    if overlap >= sentences_per_chunk:
        raise ValueError(
            "overlap must be smaller than "
            "sentences_per_chunk"
        )

    sentences = split_sentences(text)

    chunks = []

    step = sentences_per_chunk - overlap

    for start in range(
        0,
        len(sentences),
        step
    ):

        end = start + sentences_per_chunk

        chunk_sentences = sentences[
            start:end
        ]

        if not chunk_sentences:
            break

        chunk = " ".join(
            chunk_sentences
        )

        chunks.append(chunk)

        # Stop when we reach the last sentence
        if end >= len(sentences):
            break

    return chunks

def retrieve_chunks(query: str, top_k: int = 3):

    query_vector = create_embedding(query)

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    )

    chunks = []

    for point in results.points:
        payload = point.payload or {}

        chunks.append({
            "text": payload.get("text", ""),
            "document_name": payload.get(
                "document_name",
                payload.get("filename", "")
            ),
            "score": point.score
        })

    return chunks