import os

from uuid import uuid4

from dotenv import load_dotenv

from qdrant_client import QdrantClient

from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)


load_dotenv()


QDRANT_URL = os.getenv("QDRANT_URL")

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "rag_documents"
)


client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
    if QDRANT_API_KEY
    else None
)


def create_collection(vector_size):

    collections = client.get_collections().collections

    exists = any(
        collection.name == COLLECTION_NAME
        for collection in collections
    )

    if not exists:

        client.create_collection(

            collection_name=COLLECTION_NAME,

            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )


def insert_vector(
    vector,
    text,
    filename
):

    point = PointStruct(

        id=str(uuid4()),

        vector=vector,

        payload={
            "text": text,
            "filename": filename
        }
    )

    client.upsert(

        collection_name=COLLECTION_NAME,

        points=[point]
    )


def search_vector(
    query_vector,
    top_k=3
):

    results = client.query_points(

        collection_name=COLLECTION_NAME,

        query=query_vector,

        limit=top_k,

        with_payload=True
    )

    return results.points