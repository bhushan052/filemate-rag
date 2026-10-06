import os
import re
from qdrant_client import QdrantClient
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


QDRANT_URL = os.getenv("QDRANT_URL")

QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY",
    ""
)

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "filemate_documents"
)

if QDRANT_API_KEY:
    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
else:
    qdrant = QdrantClient(
        url=QDRANT_URL
    )