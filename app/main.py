import os
import re
import uuid
from pathlib import Path
import logging
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader
from docx import Document

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)



# ==========================================
# Logging Configuration
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

LOG_DIR = BASE_DIR / "logs"

LOG_DIR.mkdir(
    exist_ok=True
)

LOG_FILE = LOG_DIR / "filemate.log"


logging.basicConfig(
    level=logging.INFO,

    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),

    handlers=[
        logging.FileHandler(
            LOG_FILE,
            encoding="utf-8"
        ),

        logging.StreamHandler()
    ]
)


logger = logging.getLogger(
    "filemate"
)


logger.info(
    "========================================"
)

logger.info(
    "FileMate application started"
)

logger.info(
    "Log file: %s",
    LOG_FILE
)

logger.info(
    "========================================"
)

from app.models import (
    QuestionRequest,
    QuestionResponse,
    UploadResponse,
    SourceInfo
)

from app.services.document_processor import (
    extract_text
)

from app.services.chunker import (
    create_chunks,
    retrieve_chunks
)

from app.services.embeddings import (
    create_embedding
)


from app.services.qdrant import (
    create_collection,
    insert_vector,
    search_vector
)

from app.services.llm import (
    generate_answer
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

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

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# --------------------------------------------------
# Clients
# --------------------------------------------------

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
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


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="FileMate RAG API",
    description="Technical / Project Document Assistant",
    version="1.0.0"
)


@app.get("/")
async def home():

    return FileResponse(
        STATIC_DIR / "index.html"
    )


# --------------------------------------------------
# Upload API
# --------------------------------------------------
@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    logger.info(
        "UPLOAD STARTED | filename=%s",
        file.filename
    )

    allowed_extensions = {
        ".pdf",
        ".docx"
    }

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are supported."
        )

    contents = await file.read()

    logger.info(
        "File received: %s (%d bytes)",
        file.filename,
        len(contents)
    )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension
    ) as temp_file:

        temp_file.write(contents)
        temp_file_path = temp_file.name

    try:

        # ------------------------------------------
        # Step 1: Extract text
        # ------------------------------------------

        logger.info("Extracting text...")

        text = extract_text(
            temp_file_path
        )

        logger.info(
            "Extracted %d characters",
            len(text)
        )

        if not text.strip():

            raise HTTPException(
                status_code=400,
                detail=(
                    "No readable text was found "
                    "in the uploaded document."
                )
            )

        # ------------------------------------------
        # Step 2: Chunk document
        # ------------------------------------------

        logger.info("Creating chunks...")

        chunks = create_chunks(text)

        logger.info(
            "Created %d chunks",
            len(chunks)
        )

        if not chunks:

            raise HTTPException(
                status_code=400,
                detail="No chunks were created."
            )

        # ------------------------------------------
        # Step 3: Create first embedding
        # ------------------------------------------

        logger.info(
            "Creating first embedding..."
        )

        first_vector = create_embedding(
            chunks[0]
        )

        logger.info(
            "Embedding dimension: %d",
            len(first_vector)
        )

        # ------------------------------------------
        # Step 4: Create Qdrant collection
        # ------------------------------------------

        logger.info(
            "Creating/checking Qdrant collection..."
        )

        create_collection(
            vector_size=len(first_vector)
        )

        # ------------------------------------------
        # Step 5: Store vectors
        # ------------------------------------------

        chunks_created = 0

        for index, chunk in enumerate(chunks):

            logger.info(
                "Processing chunk %d/%d",
                index + 1,
                len(chunks)
            )

            # Reuse first vector for first chunk
            if index == 0:

                vector = first_vector

            else:

                vector = create_embedding(
                    chunk
                )

            insert_vector(
                vector=vector,
                text=chunk,
                filename=file.filename
            )

            chunks_created += 1

        logger.info(
            "Upload completed successfully: %s",
            file.filename
        )

        return {
            "message": (
                "Document uploaded and indexed successfully"
            ),
            "filename": file.filename,
            "chunks_created": chunks_created
        }

    except HTTPException:
        raise

    except Exception as e:

        logger.exception(
            "Document processing failed: %s",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        if os.path.exists(temp_file_path):

            os.remove(
                temp_file_path
            )

            logger.info(
                "Temporary file removed"
            )
# --------------------------------------------------
# Ask API
# --------------------------------------------------

@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail=
                "Question cannot be empty."
        )

    try:

        retrieved_chunks = retrieve_chunks(
            question,
            top_k=3
        )

        if not retrieved_chunks:

            return {

                "answer":
                    "The information is not "
                    "available in the uploaded "
                    "documents.",

                "explanation": "",

                "details": "",

                "source": ""
            }

        answer = generate_answer(
            question,
            retrieved_chunks
        )

        parsed = parse_answer(
            answer
        )
        

        sources = list(
            dict.fromkeys(
                item["document_name"]
                for item in retrieved_chunks
                if item["document_name"]
            )
        )

        return {

            "answer":
                parsed["answer"],

            "explanation":
                parsed["explanation"],

            "details":
                parsed["details"],

            "source":
                ", ".join(sources)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

def parse_answer(response: str) -> dict:
    return {
        "answer": response.strip(),
        "explanation": "Answer generated from the retrieved document context.",
        "details": "",
        "source": ""
    }    