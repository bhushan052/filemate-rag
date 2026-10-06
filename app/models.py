from typing import Optional

from pydantic import BaseModel, Field


# ==========================================
# Ask Question Request
# ==========================================

class QuestionRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask about uploaded documents"
    )


# ==========================================
# Ask Question Response
# ==========================================

class QuestionResponse(BaseModel):

    answer: str

    explanation: str

    details: str

    source: str


# ==========================================
# Upload Response
# ==========================================

class UploadResponse(BaseModel):

    message: str

    document: str

    chunks: int


# ==========================================
# Source Information
# ==========================================

class SourceInfo(BaseModel):

    document_name: str

    chunk_id: Optional[int] = None

    score: Optional[float] = None

    chunk_text: str