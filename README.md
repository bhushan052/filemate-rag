# FileMate RAG

FileMate RAG is a technical/project document assistant built using FastAPI, RAG, embeddings, and Qdrant vector database.

## Features

- Upload PDF and DOCX documents
- Extract text from documents
- Sentence-based chunking with overlap
- Generate embeddings
- Store document vectors in Qdrant
- Semantic search over uploaded documents
- Ask questions using the `/ask` API
- Web-based interface for document upload and question answering

## Tech Stack

- Python
- FastAPI
- Qdrant
- RAG
- Embeddings
- PDF / DOCX processing
- HTML / JavaScript

## Project Structure

```text
filemate-rag/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── models.py
│   ├── services/
│   │   ├── chunker.py
│   │   ├── document_processor.py
│   │   ├── embeddings.py
│   │   └── qdrant.py
|   |   └── llm.py 
│   └── static/
│       └── index.html
├── logs/
├── .env
├── requirements.txt
└── README.md
