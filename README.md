# QA Knowledge Assistant - Local RAG System

## Overview

A local Retrieval-Augmented Generation (RAG) application built using:

* Python
* Sentence Transformers
* ChromaDB
* Ollama
* Llama 3.2

The application:

1. Reads documents
2. Splits them into chunks
3. Creates embeddings
4. Stores embeddings in ChromaDB
5. Retrieves relevant chunks based on user questions
6. Uses Ollama to generate answers grounded in retrieved content

---

## Architecture

Documents
→ Chunking
→ Embeddings
→ ChromaDB
→ Retrieval
→ Prompt Augmentation
→ Ollama (Llama 3.2)
→ Answer

---

## Project Structure

```text
rag-project/
│
├── docs/
│   └── playwright_guide.txt
│
├── chroma_db/
│
├── ingest.py
├── query.py
├── rag_chat.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Prerequisites

### Python

Verify installation:

```bash
python3 --version
```

### Ollama

Verify installation:

```bash
ollama --version
```

Download model:

```bash
ollama pull llama3.2
```

Test model:

```bash
ollama run llama3.2
```

---

## Setup

### Create Virtual Environment

```bash
python3 -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install chromadb sentence-transformers pypdf ollama
```

Generate requirements:

```bash
pip freeze > requirements.txt
```

---

## Ingest Documents

Run:

```bash
python3 ingest.py
```

This:

* Reads document content
* Creates embeddings
* Stores chunks in ChromaDB

Expected output:

```text
Stored chunk: Authentication
Stored chunk: API Testing
...
Total chunks stored: 6
```

---

## Semantic Retrieval

Run:

```bash
python3 query.py
```

Example:

```text
Question:
How is authentication handled?
```

Returns the most relevant chunks from ChromaDB.

---

## Run Full RAG Pipeline

Run:

```bash
python3 rag_chat.py
```

Example:

```text
How is authentication handled?
```

Flow:

Question
→ Embedding
→ Retrieval
→ Context Construction
→ Ollama
→ Answer

---

## Current Features

* Local embeddings
* ChromaDB vector storage
* Semantic search
* Metadata support
* Ollama integration
* Local RAG pipeline

---

## Future Improvements

* Multiple document ingestion
* PDF support
* Citations
* Similarity scores
* Chunk overlap
* Metadata filtering
* Hybrid search
* Evaluation framework
* Agent workflows
* MCP integration

---

## Learning Concepts Covered

* Embeddings
* Vector Databases
* Semantic Search
* Chunking
* Metadata
* Retrieval
* Prompt Augmentation
* Retrieval-Augmented Generation (RAG)

---

Created as a hands-on AI Engineering learning project.
