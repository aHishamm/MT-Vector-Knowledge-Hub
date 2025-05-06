from fastapi import FastAPI, Depends, HTTPException, status, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlmodel import Session, select
from typing import List, Optional, Dict
import os
from pathlib import Path

from core.models import Document, User
from core.utils import process_directory_to_vectors, generate_embeddings_together, chat_with_llm

# Create FastAPI app
app = FastAPI(
    title="MT Vector Knowledge Base",
    description="A knowledge base for your vector database using FastAPI and PostgreSQL with pgvector.",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database dependency
def get_db():
    from sqlmodel import create_engine
    engine = create_engine(os.getenv("DATABASE_URL"))
    with Session(engine) as session:
        yield session

# Background task to process directory to avoid blocking
def process_directory_background(
    directory_path: str,
    chunk_size: int,
    chunk_overlap: int,
    db_url: str
):
    from sqlmodel import create_engine, Session
    engine = create_engine(db_url)
    with Session(engine) as session:
        try:
            process_directory_to_vectors(
                directory_path=directory_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                db_session=session
            )
        except Exception as e:
            print(f"Background processing error: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Welcome to MT Vector Knowledge Base"}

@app.post("/process-directory", status_code=status.HTTP_202_ACCEPTED)
def process_directory_endpoint(
    directory_path: str,
    chunk_size: int = Query(1000, gt=0),
    chunk_overlap: int = Query(200, ge=0),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Process all files in a directory, extract text, vectorize, chunk, 
    and store in database.
    """
    # Validate directory path
    if not os.path.isdir(directory_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Directory not found: {directory_path}"
        )
    
    try:
        if background_tasks:
            # Process in background
            background_tasks.add_task(
                process_directory_background,
                directory_path=directory_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                db_url=os.getenv("DATABASE_URL")
            )
            return {
                "message": f"Processing directory in background: {directory_path}",
                "status": "processing"
            }
        else:
            # Process immediately
            document_ids = process_directory_to_vectors(
                directory_path=directory_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                db_session=db
            )
            
            return {
                "message": f"Successfully processed directory: {directory_path}",
                "document_count": len(document_ids),
                "document_ids": document_ids
            }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing directory: {str(e)}"
        )

@app.post("/generate-embeddings")
def generate_embeddings(
    text: str,
    model: Optional[str] = None
):
    """
    Generate embeddings for the provided text using Together AI API.
    """
    try:
        embedding = generate_embeddings_together(text, model) if model else generate_embeddings_together(text)
        return {
            "embedding": embedding,
            "dimensions": len(embedding),
            "model": model
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating embeddings: {str(e)}"
        )

@app.post("/chat")
def chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = Query(0.7, ge=0, le=1),
    max_tokens: int = Query(1000, gt=0)
):
    """
    Chat with an LLM using Together AI API.
    """
    try:
        if model:
            response = chat_with_llm(messages, model, temperature, max_tokens)
        else:
            response = chat_with_llm(messages, temperature=temperature, max_tokens=max_tokens)
        
        return {
            "response": response
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error chatting with LLM: {str(e)}"
        )

@app.get("/documents", response_model=List[Document])
def get_documents(
    limit: int = Query(10, gt=0, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    Retrieve documents from the database.
    """
    documents = db.exec(select(Document).offset(offset).limit(limit)).all()
    return documents

@app.get("/documents/{document_id}", response_model=Document)
def get_document(document_id: int, db: Session = Depends(get_db)):
    """
    Retrieve a specific document by ID.
    """
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found"
        )
    return document