import os
import json
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    pipeline
)
from langchain.embeddings import HuggingFaceEmbeddings
from sqlmodel import SQLModel, Field, Session, select, create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import re
import nltk
from bs4 import BeautifulSoup
import textract
import pdfplumber
from fastapi import HTTPException, status
from dotenv import load_dotenv
from pathlib import Path
from core.models import Document
from core.settings import LOCAL_DEFAULT_EMBEDDING, REMOTE_DEFAULT_EMBEDDING, REMOTE_DEFAULT_MODEL
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

_together_client = None

def get_together_client():
    global _together_client
    if _together_client is None:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY environment variable is not set")
        from together import Together
        _together_client = Together(api_key=api_key)
    
    return _together_client

def generate_embeddings_together(text: str, model: str = REMOTE_DEFAULT_EMBEDDING) -> List[float]:
    client = get_together_client()
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def chat_with_llm(messages: List[Dict[str, str]], model: str = REMOTE_DEFAULT_MODEL, temperature: float = 0.7, max_tokens: int = 1000) -> str:
    client = get_together_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def process_directory_to_vectors(directory_path: str, chunk_size: int = 1000, chunk_overlap: int = 200, db_session: Session = None) -> List[int]:
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory {directory_path} does not exist")
    
    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(model_name=LOCAL_DEFAULT_EMBEDDING)
    
    # Initialize result list
    document_ids = []
    
    # Create database session if not provided
    session_owner = False
    if db_session is None:
        engine = create_engine(os.getenv("DATABASE_URL"))
        db_session = Session(engine)
        session_owner = True
    
    try:
        # Process all files in directory
        for root, _, files in os.walk(directory_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                file_extension = Path(filename).suffix.lower()
                
                # Extract text based on file type
                text = ""
                try:
                    if file_extension in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json']:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read()
                    elif file_extension == '.pdf':
                        with pdfplumber.open(file_path) as pdf:
                            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
                    elif file_extension in ['.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls']:
                        text = textract.process(file_path, encoding='utf-8').decode('utf-8')
                    elif file_extension in ['.htm', '.html']:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            soup = BeautifulSoup(f.read(), 'html.parser')
                            text = soup.get_text(separator="\n")
                    else:
                        # Skip unsupported file types
                        continue
                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")
                    continue
                
                # Skip if no text was extracted
                if not text.strip():
                    continue
                
                # Split text into chunks
                chunks = create_chunks(text, chunk_size, chunk_overlap)
                
                # Process each chunk
                for i, chunk in enumerate(chunks):
                    title = f"{filename} - Part {i+1}"
                    # Create embedding vector
                    embedding_vector = embeddings.embed_query(chunk)
                    
                    # Create document in database
                    doc = Document(
                        title=title,
                        content=chunk,
                        embedding=embedding_vector
                    )
                    
                    # Add and commit to database
                    db_session.add(doc)
                    db_session.commit()
                    db_session.refresh(doc)
                    document_ids.append(doc.id)
        
        return document_ids
    
    finally:
        # Close session if we created it
        if session_owner and db_session:
            db_session.close()

def create_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Split text into sentences
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        # If the sentence is too long, split it into smaller parts
        if sentence_size > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep the overlapping sentences
                overlap_size = 0
                overlap_chunk = []
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= chunk_overlap:
                        overlap_chunk.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                current_chunk = overlap_chunk
                current_size = overlap_size
            
            # Split the long sentence
            words = sentence.split()
            current_sentence = []
            current_sentence_size = 0
            
            for word in words:
                if current_sentence_size + len(word) + 1 <= chunk_size:
                    current_sentence.append(word)
                    current_sentence_size += len(word) + 1
                else:
                    # Add the current part of the sentence to the chunks
                    sentence_part = " ".join(current_sentence)
                    if current_size + len(sentence_part) <= chunk_size:
                        current_chunk.append(sentence_part)
                        current_size += len(sentence_part) + 1
                    else:
                        chunks.append(" ".join(current_chunk))
                        # Keep the overlapping sentences
                        overlap_size = 0
                        overlap_chunk = []
                        for s in reversed(current_chunk):
                            if overlap_size + len(s) <= chunk_overlap:
                                overlap_chunk.insert(0, s)
                                overlap_size += len(s)
                            else:
                                break
                        current_chunk = overlap_chunk
                        current_chunk.append(sentence_part)
                        current_size = overlap_size + len(sentence_part) + 1
                    
                    # Reset the current sentence
                    current_sentence = [word]
                    current_sentence_size = len(word) + 1
            
            # Add any remaining part of the sentence
            if current_sentence:
                sentence_part = " ".join(current_sentence)
                if current_size + len(sentence_part) <= chunk_size:
                    current_chunk.append(sentence_part)
                    current_size += len(sentence_part) + 1
                else:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence_part]
                    current_size = len(sentence_part) + 1
        else:
            # If adding this sentence exceeds the chunk size, start a new chunk
            if current_size + sentence_size + 1 > chunk_size:
                chunks.append(" ".join(current_chunk))
                
                # Keep the overlapping sentences
                overlap_size = 0
                overlap_chunk = []
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= chunk_overlap:
                        overlap_chunk.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_chunk.append(sentence)
                current_size = overlap_size + sentence_size + 1
            else:
                current_chunk.append(sentence)
                current_size += sentence_size + 1
    
    # Add the final chunk if it has content
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

