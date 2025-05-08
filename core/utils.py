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
import torch
# Update LangChain import to use the new import structure
from langchain_community.embeddings import HuggingFaceEmbeddings
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
from tqdm import tqdm
from core.models import Document
from core.settings import LOCAL_DEFAULT_EMBEDDING, REMOTE_DEFAULT_EMBEDDING, REMOTE_DEFAULT_MODEL, LOCAL_DEFAULT_MODEL, DEFAULT_MODEL_PATH
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

_local_embedding_model = None
_local_llm_model = None
_local_llm_tokenizer = None

def get_device():
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            print("Using MPS (Apple Silicon) device for model inference")
            return torch.device("mps")
    except Exception as e:
        print(f"Error checking MPS availability: {e}")
    try:
        if torch.cuda.is_available():
            print("Using CUDA device for model inference")
            return torch.device("cuda")
    except Exception as e:
        print(f"Error checking CUDA availability: {e}")
    # Fallback to CPU
    print("No GPU acceleration available, using CPU for model inference")
    return torch.device("cpu")
_device = get_device()
def get_model_path():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, DEFAULT_MODEL_PATH.lstrip('/'))
    os.makedirs(model_path, exist_ok=True)
    return model_path

def get_local_embedding_model():
    global _local_embedding_model
    if _local_embedding_model is None:
        model_path = get_model_path()
        print(f"Loading embedding model {LOCAL_DEFAULT_EMBEDDING} with cache in {model_path}")
        _local_embedding_model = HuggingFaceEmbeddings(
            model_name=LOCAL_DEFAULT_EMBEDDING,
            model_kwargs={
                "device": _device,
                "cache_dir": model_path
            }
        )
    
    return _local_embedding_model
def generate_embeddings_local(text: str) -> List[float]:
    model = get_local_embedding_model()
    return model.embed_query(text)

def get_local_llm_model():
    global _local_llm_model, _local_llm_tokenizer
    
    if _local_llm_model is None or _local_llm_tokenizer is None:
        model_path = get_model_path()
        
        print(f"Loading LLM model {LOCAL_DEFAULT_MODEL} with cache in {model_path}")
        _local_llm_tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_DEFAULT_MODEL, 
            cache_dir=model_path
        )
        _local_llm_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_DEFAULT_MODEL, 
            device_map=_device,
            cache_dir=model_path
        )
        _local_llm_model = _local_llm_model.to(_device)
    
    return _local_llm_model, _local_llm_tokenizer

def chat_with_llm_local(messages: List[Dict[str, str]], temperature: float = 0.7, max_new_tokens: int = 1000) -> str:
    model, tokenizer = get_local_llm_model()
    prompt = ""
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if role == "system":
            prompt += f"System: {content}\n\n"
        elif role == "user":
            prompt += f"User: {content}\n\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n\n"
    prompt += "Assistant: "
    inputs = tokenizer(prompt, return_tensors="pt").to(_device)
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
    )
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_response = full_response[len(prompt):]
    return assistant_response
#API implementation 
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
def chat_with_llm_together(messages: List[Dict[str, str]], model: str = REMOTE_DEFAULT_MODEL, temperature: float = 0.7, max_tokens: int = 1000) -> str:
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
    
    # Initialize result list
    document_ids = []
    
    session_owner = False
    if db_session is None:
        engine = create_engine(os.getenv("DATABASE_URL"))
        db_session = Session(engine)
        session_owner = True
    
    try:
        all_files = []
        for root, _, files in os.walk(directory_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                file_extension = Path(filename).suffix.lower()
                if file_extension in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.pdf', 
                                     '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.htm', '.html']:
                    all_files.append((root, filename, file_path, file_extension))
        
        for root, filename, file_path, file_extension in tqdm(all_files, desc="Processing files", unit="file"):
            text = ""
            try:
                if file_extension in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json']:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                elif file_extension == '.pdf':
                    with pdfplumber.open(file_path) as pdf:
                        # Add progress bar for PDF pages
                        pages = list(pdf.pages)
                        text = "\n".join([page.extract_text() or "" for page in tqdm(
                            pages, 
                            desc=f"Extracting PDF text from {filename}", 
                            unit="page",
                            leave=False
                        )])
                elif file_extension in ['.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls']:
                    text = textract.process(file_path, encoding='utf-8').decode('utf-8')
                elif file_extension in ['.htm', '.html']:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        soup = BeautifulSoup(f.read(), 'html.parser')
                        text = soup.get_text(separator="\n")
                else:
                    continue
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                continue
            if not text.strip():
                continue
            chunks = create_chunks(text, chunk_size, chunk_overlap)
            for i, chunk in enumerate(tqdm(
                chunks, 
                desc=f"Processing chunks from {filename}", 
                unit="chunk",
                leave=False
            )):
                title = f"{filename} - Part {i+1}"
                embedding_vector = generate_embeddings_local(chunk)
                doc = Document(
                    title=title,
                    content=chunk,
                    embedding=embedding_vector
                )
                db_session.add(doc)
                db_session.commit()
                db_session.refresh(doc)
                document_ids.append(doc.id)
        
        return document_ids
    
    finally:
        if session_owner and db_session:
            db_session.close()
def create_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_size = 0    
    for sentence in tqdm(sentences, desc="Chunking text", unit="sentence", leave=False):
        sentence_size = len(sentence)
        if sentence_size > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
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
            words = sentence.split()
            current_sentence = []
            current_sentence_size = 0
            for word in words:
                if current_sentence_size + len(word) + 1 <= chunk_size:
                    current_sentence.append(word)
                    current_sentence_size += len(word) + 1
                else:
                    sentence_part = " ".join(current_sentence)
                    if current_size + len(sentence_part) <= chunk_size:
                        current_chunk.append(sentence_part)
                        current_size += len(sentence_part) + 1
                    else:
                        chunks.append(" ".join(current_chunk))
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
                    current_sentence = [word]
                    current_sentence_size = len(word) + 1
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
            if current_size + sentence_size + 1 > chunk_size:
                chunks.append(" ".join(current_chunk))
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
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks