#Need major overhaul after moving to Django to update the DB with correct data 
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
    pipeline,
)
import torch
#from langchain_community.embeddings import HuggingFaceEmbeddings #deprecated 
from langchain_huggingface import HuggingFaceEmbeddings
from bs4 import BeautifulSoup
import textract
import pdfplumber
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from together import Together
from coreapi.models import Document
from coreapi.settings import (
    LOCAL_DEFAULT_EMBEDDING_MODEL, LOCAL_DEFAULT_MODEL, 
    REMOTE_DEFAULT_EMBEDDING_MODEL, REMOTE_DEFAULT_MODEL
)
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
class HuggingFaceEmbeddingInitializer:
    """
    Initializes Hugging Face tokenizer and embedding model based on settings.
    """
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        from coreapi.settings import LOCAL_DEFAULT_EMBEDDING_MODEL
        self.model_name = model_name or LOCAL_DEFAULT_EMBEDDING_MODEL
        self.tokenizer = None
        self.model = None
        self.embedding = None
        if device: 
            self.device = device
        else: 
            if torch.cuda.is_available(): 
                self.device = 'cuda' 
            elif torch.backends.mps.is_available(): 
                self.device = 'mps'
            else: 
                self.device = 'cpu'
        self.model_dir = Path(__file__).parent.parent / "HF_models" / self.model_name.replace("/", "_")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()
    def _is_valid_hf_model_dir(self, model_dir: Path) -> bool:
        config_path = model_dir / "config.json"
        if not config_path.exists():
            return False
        try:
            import json
            with open(config_path, "r") as f:
                config = json.load(f)
            return "model_type" in config
        except Exception:
            return False
    def _initialize(self):
        use_local = self._is_valid_hf_model_dir(self.model_dir)
        model_source = str(self.model_dir) if use_local else self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            cache_dir=str(self.model_dir)
        )
        self.model = AutoModel.from_pretrained(
            model_source,
            cache_dir=str(self.model_dir)
        ).to(self.device)
        self.embedding = HuggingFaceEmbeddings(model_name=self.model_name,model_kwargs={"device": self.device})
    def get_tokenizer(self):
        return self.tokenizer
    def get_model(self):
        return self.model
    def get_embedding(self):
        return self.embedding
    
class TogetherEmbeddingInitializer:
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        from coreapi.settings import REMOTE_DEFAULT_EMBEDDING_MODEL
        self.model_name = model_name or REMOTE_DEFAULT_EMBEDDING_MODEL
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        self.client = None
        self._initialize()
    def _initialize(self):
        self.client = Together(api_key=self.api_key)
    def get_client(self):
        return self.client
    def get_model_name(self):
        return self.model_name    

def generate_hf_embedding(text: str, model_name: Optional[str] = None) -> list:
    """
    Generate an embedding for a string using HuggingFaceEmbeddingInitializer.
    Returns the embedding vector as a list.
    """
    initializer = HuggingFaceEmbeddingInitializer(model_name=model_name)
    embedding_model = initializer.get_embedding()
    embedding = embedding_model.embed_documents([text])[0] #Will return a vector of 1024 values
    return embedding

def generate_together_embedding(text: str, model_name: Optional[str] = None, api_key: Optional[str] = None) -> list:
    """
    Generate an embedding for a string using TogetherEmbeddingInitializer.
    Returns the embedding vector as a list.
    """
    initializer = TogetherEmbeddingInitializer(model_name=model_name, api_key=api_key)
    client = initializer.get_client()
    model = initializer.get_model_name()
    response = client.embeddings.create(input=[text], model=model)
    embedding = response['data'][0]['embedding']
    return embedding