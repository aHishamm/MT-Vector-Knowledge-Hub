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
from langchain_community.embeddings import HuggingFaceEmbeddings
import re
import nltk
from bs4 import BeautifulSoup
import textract
import pdfplumber
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from coreapi.models import Document
from django.conf import settings

def load_json(file_path: str) -> Any:
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data: Any, file_path: str) -> None:
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def read_text_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()

def write_text_file(data: str, file_path: str) -> None:
    with open(file_path, 'w') as file:
        file.write(data)

def extract_text_from_pdf(file_path: str) -> str:
    with pdfplumber.open(file_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

def extract_text_from_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text()

def extract_text_from_docx(file_path: str) -> str:
    return textract.process(file_path).decode('utf-8')

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\W+', ' ', text)
    return text

def tokenize_text(text: str, tokenizer_name: str) -> List[str]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer.tokenize(text)

def embed_text(text: str, model_name: str) -> np.ndarray:
    model = HuggingFaceEmbeddings(model_name)
    return model.embed(text)

def classify_text(text: str, model_name: str) -> Dict[str, float]:
    classifier = pipeline('text-classification', model=model_name)
    return classifier(text)[0]

def generate_text(prompt: str, model_name: str) -> str:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def sentiment_analysis(text: str, model_name: str) -> Dict[str, float]:
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return {model.config.id2label[i]: score.item() for i, score in enumerate(scores[0])}