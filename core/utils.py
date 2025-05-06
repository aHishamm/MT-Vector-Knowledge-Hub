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
import together
from langchain.embeddings import HuggingFaceEmbeddings
from sqlmodel import SQLModel, Field, Session, select, create_engine
from sqlmodel.engine.result import ScalarResult
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import re
import nltk
from bs4 import BeautifulSoup
import textract
import pdfplumber
from fastapi import Depends, HTTPException, status