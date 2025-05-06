# MT Vector Knowledge Hub
<p align="center">
<img src="https://github.com/user-attachments/assets/1d61a544-927b-4a7b-872c-36449bb2d3e3" alt="MT Vector logo" width="450"/>
</p>
## Project Description

MT Vector Knowledge Hub is a sophisticated document retrieval and question-answering system built on modern vector database technology. The system embeds documents into a vector space using state-of-the-art language models, stores them in a PostgreSQL database, and enables semantic search and retrieval based on natural language queries. 

This knowledge base application leverages transformer models from Hugging Face for embeddings and LLMs from Together AI for generating responses, creating a powerful tool for knowledge management and information retrieval.

## Core Features

### Implemented Features

- [x] **Docker Containerization**: Complete application containerization with Docker and Docker Compose
- [x] **Basic Project Structure**: Core modules organized for extensibility and maintainability

### Features In Progress / To Be Implemented

- [ ] **FastAPI Backend**: 
  - [ ] RESTful API setup with FastAPI for handling requests and responses
  - [ ] API endpoint implementation for document upload and querying
  - [ ] Error handling and validation

- [ ] **Database Integration**: 
  - [ ] PostgreSQL database setup with SQLModel ORM for data persistence
  - [ ] Database schema design and implementation
  - [ ] Migration system for database changes

- [ ] **Document Processing Pipeline**:
  - [ ] Document loading from various sources (PDF, TXT, HTML)
  - [ ] Document text extraction and preprocessing
  - [ ] Document chunking for efficient embedding
  
- [ ] **Vector Embedding System**:
  - [ ] Integration of Hugging Face embedding models
  - [ ] Vector embedding generation for documents
  - [ ] Storage of embeddings in the database
  
- [ ] **Retrieval System**:
  - [ ] Vector similarity search implementation
  - [ ] Document reranking with crossencoder models
  - [ ] Relevance scoring mechanism
  
- [ ] **Question Answering**:
  - [ ] Integration with Together AI's LLM models
  - [ ] Prompt engineering for accurate responses
  - [ ] Context-aware answer generation
  
- [ ] **User Interface**:
  - [ ] Frontend for document upload and query
  - [ ] Visualization of search results
  - [ ] User authentication and permission management
  
- [ ] **Performance Optimization**:
  - [ ] Caching mechanisms for frequent queries
  - [ ] Batch processing for large document sets
  - [ ] Asynchronous task handling
  
- [ ] **Testing**:
  - [ ] Unit tests for core functionalities
  - [ ] Integration tests for end-to-end workflows
  - [ ] Load testing with Locust

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Together AI API key

### Installation

1. Clone the repository
2. Create a `.env` file with your Together AI API key:
   ```
   TOGETHER_API_KEY=your_together_api_key_here
   ```
3. Build and start the containers:
   ```
   docker-compose up --build
   ```

### Usage

Once the application is running, you can access the API at `http://localhost:8000` (after implementing the FastAPI backend).

## Architecture

The application follows a modular architecture:

- `core/app.py`: FastAPI application entry point (to be implemented)
- `core/models.py`: SQLModel database models (to be implemented)
- `core/settings.py`: Application configuration
- `core/utils.py`: Utility functions for vector operations and LLM integration

## Future Roadmap

- Knowledge graph integration for enhanced context understanding
- Multi-modal document support (images, audio)
- Fine-tuning capabilities for domain-specific knowledge
- API endpoint for bulk document processing
- Scheduled document refreshing and re-embedding

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
