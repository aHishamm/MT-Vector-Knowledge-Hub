# MT Vector Knowledge Hub
<p align="center">
<img src="https://github.com/user-attachments/assets/1d61a544-927b-4a7b-872c-36449bb2d3e3" alt="MT Vector logo" width="450"/>
</p>

## Project Description
MT Vector Knowledge Hub is a document retrieval and question-answering system built on modern vector database technology. The system embeds documents into a vector space using state-of-the-art language models, stores them in a PostgreSQL database, and enables semantic search and retrieval based on natural language queries.
This knowledge base application leverages transformer models from Hugging Face for embeddings and LLMs from Together AI for generating responses, creating a tool for knowledge management and information retrieval.
### Not implemented yet
- **Vector Similarity Search**: Planned; embeddings are stored but not yet queried using true vector search (e.g., pgvector)
- **Document Reranking**: Not yet implemented
- **Full Django Migration**: Some utility functions and background processing require further adaptation from the previous FastAPI/core structure
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
   docker compose up --build -d
   ```
### Django Setup (First Time Only)
1. Run migrations:
   ```
   docker compose exec app python manage.py migrate
   ```
2. (Optional) Create a superuser for the admin interface:
   ```
   docker compose exec app python manage.py createsuperuser
   ```
### Usage
Once the application is running, you can access the API at `http://localhost:8000`.
#### Example API Endpoints
- `GET /` — API root
- `POST /process-directory/` — Process a directory of documents
- `POST /generate-embeddings/` — Generate vector embeddings for text
- `POST /chat/` — Chat with the LLM
- `GET /documents/` — List documents
- `GET /documents/<id>/` — Retrieve a document by ID
### Admin Interface
Visit `http://localhost:8000/admin/` to manage documents (after creating a superuser).
## Architecture
The application follows a modular Django architecture:
- `coreapi/` — Main Django app with models, views, serializers, and API logic
- `mt_vector_kb/` — Django project configuration (settings, URLs, WSGI/ASGI)
- `docker/` — Database initialization scripts