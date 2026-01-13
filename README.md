# RAG Chatbot System

A Retrieval-Augmented Generation (RAG) chatbot for humanoid AI and robotics documentation.

## Features
- Web scraping from sitemap
- Vector embeddings with Cohere
- Vector storage in Qdrant
- AI agent with Groq/Llama
- Semantic search and retrieval

## Setup

1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

2. Install dependencies
```bash
uv sync
# or
uv pip install -r requirements.txt
```

3. Create `.env` file
```env
GROQ_API_KEY=your_groq_key_here
COHERE_API_KEY=your_cohere_key_here
```

4. Ingest data
```bash
uv run main.py
```

5. Run chatbot
```bash
uv run chatbot.py
```

## Files
- `main.py` - Data ingestion pipeline
- `retrieve.py` - Document retrieval functions
- `chatbot.py` - AI agent chatbot

## APIs Used
- Groq (LLM)
- Cohere (Embeddings)
- Qdrant (Vector Database)
