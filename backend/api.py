print("1. Starting imports...")
from fastapi import FastAPI, HTTPException
print("2. FastAPI imported")


from fastapi.middleware.cors import CORSMiddleware
print("3. Importing agents...")
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from agents import Agent,AsyncOpenAI,OpenAIChatCompletionsModel,RunConfig,function_tool,Runner
print("4. Imports done")
# from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import cohere

load_dotenv()

# Configuration
groq_api_key = os.getenv("GROQ_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# Initialize clients
print("Creating Qdrant client...")
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
print("✓ Qdrant connected")
print("Creating Cohere client...")
cohere_client = cohere.Client(cohere_api_key)
print("✓ Cohere connected")

external_client = AsyncOpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)

model = OpenAIChatCompletionsModel(
    model="llama-3.3-70b-versatile",
    openai_client=external_client
)

config = RunConfig(model=model)

# FastAPI app
app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# CORS middleware - allows frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    top_k: Optional[int] = 3

class ChatResponse(BaseModel):
    response: str
    sources: List[dict] = []

class HealthResponse(BaseModel):
    status: str
    message: str

# Helper functions
def get_query_embedding(query: str) -> List[float]:
    """Generate embedding for query."""
    try:
        response = cohere_client.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        return response.embeddings[0]
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return []

@function_tool
def retrieve_documents(query: str, top_k: int = 3) -> str:
    """Retrieve relevant documents from knowledge base."""
    try:
        collection_name = "ai-book"
        query_vector = get_query_embedding(query)
        
        if not query_vector:
            return "Failed to generate query embedding."
        
        search_results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k
        ).points
        
        if not search_results:
            return "No relevant documents found."
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            text = result.payload.get('text', '')
            url = result.payload.get('url', '')
            title = result.payload.get('title', 'Unknown')
            score = result.score
            
            context_parts.append(
                f"Source {i} (Relevance: {score:.2f}):\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content: {text[:500]}...\n"
            )
        
        return "\n" + "-"*80 + "\n".join(context_parts)
        
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"

# Create agent
agent = Agent(
    name="robotics-tutor",
    instructions="""You are an expert tutor for humanoid AI and robotics made by ayisha.
    Use the retrieve_documents tool to find relevant information, then provide clear,
    comprehensive answers. Always cite sources with URLs.""",
    tools=[retrieve_documents]
)

# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "RAG Chatbot API is running"
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check."""
    try:
        # Test Qdrant connection
        qdrant_client.get_collections()
        return {
            "status": "healthy",
            "message": "All services connected"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    
    Send a message and get AI response with sources.
    """
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Run agent
        result = await Runner.run(
            agent,
            request.message,
            run_config=config
        )
        
        # Extract sources (if available in result)
        sources = []
        
        return {
            "response": result.final_output,
            "sources": sources
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/search")
async def search_documents(request: ChatRequest):
    """
    Search endpoint - only retrieves documents without AI response.
    """
    try:
        query_vector = get_query_embedding(request.message)
        
        if not query_vector:
            raise HTTPException(status_code=500, detail="Failed to generate embedding")
        
        search_results = qdrant_client.query_points(
            collection_name="ai-book",
            query=query_vector,
            limit=request.top_k
        ).points
        
        documents = []
        for result in search_results:
            documents.append({
                "score": result.score,
                "title": result.payload.get('title', ''),
                "url": result.payload.get('url', ''),
                "text": result.payload.get('text', '')[:300] + "...",
                "chunk_id": result.payload.get('chunk_id', 0)
            })
        
        return {
            "query": request.message,
            "results": documents
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/debug")
async def debug():
    """Debug endpoint to check environment variables."""
    return {
        "groq_key_set": groq_api_key is not None,
        "groq_key_length": len(groq_api_key) if groq_api_key else 0,
        "cohere_key_set": cohere_api_key is not None,
        "cohere_key_length": len(cohere_api_key) if cohere_api_key else 0
    }     

# Run with: uvicorn api:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)