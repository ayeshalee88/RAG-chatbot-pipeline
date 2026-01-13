
import os 
from dotenv import load_dotenv
from agents import Agent,AsyncOpenAI,OpenAIChatCompletionsModel,RunConfig,function_tool,Runner
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import cohere 
import uuid
from typing import List, Dict
import time
import requests


load_dotenv()
api_key=os.getenv("GROK_API_KEY")

cohere_api_key="QR4xhtzuCUpO97QirSj4sSThMz2JPjEdvJ6xdARe"
embed_model="emb engv1"

qdrant_client=QdrantClient(
    url="https://1ec726c8-5b62-4d5d-924c-4d5f0243d102.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.uQba9hsNRzJi0RdkQ2xoPQPD5bdo6ZrZpvbhXn2e9Qk"
)
cohere_client = cohere.Client(cohere_api_key)


if not api_key:
     raise ValueError("GROK_API_KEYis not set. Please ensure it is defined in your .env file.")

#Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
api_key=api_key,
base_url="https://api.groq.com/openai/v1",
)

model = OpenAIChatCompletionsModel(
model="llama-3.3-70b-versatile",
openai_client=external_client
)

config = RunConfig(
model=model,
model_provider=external_client,
tracing_disabled=True
)

def get_query_embedding(query: str, model: str = "embed-english-v3.0") -> List[float]:
    """
    Generate embedding for a query string.
    
    Args:
        query: The search query
        model: Cohere embedding model to use
        
    Returns:
        List of floats representing the query embedding
    """
    try:
        response = cohere_client.embed(
            texts=[query],
            model=model,
            input_type="search_query"  # Note: different from "search_document"
        )
        return response.embeddings[0]
    except Exception as e:
        print(f"Error creating query embedding: {e}")
        return []
    
@function_tool
def retrieve_documents(query: str, collection_name: str = "ai-book", top_k: int = 5) -> List[Dict]:
    """
    Retrieve most relevant documents from Qdrant based on query.
    
    Args:
        query: The search query
        collection_name: Name of the Qdrant collection
        top_k: Number of top results to return
        
    Returns:
        List of dictionaries containing retrieved documents with metadata
    """
    try:
        # Get query embedding
        print(f"Generating embedding for query: '{query}'")
        query_vector = get_query_embedding(query)
        
        if not query_vector:
            print("Failed to generate query embedding")
            return []
        
        # Search in Qdrant
        print(f"Searching in collection '{collection_name}'...")
        search_results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k
        ).points
        
        # Format results
        documents = []
        for i, result in enumerate(search_results):
            doc = {
                'rank': i + 1,
                'score': result.score,
                'text': result.payload.get('text', ''),
                'url': result.payload.get('url', ''),
                'title': result.payload.get('title', ''),
                'chunk_id': result.payload.get('chunk_id', 0)
            }
            documents.append(doc)
        
        print(f"Found {len(documents)} relevant documents")
        return documents
        
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []

agent=Agent(
    name="cohere-embedding-agent",
    instructions="You are a tutor for humainoid ai and robotics.when a user asks a question call retrieve_tool",
    tools=[retrieve_documents]
)

result=Runner.run_sync(
    agent,
    "does there is ros topic?",
    run_config=config
)

print("Agent Result:", result)

