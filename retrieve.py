from qdrant_client import QdrantClient
import os
from qdrant_client.models import Distance, VectorParams, PointStruct
import cohere
import requests
from bs4 import BeautifulSoup
from xml.etree import ElementTree
from urllib.parse import urlparse
import uuid
from typing import List, Dict
import time

sitemapurl="https://ai-book-ochre.vercel.app/sitemap.xml"
collection="ai-book"

cohere_api_key="QR4xhtzuCUpO97QirSj4sSThMz2JPjEdvJ6xdARe"
embed_model="emb engv1"

qdrant_client=QdrantClient(
    url="https://1ec726c8-5b62-4d5d-924c-4d5f0243d102.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.uQba9hsNRzJi0RdkQ2xoPQPD5bdo6ZrZpvbhXn2e9Qk"
)
cohere_client = cohere.Client(cohere_api_key)

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


def retrieve_with_context(query: str, collection_name: str = "ai-book", top_k: int = 3) -> str:
    """
    Retrieve documents and format them as context for LLM.
    
    Args:
        query: The search query
        collection_name: Name of the Qdrant collection
        top_k: Number of top results to return
        
    Returns:
        Formatted context string ready for LLM prompt
    """
    documents = retrieve_documents(query, collection_name, top_k)
    
    if not documents:
        return "No relevant documents found."
    
    # Format context
    context_parts = []
    for doc in documents:
        context_parts.append(
            f"[Source {doc['rank']} - {doc['title']} (Score: {doc['score']:.3f})]\n"
            f"URL: {doc['url']}\n"
            f"{doc['text']}\n"
        )
    
    return "\n---\n".join(context_parts)


def search(query: str, top_k: int = 5, show_details: bool = True):
    """
    Convenient search function that prints results.
    
    Args:
        query: The search query
        top_k: Number of top results to return
        show_details: Whether to print detailed results
    """
    print(f"\n{'='*80}")
    print(f"SEARCH QUERY: {query}")
    print(f"{'='*80}\n")
    
    documents = retrieve_documents(query, collection, top_k)
    
    if not documents:
        print("No results found.")
        return documents
    
    if show_details:
        for doc in documents:
            print(f"\n{'-'*80}")
            print(f"Rank: {doc['rank']} | Score: {doc['score']:.4f}")
            print(f"Title: {doc['title']}")
            print(f"URL: {doc['url']}")
            print(f"Chunk ID: {doc['chunk_id']}")
            print(f"\nText Preview:")
            print(doc['text'][:300] + "..." if len(doc['text']) > 300 else doc['text'])
            print(f"{'-'*80}")
    else:
        for doc in documents:
            print(f"{doc['rank']}. [{doc['score']:.4f}] {doc['title']}")
    
    return documents


# Example usage
if __name__ == "__main__":
    # Example 1: Simple search
    print("Example 1: Simple Search")
    results = search("What is reinforcement learning?", top_k=3)
    
    # Example 2: Get formatted context for LLM
    print("\n\nExample 2: Get Context for LLM")
    context = retrieve_with_context("How does computer vision work in robotics?", top_k=2)
    print("\n--- FORMATTED CONTEXT FOR LLM ---")
    print(context)
    
    # Example 3: Get raw embeddings
    print("\n\nExample 3: Get Query Embedding")
    embedding = get_query_embedding("humanoid robot design")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")