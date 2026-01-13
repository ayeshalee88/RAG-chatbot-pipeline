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

def get_urls_from_sitemap(sitemap_url: str, replace_domain: bool = True) -> List[str]:
    """Extract all URLs from a sitemap XML and optionally fix domain mismatches."""
    try:
        response = requests.get(sitemap_url, timeout=10)
        response.raise_for_status()
        
        root = ElementTree.fromstring(response.content)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        # Fix domain mismatch (sitemap has wrong domain)
        if replace_domain and urls:
            # Extract the actual domain from sitemap_url
            sitemap_domain = urlparse(sitemap_url).netloc
            
            # Replace the wrong domain in URLs
            fixed_urls = []
            for url in urls:
                parsed = urlparse(url)
                if parsed.netloc != sitemap_domain:
                    # Replace with correct domain
                    fixed_url = url.replace(parsed.netloc, sitemap_domain)
                    fixed_urls.append(fixed_url)
                    print(f"Fixed URL: {parsed.netloc} -> {sitemap_domain}")
                else:
                    fixed_urls.append(url)
            urls = fixed_urls
        
        print(f"Found {len(urls)} URLs in sitemap")
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

def extract_text_from_url(url: str) -> Dict[str, str]:
    """Extract text content from a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Get title
        title = soup.title.string if soup.title else url
        
        return {
            'url': url,
            'title': title,
            'text': text[:10000]  # Limit text length
        }
    except Exception as e:
        print(f"Error extracting text from {url}: {e}")
        return None

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks

def create_embeddings(texts: List[str], model: str = "embed-english-v3.0") -> List[List[float]]:
    """Create embeddings using Cohere."""
    try:
        response = cohere_client.embed(
            texts=texts,
            model=model,
            input_type="search_document"
        )
        return response.embeddings
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return []
    

def create_collection(client: QdrantClient, collection_name: str, vector_size: int = 1024):
    """Create a Qdrant collection if it doesn't exist."""
    try:
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if collection_name in collection_names:
            print(f"Collection '{collection_name}' already exists")
        else:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"Collection '{collection_name}' created successfully")
    except Exception as e:
        print(f"Error creating collection: {e}")


def save_to_qdrant(client: QdrantClient, collection_name: str, 
                   embeddings: List[List[float]], metadata: List[Dict]):
    """Save embeddings and metadata to Qdrant."""
    try:
        points = []
        for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=meta
            )
            points.append(point)
        
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Successfully saved {len(points)} points to Qdrant")
    except Exception as e:
        print(f"Error saving to Qdrant: {e}")



def pipeline(sitemap_url: str, collection_name: str, batch_size: int = 10):
    """
    Complete pipeline to process sitemap and store in Qdrant.
    
    Args:
        sitemap_url: URL of the sitemap
        collection_name: Name of the Qdrant collection
        batch_size: Number of documents to process in each batch
    """
    print("Starting RAG pipeline...")
    
    # Step 1: Get URLs from sitemap
    print("\n1. Fetching URLs from sitemap...")
    urls = get_urls_from_sitemap(sitemap_url)
    
    if not urls:
        print("No URLs found. Exiting.")
        return
    
    # Step 2: Create collection (with proper vector size for embed-english-v3.0)
    print("\n2. Creating collection...")
    create_collection(qdrant_client, collection_name, vector_size=1024)
    
    # Step 3: Process URLs in batches
    print("\n3. Processing URLs...")
    all_chunks = []
    all_metadata = []
    
    for i, url in enumerate(urls):
        print(f"Processing {i+1}/{len(urls)}: {url}")
        
        # Extract text
        doc = extract_text_from_url(url)
        if not doc:
            continue
        
        # Chunk text
        chunks = chunk_text(doc['text'])
        
        # Create metadata for each chunk
        for j, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                'url': doc['url'],
                'title': doc['title'],
                'chunk_id': j,
                'text': chunk
            })
        
        # Process in batches to avoid overwhelming the API
        if len(all_chunks) >= batch_size:
            print(f"Creating embeddings for batch of {len(all_chunks)} chunks...")
            embeddings = create_embeddings(all_chunks)
            
            if embeddings:
                print(f"Saving batch to Qdrant...")
                save_to_qdrant(qdrant_client, collection_name, embeddings, all_metadata)
            
            all_chunks = []
            all_metadata = []
            time.sleep(1)  # Rate limiting
    
    # Process remaining chunks
    if all_chunks:
        print(f"Creating embeddings for final batch of {len(all_chunks)} chunks...")
        embeddings = create_embeddings(all_chunks)
        
        if embeddings:
            print(f"Saving final batch to Qdrant...")
            save_to_qdrant(qdrant_client, collection_name, embeddings, all_metadata)
    
    print("\n✓ Pipeline completed successfully!")


# Run the pipeline
if __name__ == "__main__":
    pipeline(sitemapurl, collection)