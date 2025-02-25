import json
import os
from typing import List, Dict, Any

from langchain.schema import Document
# from langchain_community.vectorstores import Chroma

# Helper function to check if the vectorstore has content
def check_vectorstore(vectorstore):
    """Check if the vectorstore has content and print diagnostics."""
    try:
        # Check if we can get the Chroma collection
        collection = vectorstore._collection
        print(f"Collection name: {collection.name}")
        print(f"Collection count: {collection.count()}")
        
        # Try a simple search with a very general query to test functionality
        results = vectorstore.similarity_search("insurance", k=1)
        print(f"Test query results count: {len(results)}")
        
        # Check the embedding function
        embedding_function = vectorstore._embedding_function
        print(f"Embedding function type: {type(embedding_function)}")
        
        # Try to retrieve all documents (might be slow for large collections)
        try:
            # Getting all IDs from the collection
            all_ids = collection.get(include=[])["ids"]
            print(f"Total document IDs in collection: {len(all_ids)}")
            
            if len(all_ids) > 0:
                # Get a sample document to verify retrieval works
                sample = collection.get(ids=[all_ids[0]], include=["documents", "metadatas"])
                print(f"Sample document exists: {bool(sample['documents'])}")
            else:
                print("No document IDs found in collection")
        except Exception as e:
            print(f"Error retrieving all documents: {e}")
            
        return collection.count() > 0
    except Exception as e:
        print(f"Error checking vectorstore: {e}")
        return False

# Function to create a fresh vector store with debugging info
def create_vector_store_with_debug(documents: List[Document], persist_directory: str = "./chroma_db"):
    """Create and persist a vector store from the documents with debug info."""
    print(f"Creating vector store with {len(documents)} documents")
    
    # Print a sample of the first document to verify content
    if documents:
        print(f"Sample document: {documents[0].page_content[:100]}...")
        print(f"Sample metadata: {documents[0].metadata}")
    
    # Make sure the directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    try:
        # Use default OpenAI embeddings (requires API key)
        print("Initializing OpenAI embeddings...")
        embeddings = OpenAIEmbeddings()
        
        # Test the embedding function
        test_embedding = embeddings.embed_query("test")
        print(f"Embedding test successful. Vector dimension: {len(test_embedding)}")
        
        # Create the vector store
        print("Creating Chroma vector store...")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        # Verify it has content
        print("Verifying vector store content...")
        has_content = check_vectorstore(vectorstore)
        
        # Persist the vector store
        print("Persisting vector store...")
        vectorstore.persist()
        
        print(f"Vector store creation {'successful' if has_content else 'failed - no content'}")
        return vectorstore
    except Exception as e:
        print(f"Error creating vector store: {e}")
        raise

# Function to load an existing vector store with debugging
def load_vector_store_with_debug(persist_directory: str = "./chroma_db"):
    """Load an existing vector store with debugging information."""
    print(f"Loading vector store from {persist_directory}")
    
    if not os.path.exists(persist_directory):
        print(f"Error: Directory {persist_directory} does not exist")
        return None
    
    try:
        print("Initializing OpenAI embeddings...")
        embeddings = OpenAIEmbeddings()
        
        print("Loading Chroma vector store...")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
        # Check if it has content
        has_content = check_vectorstore(vectorstore)
        print(f"Vector store loading {'successful' if has_content else 'failed - no content'}")
        
        return vectorstore
    except Exception as e:
        print(f"Error loading vector store: {e}")
        raise

# Improved query function with debugging
def query_vector_store_with_debug(query: str, vectorstore, k: int = 4):
    """Query the vector store with debugging information."""
    print(f"Querying vector store with: '{query}'")
    
    try:
        # First check if the vector store has content
        if not check_vectorstore(vectorstore):
            print("Warning: Vector store appears to be empty")
            return []
        
        # Try the query
        print(f"Executing similarity search with k={k}...")
        results = vectorstore.similarity_search(query, k=k)
        
        print(f"Query returned {len(results)} results")
        if results:
            print(f"First result similarity score: {vectorstore._collection.get(include=['distances'])}")
        
        return results
    except Exception as e:
        print(f"Error querying vector store: {e}")
        return []

# Example usage function
def example_usage():
    # Path to your JSON data
    json_path = "insurance_data.json"
    
    # Save your data to a JSON file if it doesn't exist
    if not os.path.exists(json_path):
        print(f"Creating example data file at {json_path}")
        with open(json_path, "w") as f:
            json.dump([
                {
                    "carrier_name": "Example",
                    "guide": "# Example Insurance\n\n## Overview\nThis is a test.\n\n## Coverage\nTest coverage."
                }
            ], f)
    
    # Load data
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # Process into documents
    documents = []
    for item in data:
        documents.append(
            Document(
                page_content=item.get("guide", ""),
                metadata={"carrier_name": item.get("carrier_name", "")}
            )
        )
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Split documents
    chunked_docs = text_splitter.split_documents(documents)
    
    # Create or load vector store
    persist_dir = "./chroma_db"
    
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(f"Loading existing vector store from {persist_dir}")
        vectorstore = load_vector_store_with_debug(persist_dir)
    else:
        print(f"Creating new vector store at {persist_dir}")
        vectorstore = create_vector_store_with_debug(chunked_docs, persist_dir)
    
    # Test query
    if vectorstore:
        results = query_vector_store_with_debug("insurance coverage", vectorstore, k=2)
        
        print("\nQuery Results:")
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Carrier: {doc.metadata.get('carrier_name', 'N/A')}")
            print(f"Content: {doc.page_content[:100]}...")
    else:
        print("Failed to create or load vector store")

if __name__ == "__main__":
    example_usage()