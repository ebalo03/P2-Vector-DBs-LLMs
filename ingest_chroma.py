import chromadb
import ollama
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="Class_Notes")

# Constants
VECTOR_DIM = 768
DISTANCE_METRIC = "COSINE"


def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    """Generates an embedding for a given text using Ollama."""
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def search_embeddings(query: str, top_k=3):
    """Searches ChromaDB for the most relevant stored chunks."""
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "distances"],
    )

    top_results = [
        {
            "file": metadata.get("file", "Unknown"),
            "page": metadata.get("page", "Unknown"),
            "chunk": metadata.get("chunk", "Unknown"),
            "similarity": float(results["distances"][0][i]),
        }
        for i, metadata in enumerate(results["metadatas"][0])
    ]

    # Print results
    for result in top_results:
        print(
            f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}, Similarity: {result['similarity']:.2f}"
        )

    return top_results


def generate_rag_response(query, context_results):
    """Generates a RAG-based response using retrieved document context."""
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {result.get('similarity'):.2f}"
            for result in context_results
        ]
    )

    print(f"context_str: {context_str}")

    prompt = f"""You are a helpful AI assistant.
    Use the following context to answer the query as accurately as possible. 
    If the context is not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    response = ollama.chat(
        model="mistral:latest", messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def interactive_search():
    """Interactive search interface."""
    print("🔍 ChromaDB RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        # Search for relevant embeddings
        context_results = search_embeddings(query)

        # Generate RAG response
        response = generate_rag_response(query, context_results)

        print("\n--- Response ---")
        print(response)


if __name__ == "__main__":
    interactive_search()
