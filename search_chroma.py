import chromadb
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="class_notes")

# Constants
VECTOR_DIM = 768


def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    """Generates an embedding for the given text."""
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def search_embeddings(query, top_k=3):
    """Searches ChromaDB for relevant stored chunks."""
    query_embedding = get_embedding(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    matches = results["documents"]

    if not matches:
        print("No relevant documents found.")
        return []

    top_results = []
    for doc, meta, dist in zip(matches[0], results["metadatas"][0], results["distances"][0]):
        top_results.append({
            "file": meta.get("file", "Unknown"),
            "page": meta.get("page", "Unknown"),
            "chunk": doc,
            "similarity": dist,
        })

    return top_results


def generate_rag_response(query, context_results):
    """Generates a response using the retrieved context."""
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}) with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )

    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    response = ollama.chat(model="tinyllama:latest", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


def interactive_search():
    """Interactive search interface."""
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        context_results = search_embeddings(query)
        response = generate_rag_response(query, context_results)
        print("\n--- Response ---")
        print(response)


if __name__ == "__main__":
    interactive_search()
