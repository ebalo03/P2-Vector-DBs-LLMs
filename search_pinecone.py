import os
from pinecone import Pinecone, ServerlessSpec
import ollama
from sentence_transformers import SentenceTransformer

# Load API key securely
API_KEY = "pcsk_6B66P9_CdVUvTQb5uBC615FedRCbKrBgbwpp5BDwn1E1jJsnTT4S9wVJHKKmyB4gQmUCDH"
INDEX_NAME = "classnotes"
DIMENSION = 768

# Initialize Pinecone client
pc = Pinecone(api_key=API_KEY)
existing_indexes = pc.list_indexes().names()

# Create index if it doesn't exist
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Connect to the index
index = pc.Index(INDEX_NAME)

# Load embedding model
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> list:
    """Generates an embedding for a given text using SentenceTransformer."""
    return EMBEDDING_MODEL.encode(text).tolist()


def search_embeddings(query, top_k=3):
    """Queries Pinecone for the most relevant stored chunks."""
    query_embedding = get_embedding(query)

    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    matches = results.get("matches", [])

    if not matches:
        print("No relevant documents found.")
        return []

    top_results = [
        {
            "file": match.get("metadata", {}).get("file", "Unknown"),
            "page": match.get("metadata", {}).get("page", "Unknown"),
            "chunk": match.get("metadata", {}).get("text", "No text available"),
            "similarity": match.get("score", 0),
        }
        for match in matches
    ]

    return top_results


def generate_rag_response(query, context_results):
    """Generates a RAG-based response using Ollama."""
    if not context_results:
        return "I don't know."

    context_str = "\n".join(
        [
            f"From {result['file']} (page {result['page']}): {result['chunk']}"
            for result in context_results
        ]
    )

    prompt = f"""You are a helpful AI assistant. 
Use the following context to answer the query. If the context is not relevant, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    response = ollama.chat(model="tinyllama:latest", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


def interactive_search():
    """Interactive search interface."""
    print("🔍 Pinecone RAG Search")
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
