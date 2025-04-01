import chromadb
import numpy as np
import os
import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB
DB_PATH = "./chroma_db"
client = chromadb.PersistentClient(path=DB_PATH)

# Load embedding model
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Fixed collection name
COLLECTION_NAME = "ClassNotes"


# Clears the ChromaDB collection
def clear_chroma_store():
    print(f"Clearing ChromaDB collection '{COLLECTION_NAME}'...")
    try:
        client.delete_collection(name=COLLECTION_NAME)  # Delete only "ClassNotes"
        print("ChromaDB collection cleared.")
    except Exception as e:
        print(f"Error clearing ChromaDB store: {e}")


# Generate an embedding for a text chunk
def get_embedding(text: str) -> list:
    return EMBEDDING_MODEL.encode(text).tolist()


# Store embeddings in ChromaDB
def store_embedding(file: str, page: int, chunk: str, embedding: list, chunk_id: str):
    collection = client.get_or_create_collection(name=COLLECTION_NAME)  # Always use "ClassNotes"

    collection.add(
        documents=[chunk],
        metadatas=[{"file": file, "page": page}],
        ids=[chunk_id]
    )
    print(f"Stored embedding for {file}, page {page}, chunk {chunk_id}")


# Extract text from a PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_by_page = {}

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        clean_text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
        words = clean_text.split()
        text_by_page[page_num] = words

    return text_by_page


# Split text into overlapping chunks
def split_text_into_chunks(text, chunk_size=600, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i: i + chunk_size])
        chunks.append(chunk)
    return chunks


# Process all PDFs in a directory
def process_pdfs(data_dir):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page.items():
                chunks = split_text_into_chunks(" ".join(text))
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk)
                    chunk_id = f"{file_name}_p{page_num}_c{chunk_index}"
                    store_embedding(file_name, page_num, chunk, embedding, chunk_id)
            print(f"Processed {file_name}")


# Query ChromaDB for relevant chunks
def query_chroma(query_text: str):
    collection = client.get_collection(name=COLLECTION_NAME)  # Always query "ClassNotes"
    embedding = get_embedding(query_text)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=5  # Retrieve top 5 relevant chunks
    )

    if not results["documents"]:
        print("No relevant documents found.")
        return

    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        print(f"\n🔹 Match {i + 1} (File: {meta['file']}, Page: {meta['page']}):")
        print(f"Chunk: {doc}\n")


# Main function
def main():
    clear_chroma_store()
    pdf_directory = "C:\\Users\\omint\\OneDrive\\Documents\\P2_LLM\\ClassNotes"
    process_pdfs(pdf_directory)
    print("\n---Done processing PDFs---\n")

    # Run a query
    query_chroma("What is a binary search tree?")


if __name__ == "__main__":
    main()
