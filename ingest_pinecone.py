import os
import fitz  # PyMuPDF
import re
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import ollama

# Configuration
API_KEY = "pcsk_6B66P9_CdVUvTQb5uBC615FedRCbKrBgbwpp5BDwn1E1jJsnTT4S9wVJHKKmyB4gQmUCDH"
INDEX_NAME = "classnotes"
DIMENSION = 768

# Initialize Pinecone client
pc = Pinecone(api_key=API_KEY)
existing_indexes = pc.list_indexes().names()

if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Index '{INDEX_NAME}' created.")
else:
    print(f"Index '{INDEX_NAME}' already exists.")

# Connect to index
index = pc.Index(INDEX_NAME)

# Load embedding model
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> list:
    """Generates an embedding for a given text using SentenceTransformer."""
    return EMBEDDING_MODEL.encode(text).tolist()


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF and returns a dictionary mapping pages to text."""
    doc = fitz.open(pdf_path)
    text_by_page = {page_num: page.get_text("text") for page_num, page in enumerate(doc)}
    return text_by_page


def split_text_into_chunks(text, chunk_size=400, overlap=50):
    """Splits text into overlapping chunks."""
    words = text.split()
    return [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]


def store_embedding(file, page, chunk, chunk_id):
    """Stores document chunks as vectors in Pinecone."""
    vector_id = "".join(c if c.isalnum() or c == "-" else "_" for c in chunk_id)
    embedding = get_embedding(chunk)

    index.upsert(vectors=[(vector_id, embedding, {"file": file, "page": page, "text": chunk})])
    print(f"Stored embedding for {file}, page {page}, chunk {chunk_id}")


def process_pdfs(data_dir):
    """Processes all PDFs in a directory and stores their embeddings in Pinecone."""
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)

            for page_num, text in text_by_page.items():
                chunks = split_text_into_chunks(text)
                for chunk_index, chunk in enumerate(chunks):
                    chunk_id = f"{file_name}_p{page_num}_c{chunk_index}"
                    store_embedding(file_name, page_num, chunk, chunk_id)
            print(f"Processed {file_name}")


if __name__ == "__main__":
    pdf_directory = "C:\\Users\\omint\\OneDrive\\Documents\\P2_LLM\\ClassNotes"
    process_pdfs(pdf_directory)
    print("\n--- Done processing PDFs ---\n")
