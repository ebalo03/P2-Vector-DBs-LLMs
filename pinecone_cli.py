import os
import fitz  # PyMuPDF
import re
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Configuration
API_KEY = "pcsk_6B66P9_CdVUvTQb5uBC615FedRCbKrBgbwpp5BDwn1E1jJsnTT4S9wVJHKKmyB4gQmUCDH"  # Replace with actual API key
INDEX_NAME = "classnotes"
DIMENSION = 768  # Adjust based on embedding model

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


# Function to embed text
def get_embedding(text: str):
    """Returns the embedding of the given text."""
    return EMBEDDING_MODEL.encode(text).tolist()


# Function to add document embeddings
def add_document(file_name, text_chunks):
    """Adds a document's text chunks to Pinecone."""
    vectors = []
    for chunk_index, chunk in enumerate(text_chunks):
        chunk_id = f"{file_name}_c{chunk_index}"
        vector_id = "".join(c if c.isalnum() or c == "-" else "_" for c in chunk_id)
        embedding = get_embedding(chunk)
        vectors.append((vector_id, embedding, {"file": file_name, "text": chunk}))

    index.upsert(vectors)
    print(f"Stored {len(vectors)} chunks from {file_name}")


# Function to retrieve relevant documents
def query_documents(query_text, top_k=5):
    """Queries Pinecone for relevant document chunks."""
    query_embedding = get_embedding(query_text)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    for i, match in enumerate(results.get("matches", [])):
        meta = match.get("metadata", {})
        print(f"\n🔹 Match {i + 1}: File: {meta.get('file', 'Unknown')}")
        print(f"Chunk: {meta.get('text', 'No text available')}\n")


# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF and returns it as a list of words."""
    doc = fitz.open(pdf_path)
    text = " ".join(page.get_text("text") for page in doc)
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Clean text


# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=400, overlap=50):
    """Splits text into overlapping chunks for better context retention."""
    words = text.split()
    return [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]


# Function to process all PDFs in a directory
def process_pdfs(directory):
    """Processes and stores all PDFs in the given directory."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    for file_name in os.listdir(directory):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(directory, file_name)
            text = extract_text_from_pdf(pdf_path)
            text_chunks = split_text_into_chunks(text)
            add_document(file_name, text_chunks)

    print(f"Processed all PDFs in {directory}")


# Main execution
if __name__ == "__main__":
    data_dir = "C:\\Users\\omint\\OneDrive\\Documents\\P2_LLM\\ClassNotes"
    process_pdfs(data_dir)

    # Run a query
    print("\nQuerying stored documents...")
    query_documents("What is a binary search tree?")

