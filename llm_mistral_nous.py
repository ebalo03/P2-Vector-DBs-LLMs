#OLLAMA

# store embeddings in redis
# enter query, response thru local llm
# can specify model in command line or def mistral
import os
import re
import fitz  # PyMuPDF
import numpy as np
import redis
import ollama
from redis.commands.search.query import Query

redis_client = redis.Redis(host="localhost", port=6379, db=0)
VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

def clear_redis_store():
    print("🧹 Clearing Redis...")
    redis_client.flushdb()
    print("✅ Redis cleared.")

def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA file TEXT page TEXT chunk TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("✅ HNSW index created.")

def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_by_page = {}

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        clean_text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        words = clean_text.split()
        text_by_page[page_num] = words

    return text_by_page

def split_text_into_chunks(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

# ------------------- Store in Redis -------------------
def store_embedding(file: str, page: str, chunk_text: str, embedding: list):
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk_text[:30]}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk": chunk_text,
            "embedding": np.array(embedding, dtype=np.float32).tobytes(),
        },
    )

def process_pdfs(data_dir):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)

            for page_num, words in text_by_page.items():
                chunks = split_text_into_chunks(" ".join(words))
                for chunk in chunks:
                    embedding = get_embedding(chunk)
                    store_embedding(file_name, str(page_num), chunk, embedding)

            print(f"Processed {file_name}")
\
def query_redis(query_text: str, top_k=5):
    embedding = get_embedding(query_text)
    query_vector = np.array(embedding, dtype=np.float32).tobytes()

    q = (
        Query(f"*=>[KNN {top_k} @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("file", "page", "chunk", "vector_distance")
        .dialect(2)
    )

    results = redis_client.ft(INDEX_NAME).search(q, query_params={"vec": query_vector})

    return [
        {
            "file": doc.file,
            "page": doc.page,
            "chunk": doc.chunk,
            "similarity": float(doc.vector_distance),
        }
        for doc in results.docs
    ]
# rag response prompt: used from other source
def generate_rag_response(query, context_chunks, model="mistral"):
    context = "\n".join([chunk["chunk"] for chunk in context_chunks])

    prompt = f"""You are a helpful assistant answering questions based on the class notes below.
If the notes do not contain the answer, say "I don't know."

Class Notes:
{context}

Question: {query}

Answer:"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

def main():
    pdf_dir = input("📁 Enter path to PDF folder: ").strip()

    # LLM CHOICE
    model_choice = input("🤖 Choose model (mistral / nous-hermes-2): ").strip().lower()
    if model_choice not in ["mistral", "nous-hermes-2"]:
        print("Invalid model. Default to mistral.")
        model_choice = "mistral"

    clear_redis_store()
    create_hnsw_index()
    process_pdfs(pdf_dir)

    print("\nAll processed & indexed.")

    while True:
        query = input("\nAsk a question: ")
        if query.lower() in ["exit", "quit"]:
            break

        chunks = query_redis(query)
        answer = generate_rag_response(query, chunks, model=model_choice)
        print("\n📣 Answer:\n", answer)

if __name__ == "__main__":
    main()
