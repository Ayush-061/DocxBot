import os
import json
import faiss
import numpy as np

from gemini_api import embedding_model

CHUNKS_PATH = "saved_data/chunks.json"
INDEX_PATH = "saved_data/index.faiss"

def create_faiss_index(text_chunks):
    embeddings = embedding_model.embed_documents(text_chunks)
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def save_faiss_index(index, path=INDEX_PATH):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    faiss.write_index(index, path)
    print(f"Saved FAISS index to: {os.path.abspath(path)}")

def load_faiss_index(path=INDEX_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index file not found at {path}")
    return faiss.read_index(path)

def save_chunks(chunks, path=CHUNKS_PATH):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved chunks to: {os.path.abspath(path)}")

def load_chunks(path=CHUNKS_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chunks file not found at {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def search_faiss_index(index, query, k=3):
    query_embedding = embedding_model.embed_query(query)
    query_vector = np.array(query_embedding).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    return distances, indices
