# embeddings.py

from gemini_api import embed_texts
import numpy as np

def embed_texts_wrapper(texts):
    
    embeddings = embed_texts(texts)  # This returns a list of lists
    return [np.array(embedding, dtype=np.float32) for embedding in embeddings]
