# embeddings.py
from openai_api import get_embedding
import numpy as np

def embed_texts(texts):
  
    embeddings = get_embedding(texts)  # get_embedding supports batch inputs
    return [np.array(e) for e in embeddings]
