# gemini_api.py
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize chat model (Gemini 2.5 Flash)
model = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_genai",
    config={"google_api_key": GOOGLE_API_KEY}
)

# Initialize embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

def get_embedding(text: str):
   
    return embedding_model.embed_query(text)

def embed_texts(texts: list[str]):
    
    return embedding_model.embed_documents(texts)

def generate_answer(question: str, context: str):
    
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\nAnswer:"
    response = model.invoke([
        HumanMessage(content=prompt)
    ])
    return response.content
