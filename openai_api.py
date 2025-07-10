
import os
from dotenv import load_dotenv
import openai

load_dotenv() 

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text_or_texts):
    if isinstance(text_or_texts, str):
        response = openai.embeddings.create(
            input=text_or_texts,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    elif isinstance(text_or_texts, list):
        response = openai.embeddings.create(
            input=text_or_texts,
            model="text-embedding-ada-002"
        )
        return [item.embedding for item in response.data]
    else:
        raise TypeError("Input must be a string or list of strings.")


def generate_answer(question, context):
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\nAnswer:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7,
    )
    return response['choices'][0]['message']['content']
