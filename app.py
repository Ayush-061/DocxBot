import streamlit as st
import numpy as np
import faiss

from gemini_api import generate_answer, get_embedding
from document_loader import load_pdf
from embeddings import embed_texts_wrapper as embed_texts
from vector_store import (
    create_faiss_index, save_faiss_index, load_faiss_index,
    save_chunks, load_chunks
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title="DocxBot", layout="centered")

st.title("📄 PDF Question Answering")
st.write(
    "Upload any PDF document, create semantic embeddings of its content, "
    "and ask questions to get relevant answers."
)

def chunk_text(text, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

with st.expander("1. Upload PDF File", expanded=True):
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    if "chunks" not in st.session_state or "index" not in st.session_state:
        try:
            st.session_state["chunks"] = load_chunks()
            st.session_state["index"] = load_faiss_index()
            st.success("✅ Loaded existing index and chunks!")
        except FileNotFoundError:
            full_text = load_pdf(uploaded_file)
            chunks = chunk_text(full_text)
            st.session_state["chunks"] = chunks
            st.info(f"Document split into **{len(chunks)}** chunks.")

            with st.spinner("Creating embeddings and building index..."):
                index = create_faiss_index(chunks)
                save_chunks(chunks)
                save_faiss_index(index)
                st.session_state["index"] = index
                st.success("✅ Created and saved index and chunks.")

    with st.expander("3. Ask Questions", expanded=True):
        question = st.text_input("Ask a question about the document:")

        if question and "index" in st.session_state:
            with st.spinner("Searching for the best answer..."):
                question_emb = np.array(get_embedding(question)).astype("float32").reshape(1, -1)
                distances, indices = st.session_state["index"].search(question_emb, k=3)
                matched_chunks = [st.session_state["chunks"][i] for i in indices[0]]
                context = "\n\n".join(matched_chunks)

                # Improved prompt to guide Gemini to answer using context
                prompt = f"""You are a helpful assistant. Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"""

                answer = generate_answer(question, prompt)

            st.markdown("### Answer:")
            st.write(answer)

else:
    st.info("Please upload a PDF file to get started.")
