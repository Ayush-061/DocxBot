import streamlit as st
import numpy as np
import faiss

from openai_api import generate_answer, get_embedding
from document_loader import load_pdf
from embeddings import embed_texts

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

st.set_page_config(page_title="PDF Question Answering", layout="centered")

# Title and description
st.title("📄 PDF Question Answering")
st.write(
    "Upload any PDF document, create semantic embeddings of its content, "
    "and ask questions to get relevant answers."
)

# Upload section with container
with st.expander("1. Upload PDF File", expanded=True):
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    full_text = load_pdf(uploaded_file)
    st.success("✅ PDF loaded successfully!")

    chunks = chunk_text(full_text)
    st.info(f"Document split into **{len(chunks)}** chunks.")

    # Embeddings and indexing section in its own expander
    with st.expander("2. Create Embeddings and Build Index", expanded=True):
        if st.button("Create Embeddings & Build Index"):
            with st.spinner("Creating embeddings — this might take some time..."):
                embeddings = embed_texts(chunks)

            dimension = len(embeddings[0])
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings))

            st.success("✅ Embeddings created and index built!")

            st.session_state["chunks"] = chunks
            st.session_state["index"] = index

    # Question answering section in a third expander
    if "index" in st.session_state:
        with st.expander("3. Ask Questions", expanded=True):
            question = st.text_input("Ask a question about the document:")

            if question:
                with st.spinner("Searching for the best answer..."):
                    question_emb = np.array(get_embedding(question)).reshape(1, -1)
                    distances, indices = st.session_state["index"].search(question_emb, k=3)
                    matched_chunks = [st.session_state["chunks"][i] for i in indices[0]]
                    context = "\n\n".join(matched_chunks)

                    answer = generate_answer(question, context)

                st.markdown("### Answer:")
                st.write(answer)
else:
    st.info("Please upload a PDF file to get started.")
