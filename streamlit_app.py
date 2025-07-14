"""Streamlit UI for the RAG PDF demo."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_MODEL = "all-MiniLM-L6-v2"


def load_pdfs(pdf_paths: List[Path]):
    """Load PDF files and return list of LangChain Documents."""
    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(str(path))
        docs.extend(loader.load())
    return docs


def build_qa_chain(docs):
    """Create RetrievalQA chain given raw documents."""
    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = text_splitter.split_documents(docs)
    embedding = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma.from_documents(split_docs, embedding)
    llm = OpenAI()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
    return qa


def main():
    st.title("📄🧠 RAG PDF Q&A")
    st.write(
        "Upload one or more PDF files, ask a question, and get an answer powered by Retrieval-Augmented Generation."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF file(s)", type="pdf", accept_multiple_files=True, key="pdf_uploader"
    )

    if uploaded_files:
        # Save uploaded files to temp dir (Streamlit returns BytesIO objects)
        temp_dir = Path(tempfile.mkdtemp())
        file_paths: List[Path] = []
        for uploaded in uploaded_files:
            path = temp_dir / uploaded.name
            with open(path, "wb") as f:
                f.write(uploaded.getbuffer())
            file_paths.append(path)

        # Build / retrieve QA chain and cache by filenames in session_state
        file_key = tuple(sorted(p.name for p in file_paths))
        if (
            "qa_chain" not in st.session_state
            or st.session_state.get("qa_chain_files") != file_key
        ):
            with st.spinner("Building vector store…"):
                docs = load_pdfs(file_paths)
                st.session_state["qa_chain"] = build_qa_chain(docs)
                st.session_state["qa_chain_files"] = file_key
            st.success("Vector store ready!")

        question = st.text_input("Ask a question about the uploaded PDFs:")
        if question:
            qa_chain: RetrievalQA = st.session_state["qa_chain"]
            with st.spinner("Generating answer…"):
                answer = qa_chain.run(question)
            st.markdown("### Answer")
            st.write(answer)

            # Optionally show sources
            # st.markdown("### Source chunks")
            # for d in qa_chain.retriever.get_relevant_documents(question):
            #     st.write(d.page_content)
    else:
        st.info("Please upload at least one PDF to begin.")


if __name__ == "__main__":
    main()
