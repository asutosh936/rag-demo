#!/usr/bin/env python3
"""
Retrieval-Augmented Generation (RAG) demo
========================================

This script builds a mini-RAG system over **user-supplied PDF files**.
It loads the PDFs, splits them into chunks, embeds them with a sentence-
transformer model, stores them in a local Chroma vector store and then
answers a user’s question with an OpenAI LLM + retrieval.

Usage
-----
python app.py <file1.pdf> [<file2.pdf> ...] --question "Your question here"

If `--question` is omitted you will be prompted interactively.

Prerequisites
-------------
1. Set environment variable `OPENAI_API_KEY`.
2. `pip install -r requirements.txt`
"""

from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from typing import List
import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA


def load_pdfs(pdf_paths: List[Path]):
    """Load and return a list of LangChain Documents from given PDF paths."""

    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(str(path))
        all_docs.extend(loader.load())
    return all_docs


def main() -> None:
    """Run the RAG pipeline over user-supplied PDFs and answer the question."""

    parser = argparse.ArgumentParser(description="Simple RAG demo over PDFs")
    parser.add_argument("pdfs", nargs="+", type=Path, help="Path(s) to PDF file(s)")
    parser.add_argument("--question", "-q", help="Question to ask about the PDFs")
    args = parser.parse_args()

    # 1. Load & split documents
    documents = load_pdfs(args.pdfs)
    if not documents:
        raise SystemExit("No documents could be loaded from the provided PDFs.")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # 2. Create embedding model and vector store (ChromaDB)
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embedding_function)

    # 3. Load OpenAI LLM (requires OPENAI_API_KEY env var)
    llm = OpenAI()

    # 4. Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    # 5. Ask a question (prompt user if not supplied)
    query = args.question or input("Enter your question: ")
    answer = qa_chain.run(query)

    print("\nQuestion:", query)
    print("Answer:", answer)


if __name__ == "__main__":
    main()
