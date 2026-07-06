import streamlit as st
from rag_pipeline import load_pdf, split_documents, create_vectorstore, get_qa_chain
import tempfile
import os

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("📄 RAG Chatbot")
st.write("Upload a PDF and ask questions from it.")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Processing PDF..."):
        documents = load_pdf(tmp_path)
        chunks = split_documents(documents)
        vectorstore = create_vectorstore(chunks)
        qa_chain = get_qa_chain(vectorstore)

    st.success("PDF processed. Ask your questions below.")

    question = st.text_input("Ask a question from your PDF:")

    if question:
        with st.spinner("Thinking..."):
            answer = qa_chain.invoke(question)
        st.write("### Answer")
        st.write(answer)

    os.unlink(tmp_path)