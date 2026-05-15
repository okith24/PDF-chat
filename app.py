import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Page config ---
st.set_page_config(page_title="Chat with PDF", page_icon="📄")
st.title("📄 Chat with your PDF")
st.write("Upload a PDF and ask questions about it!")

# --- Upload PDF ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:

    # Step 1: Extract text
    st.info("Reading your PDF...")
    reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text()

    # Step 2: Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(raw_text)
    st.success(f" PDF split into {len(chunks)} chunks")

    # Step 3: Embed chunks and store in FAISS
    st.info("Building vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    st.success(" Vector store ready!")

    # Step 4: Ask a question
    question = st.text_input("Ask a question about your PDF:")

    if question:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_chunks = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in relevant_chunks])

        st.subheader("Answer:")
        st.write(context)