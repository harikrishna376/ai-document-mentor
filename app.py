import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

st.set_page_config(page_title="Free AI Mentor", page_icon="🤖")

# --- SECRETS ---
groq_key = st.secrets.get("GROQ_API_KEY")
hf_token = st.secrets.get("HF_TOKEN")

if not groq_key or not hf_token:
    st.warning("Please add GROQ_API_KEY and HF_TOKEN to Streamlit Secrets.")
    st.stop()

st.title("🤖 Free AI Document Mentor")

# Initialize session state for the QA chain
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    # Only process if we haven't already
    if st.session_state.qa_chain is None:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Analyzing document..."):
            loader = PyPDFLoader("temp.pdf")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = loader.load_and_split(text_splitter)
            
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embeddings)
            
            llm = ChatGroq(groq_api_key=groq_key, model_name="llama-3.1-8b-instant")
            
            # Store the chain in session state
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )
            st.success("Ready! Ask your questions below.")

# 5. Question Interface (Outside the processing block)
if st.session_state.qa_chain:
    query = st.text_input("Ask anything about the document:")
    if query:
        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain.invoke(query)
            st.write("### AI Response:")
            st.info(response["result"])
