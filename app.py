import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Free AI Mentor", page_icon="🤖")

# --- SECRETS MANAGEMENT ---
# In Streamlit Cloud, add GROQ_API_KEY and HF_TOKEN to Secrets
groq_key = st.secrets.get("GROQ_API_KEY")
hf_token = st.secrets.get("HF_TOKEN")

if not groq_key or not hf_token:
    st.warning("Please add GROQ_API_KEY and HF_TOKEN to Streamlit Secrets.")
    st.stop()

st.title("🤖 100% Free AI Document Mentor")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Processing for free..."):
        # 1. Load and Chunk
        loader = PyPDFLoader("temp.pdf")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = loader.load_and_split(text_splitter)
        
        # 2. Free Embeddings (Hugging Face)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # 3. Free Vector Store (FAISS is better for free/local apps than Chroma)
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        # 4. Free LLM (Groq - Llama 3)
        llm = ChatGroq(
            groq_api_key=groq_key,
            model_name="llama3-8b-8192"
        )
        
        # 5. Build Chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        
        query = st.text_input("Ask anything about the document:")
        if query:
            response = qa.invoke(query)
            st.write("### AI Response:")
            st.info(response["result"])
