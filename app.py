import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Document Mentor", page_icon="📄")

# --- API KEY MANAGEMENT ---
# Priority 1: Check Streamlit Secrets (for Deployment)
# Priority 2: Sidebar Input (for Local Testing)
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    api_key_status = True
else:
    st.sidebar.warning("API Key not found in Secrets. Please enter it below.")
    user_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if user_api_key:
        os.environ["OPENAI_API_KEY"] = user_api_key
        api_key_status = True
    else:
        api_key_status = False

# --- MAIN UI ---
st.title("📄 AI Document Mentor")
st.markdown("""
    Upload a PDF and ask questions. This AI uses **RAG** (Retrieval-Augmented Generation) 
    to answer based *only* on the content of your file.
""")

if not api_key_status:
    st.info("Waiting for OpenAI API Key to start...")
else:
    # 1. File Upload Section
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        # Save uploaded file to a temporary location
        with open("temp_upload.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Analyzing document..."):
            try:
                # 2. Document Loading & Chunking
                loader = PyPDFLoader("temp_upload.pdf")
                data = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=150
                )
                chunks = text_splitter.split_documents(data)
                
                # 3. Vector Embeddings & Storage
                # We use an in-memory Chroma database
                vectorstore = Chroma.from_documents(
                    documents=chunks, 
                    embedding=OpenAIEmbeddings()
                )
                
                # 4. Setup Retrieval Chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever()
                )
                
                st.success("Document processed! Ready for questions.")
                
                # 5. Question & Answer Interface
                user_question = st.text_input("Ask a question about the PDF:")
                
                if user_question:
                    with st.spinner("Searching for answer..."):
                        response = qa_chain.invoke(user_question)
                        st.write("### Answer:")
                        st.info(response["result"])
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")
