import streamlit as st
import os
import uuid
import datetime
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain

# --- 1. GLOBAL SETTINGS ---
st.set_page_config(page_title="AI Mentor Pro", layout="wide")

# --- 2. MANDATORY DATABASE CONNECTION ---
# If this fails, the app will show the error immediately
client = MongoClient(st.secrets["MONGO_URI"])
db = client["AI_Mentor_DB"]
history_collection = db["chat_history"]

# --- 3. SESSION INITIALIZATION ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# --- 4. SIDEBAR: PERSISTENT HISTORY ---
with st.sidebar:
    st.title("📂 Past Conversations")
    
    # This will throw a visible error if the connection times out
    past_sessions = history_collection.distinct("session_id")
    
    for s_id in past_sessions:
        if st.button(f"💬 Chat {s_id[:8]}", key=s_id):
            # Fetch and Load History
            saved_data = list(history_collection.find({"session_id": s_id}).sort("timestamp", 1))
            st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in saved_data]
            st.session_state.session_id = s_id
            st.rerun()

    st.divider()
    if st.button("➕ Start New Chat", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.qa_chain = None
        st.rerun()

# --- 5. MAIN INTERFACE ---
st.title("🤖 AI Document Mentor")

# Uploader placed prominently at the top
uploaded_file = st.file_uploader("STEP 1: Upload your PDF document", type="pdf")

if uploaded_file and st.session_state.qa_chain is None:
    with st.spinner("Step 2: Analyzing Document..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader("temp.pdf")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = loader.load_and_split(text_splitter)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        llm = ChatGroq(
            groq_api_key=st.secrets["GROQ_API_KEY"], 
            model_name="llama-3.1-8b-instant"
        )
        
        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )
        st.success("Analysis Complete! You can now chat with your document.")

# --- 6. CHAT INTERFACE ---
# Display historical messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about the PDF..."):
    # Immediate User Feedback
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.qa_chain:
        with st.chat_message("assistant"):
            with st.spinner("AI is thinking..."):
                result = st.session_state.qa_chain.invoke({
                    "question": prompt, 
                    "chat_history": st.session_state.chat_history
                })
                response = result["answer"]
                st.markdown(response)
        
        # --- MANDATORY SAVE TO DATABASE ---
        timestamp = datetime.datetime.utcnow()
        history_collection.insert_many([
            {"session_id": st.session_state.session_id, "role": "user", "content": prompt, "timestamp": timestamp},
            {"session_id": st.session_state.session_id, "role": "assistant", "content": response, "timestamp": timestamp + datetime.timedelta(seconds=1)}
        ])
        
        # Update Local State
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.chat_history.append((prompt, response))
    else:
        st.warning("⚠️ Action Required: Please upload a PDF file above before asking questions.")
