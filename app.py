import streamlit as st
import os
import uuid
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain

# --- DATABASE SETUP ---
client = MongoClient(st.secrets["MONGO_URI"])
db = client["AI_Mentor_DB"]
history_collection = db["chat_history"]

# --- INITIALIZE SESSION ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4()) # Unique ID for this chat
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# --- SIDEBAR: HISTORY ---
with st.sidebar:
    st.title("📂 Past Conversations")
    # Fetch all unique session IDs from Mongo
    past_sessions = history_collection.distinct("session_id")
    for s_id in past_sessions:
        if st.button(f"Chat: {s_id[:8]}...", key=s_id):
            # Load messages from Mongo for this session
            saved_messages = list(history_collection.find({"session_id": s_id}))
            st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in saved_messages]
            st.session_state.session_id = s_id
            st.rerun()
    
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

st.title("🤖 AI Document Mentor")

# --- PDF & RAG LOGIC (Keep as is, but add this after AI response) ---
# ... (Previous RAG processing code here) ...

if prompt := st.chat_input("Type your message..."):
    # ... (Invoke LLM) ...
    
    # --- THE PERSISTENCE STEP ---
    # Save User Message to Mongo
    history_collection.insert_one({
        "session_id": st.session_state.session_id,
        "role": "user",
        "content": prompt
    })
    # Save AI Response to Mongo
    history_collection.insert_one({
        "session_id": st.session_state.session_id,
        "role": "assistant",
        "content": response
    })
