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

# --- DATABASE SETUP ---
client = MongoClient(st.secrets["MONGO_URI"])
db = client["AI_Mentor_DB"]
history_collection = db["chat_history"]

# --- INITIALIZE SESSION ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# --- SIDEBAR: CHAT HISTORY ---
with st.sidebar:
    st.title("📂 Past Conversations")
    
    try:
        # Get unique session IDs from MongoDB
        past_sessions = history_collection.distinct("session_id")
        for s_id in past_sessions:
            if st.button(f"💬 Chat {s_id[:8]}", key=s_id):
                # Fetch full history for this session
                saved_data = list(history_collection.find({"session_id": s_id}).sort("timestamp", 1))
                st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in saved_data]
                # Update chat history for the LLM
                st.session_state.chat_history = []
                for i in range(0, len(saved_data), 2):
                    if i+1 < len(saved_data):
                        st.session_state.chat_history.append((saved_data[i]["content"], saved_data[i+1]["content"]))
                st.session_state.session_id = s_id
                st.rerun()
    except:
        st.error("Database connection error.")

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.qa_chain = None # Reset PDF context
        st.rerun()

# --- MAIN CHAT LOGIC ---
st.title("🤖 AI Document Mentor")
# (Include your PDF Uploader and Vectorstore Logic here)

if prompt := st.chat_input("Ask a question..."):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get AI Answer (Invoke Chain)
    if st.session_state.qa_chain:
        with st.chat_message("assistant"):
            result = st.session_state.qa_chain.invoke({
                "question": prompt, 
                "chat_history": st.session_state.chat_history
            })
            response = result["answer"]
            st.markdown(response)

        # 3. SAVE TO MONGODB (The Persistence Step)
        timestamp = datetime.datetime.utcnow()
        history_collection.insert_many([
            {"session_id": st.session_state.session_id, "role": "user", "content": prompt, "timestamp": timestamp},
            {"session_id": st.session_state.session_id, "role": "assistant", "content": response, "timestamp": timestamp + datetime.timedelta(seconds=1)}
        ])
        
        # 4. Update memory
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.chat_history.append((prompt, response))
    else:
        st.error("Please upload a PDF first!")
