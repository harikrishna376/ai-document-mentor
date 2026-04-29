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

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="AI Mentor Pro", layout="wide")

# Custom CSS for the "ChatGPT" look
st.markdown("""
    <style>
    .auth-card {
        background-color: #202123;
        padding: 40px;
        border-radius: 12px;
        text-align: center;
        max-width: 450px;
        margin: 60px auto;
        color: white;
        border: 1px solid #4d4d4f;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        height: 3em;
        font-weight: 600;
    }
    /* Centering the card */
    [data-testid="stVerticalBlock"] > div:has(div.auth-card) {
        display: flex;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATABASE & SESSION INITIALIZATION ---
if "auth_state" not in st.session_state:
    st.session_state.auth_state = "landing"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# MongoDB setup
client = MongoClient(st.secrets["MONGO_URI"])
db = client["AI_Mentor_DB"]
history_collection = db["chat_history"]
users_collection = db["users"]

# --- 3. AUTHENTICATION SCREENS ---

# Landing Page
if st.session_state.auth_state == "landing":
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    st.title("AI Mentor")
    st.write("Analyze documents with precision.")
    if st.button("Log in"):
        st.session_state.auth_state = "login"
        st.rerun()
    if st.button("Sign up"):
        st.session_state.auth_state = "signup"
        st.rerun()
    st.divider()
    if st.button("Continue as Guest"):
        st.session_state.auth_state = "authenticated"
        st.session_state.user_id = "guest"
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Login Page
elif st.session_state.auth_state == "login":
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    st.title("Log In")
    email = st.text_input("Email", key="login_email")
    pwd = st.text_input("Password", type="password", key="login_pwd")
    if st.button("Continue"):
        user = users_collection.find_one({"email": email, "password": pwd})
        if user:
            st.session_state.auth_state = "authenticated"
            st.session_state.user_id = email
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
        else:
            st.error("Invalid credentials. Please Sign Up first.")
    if st.button("Back"):
        st.session_state.auth_state = "landing"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Signup Page
elif st.session_state.auth_state == "signup":
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    st.title("Create Account")
    new_email = st.text_input("New Email", key="signup_email")
    new_pwd = st.text_input("New Password", type="password", key="signup_pwd")
    if st.button("Create Account"):
        if not new_email or not new_pwd:
            st.warning("Please fill in all fields.")
        elif users_collection.find_one({"email": new_email}):
            st.error("Account already exists with this email.")
        else:
            users_collection.insert_one({"email": new_email, "password": new_pwd})
            st.success("✅ Success! Now go to Log In.")
            if st.button("Go to Login Page"):
                st.session_state.auth_state = "login"
                st.rerun()
    if st.button("Back"):
        st.session_state.auth_state = "landing"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- 4. THE MAIN APP (AUTHENTICATED) ---
elif st.session_state.auth_state == "authenticated":
    
    # Ensure session_id exists
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # SIDEBAR: HISTORY & LOGOUT
    with st.sidebar:
        st.title("📂 Conversations")
        if st.button("Logout"):
            st.session_state.auth_state = "landing"
            st.session_state.qa_chain = None
            st.rerun()
        st.divider()
        
        # Load user-specific history
        past_sessions = history_collection.distinct("session_id", {"user_id": st.session_state.user_id})
        for s_id in past_sessions:
            if st.button(f"💬 Chat {s_id[:8]}", key=s_id):
                saved = list(history_collection.find({"session_id": s_id}).sort("timestamp", 1))
                st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in saved]
                st.session_state.session_id = s_id
                st.rerun()
        
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

    # MAIN CONTENT: PDF ANALYSIS
    st.title("🤖 AI Document Mentor")
    
    uploaded_file = st.file_uploader("Step 1: Upload your PDF", type="pdf")

    if uploaded_file and st.session_state.qa_chain is None:
        with st.spinner("Analyzing document structure..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150))
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embeddings)
            
            llm = ChatGroq(groq_api_key=st.secrets["GROQ_API_KEY"], model_name="llama-3.1-8b-instant")
            st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm, 
                retriever=vectorstore.as_retriever()
            )
            st.success("Document ready for questions!")

    # CHAT DISPLAY
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # CHAT INPUT
    if prompt := st.chat_input("Ask a question about the file..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.qa_chain:
            with st.chat_message("assistant"):
                result = st.session_state.qa_chain.invoke({
                    "question": prompt, 
                    "chat_history": st.session_state.chat_history
                })
                response = result["answer"]
                st.markdown(response)
            
            # DATABASE STORAGE
            ts = datetime.datetime.utcnow()
            history_collection.insert_many([
                {
                    "user_id": st.session_state.user_id, 
                    "session_id": st.session_state.session_id, 
                    "role": "user", 
                    "content": prompt, 
                    "timestamp": ts
                },
                {
                    "user_id": st.session_state.user_id, 
                    "session_id": st.session_state.session_id, 
                    "role": "assistant", 
                    "content": response, 
                    "timestamp": ts + datetime.timedelta(seconds=1)
                }
            ])
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.chat_history.append((prompt, response))
        else:
            st.warning("Please upload a PDF first to start the AI mentor.")
