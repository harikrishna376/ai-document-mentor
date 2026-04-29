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

# Custom CSS for ChatGPT-like UI
st.markdown("""
    <style>
    .main { background-color: #000000; }
    .auth-card {
        background-color: #202123;
        padding: 40px;
        border-radius: 10px;
        text-align: center;
        max-width: 400px;
        margin: auto;
        color: white;
        border: 1px solid #4d4d4f;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #3e3f4b;
        color: white;
        border: 1px solid #565869;
    }
    .stButton>button:hover {
        background-color: #2A2B32;
        border-color: #acacbe;
    }
    .guest-btn button {
        background-color: #10a37f !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATABASE & SESSION STATE ---
if "auth_state" not in st.session_state:
    st.session_state.auth_state = "landing" # landing, login, signup, authenticated

client = MongoClient(st.secrets["MONGO_URI"])
db = client["AI_Mentor_DB"]
history_collection = db["chat_history"]
users_collection = db["users"] # New collection for users

# --- 3. AUTHENTICATION PAGES ---

def landing_page():
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/04/ChatGPT_logo.svg", width=60)
    st.title("Welcome to AI Mentor")
    st.write("Log in with your account to continue")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Log in"):
            st.session_state.auth_state = "login"
            st.rerun()
    with col2:
        if st.button("Sign up"):
            st.session_state.auth_state = "signup"
            st.rerun()
    
    st.markdown('<div class="guest-btn">', unsafe_allow_html=True)
    if st.button("Continue as Guest"):
        st.session_state.auth_state = "authenticated"
        st.session_state.user_type = "guest"
        st.rerun()
    st.markdown('</div></div>', unsafe_allow_html=True)

def login_page():
    with st.container():
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.title("Log In")
        email = st.text_input("Email address")
        pwd = st.text_input("Password", type="password")
        if st.button("Continue"):
            user = users_collection.find_one({"email": email, "password": pwd})
            if user:
                st.session_state.auth_state = "authenticated"
                st.session_state.user_type = "user"
                st.session_state.user_email = email
                st.rerun()
            else:
                st.error("Invalid credentials")
        if st.button("Back"):
            st.session_state.auth_state = "landing"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

def signup_page():
    with st.container():
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.title("Create Account")
        new_email = st.text_input("Email address")
        new_pwd = st.text_input("Password", type="password")
        if st.button("Sign Up"):
            if users_collection.find_one({"email": new_email}):
                st.error("User already exists")
            else:
                users_collection.insert_one({"email": new_email, "password": new_pwd})
                st.success("Account created! Please log in.")
                st.session_state.auth_state = "login"
                st.rerun()
        if st.button("Back"):
            st.session_state.auth_state = "landing"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- 4. MAIN NAVIGATION ---

if st.session_state.auth_state == "landing":
    landing_page()
elif st.session_state.auth_state == "login":
    login_page()
elif st.session_state.auth_state == "signup":
    signup_page()
elif st.session_state.auth_state == "authenticated":
    # --- RAG APP CODE START ---
    
    with st.sidebar:
        st.title("📂 Conversations")
        if st.button("Log Out"):
            st.session_state.auth_state = "landing"
            st.rerun()
        st.divider()
        # History logic (filtering by user if not guest)
        query = {"session_id": {"$exists": True}}
        if st.session_state.get("user_type") == "user":
            query = {"user_email": st.session_state.user_email}
        
        past_sessions = history_collection.distinct("session_id", query)
        for s_id in past_sessions:
            if st.button(f"💬 Chat {s_id[:8]}", key=s_id):
                saved = list(history_collection.find({"session_id": s_id}).sort("timestamp", 1))
                st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in saved]
                st.session_state.session_id = s_id
                st.rerun()

    st.title("🤖 AI Document Mentor")
    # ... Rest of your PDF uploader and Chat logic from previous version ...
