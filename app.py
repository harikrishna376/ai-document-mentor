import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Mentor Pro", layout="wide")

# --- SIDEBAR: CHAT HISTORY LIST ---
with st.sidebar:
    st.title("📂 Past Chats")
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
    st.divider()
    st.info("Currently, chats are saved for this session only. Database integration is next!")

# --- INITIALIZING SESSION STATES ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # For UI display
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # For LLM memory
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# --- TOP SECTION: PDF UPLOAD ---
st.title("🤖 AI Document Mentor")
uploaded_file = st.file_uploader("Upload your document to start the conversation", type="pdf")

if uploaded_file and st.session_state.qa_chain is None:
    with st.spinner("Building intelligence..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader("temp.pdf")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = loader.load_and_split(text_splitter)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        # Using Secrets for free API access
        llm = ChatGroq(
            groq_api_key=st.secrets["GROQ_API_KEY"], 
            model_name="llama-3.1-8b-instant"
        )
        
        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_generated_question=True
        )
        st.toast("PDF Analyzed Successfully!", icon="✅")

# --- CHAT WINDOW ---
# Display existing messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Type your message here..."):
    if st.session_state.qa_chain is None:
        st.error("Please upload a PDF first!")
    else:
        # 1. Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain.invoke({
                    "question": prompt, 
                    "chat_history": st.session_state.chat_history
                })
                response = result["answer"]
                st.markdown(response)
        
        # 3. Save to memory and session state
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.chat_history.append((prompt, response))
