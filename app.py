import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")

st.markdown("""
<style>
    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main background */
    .main {
        background-color: #f7f8fc;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
        padding: 20px 10px;
    }
    
    /* All sidebar text white */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] markdown {
        color: #ffffff !important;
    }
    
    /* Input boxes - black text visible */
    [data-testid="stSidebar"] input {
        background-color: #ffffff !important;
        color: #111111 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 8px 12px !important;
    }
    
    /* File uploader area */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background-color: #2a2a4a !important;
        border-radius: 10px !important;
        padding: 10px !important;
        border: 1px dashed #555 !important;
    }
    
    /* File uploader text */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] span {
        color: #aaaaaa !important;
        font-size: 12px !important;
    }
    
    /* Browse files button */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background-color: #4a4a7a !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
    }
    
    /* Delete file button (X) */
    [data-testid="stSidebar"] [data-testid="stFileUploaderDeleteBtn"] button {
        background-color: #cc3333 !important;
        color: white !important;
        border-radius: 50% !important;
        width: 24px !important;
        height: 24px !important;
        padding: 0 !important;
        border: none !important;
    }
    
    /* Clear chat button */
    [data-testid="stSidebar"] .stButton button {
        background-color: #cc3333 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        width: 100% !important;
        padding: 10px !important;
        font-weight: bold !important;
    }
    
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #ff4444 !important;
    }
    
    /* History items */
    .history-item {
        background-color: #2a2a4a;
        padding: 10px 14px;
        border-radius: 8px;
        margin: 5px 0;
        font-size: 13px;
        color: #cccccc !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        border-left: 3px solid #4a90e2;
    }
    
    /* PDF badge */
    .pdf-badge {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-size: 14px;
        color: #2e7d32;
        font-weight: 500;
    }
    
    /* User chat bubble */
    .user-bubble {
        background: linear-gradient(135deg, #4a90e2, #357abd);
        color: white;
        padding: 14px 20px;
        border-radius: 20px 20px 4px 20px;
        margin: 10px 0 10px auto;
        max-width: 60%;
        font-size: 15px;
        box-shadow: 0 2px 8px rgba(74,144,226,0.3);
    }
    
    /* Bot chat bubble */
    .bot-bubble {
        background-color: #ffffff;
        color: #222222;
        padding: 14px 20px;
        border-radius: 20px 20px 20px 4px;
        margin: 10px auto 10px 0;
        max-width: 60%;
        font-size: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 3px solid #4a90e2;
    }
    
    /* Welcome screen */
    .welcome-box {
        text-align: center;
        padding: 100px 20px;
        color: #aaaaaa;
    }
    
    /* Divider */
    .sidebar-divider {
        border: none;
        border-top: 1px solid #333355;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

# ── SIDEBAR ──────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 PDF Chatbot")
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    api_key = st.text_input("🔑 Google API Key:", 
                             type="password", 
                             placeholder="AIzaSy...")
    if api_key:
        st.session_state.api_key = api_key

    uploaded_file = st.file_uploader("📂 Upload PDF", type="pdf")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Chat history list
    if st.session_state.chat_history:
        st.markdown("### 🕒 History")
        for chat in st.session_state.chat_history:
            short = chat["question"][:38] + "..." \
                if len(chat["question"]) > 38 else chat["question"]
            st.markdown(
                f'<div class="history-item">💬 {short}</div>',
                unsafe_allow_html=True
            )
        st.markdown("")

    # Clear button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.pdf_loaded = False
        st.session_state.vectorstore = None
        st.rerun()

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown("**How it works:**")
    st.markdown("1️⃣ Enter your API Key")
    st.markdown("2️⃣ Upload a PDF file")
    st.markdown("3️⃣ Ask any question!")

# ── MAIN AREA ─────────────────────────────────────────
st.markdown("# 💬 Chat with your PDF")

# Load PDF only once
if uploaded_file and st.session_state.api_key and not st.session_state.pdf_loaded:
    os.environ["GOOGLE_API_KEY"] = st.session_state.api_key

    with st.spinner("📖 Reading and processing your PDF — please wait..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(pages)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
        vectorstore = Chroma.from_documents(chunks, embeddings)

        st.session_state.vectorstore = vectorstore
        st.session_state.pdf_loaded = True
        st.session_state.pdf_name = uploaded_file.name
        st.session_state.total_pages = len(pages)
        st.session_state.total_chunks = len(chunks)

# Show chat area if PDF is loaded
if st.session_state.pdf_loaded:

    # PDF info badge
    st.markdown(f"""
    <div class="pdf-badge">
        ✅ &nbsp;<b>{st.session_state.pdf_name}</b> &nbsp;|&nbsp; 
        📄 {st.session_state.total_pages} pages &nbsp;|&nbsp; 
        🧩 {st.session_state.total_chunks} chunks
    </div>
    """, unsafe_allow_html=True)

    # Display chat history
    for chat in st.session_state.chat_history:
        st.markdown(
            f'<div class="user-bubble">🧑&nbsp; {chat["question"]}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="bot-bubble">🤖&nbsp; {chat["answer"]}</div>',
            unsafe_allow_html=True
        )

    # Chat input
    question = st.chat_input("Ask anything about your PDF...")

    if question:
        os.environ["GOOGLE_API_KEY"] = st.session_state.api_key

        with st.spinner("🔍 Finding answer..."):
            retriever = st.session_state.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )
            docs = retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"""You are a helpful assistant. 
Answer the question based on the context below.
Be clear, accurate and concise.

Context:
{context}

Question: {question}

Answer:"""

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.3
            )
            response = llm.invoke(prompt)
            answer = response.content

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })
        st.rerun()

else:
    # Welcome screen
    st.markdown("""
    <div class="welcome-box">
        <h1 style="font-size:60px;">📄</h1>
        <h2 style="color:#555;">Welcome to PDF Chatbot</h2>
        <p style="font-size:17px; color:#888;">
            Enter your Google API Key and upload a PDF<br>
            from the sidebar to get started!
        </p>
    </div>
    """, unsafe_allow_html=True)