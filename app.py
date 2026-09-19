import os
import re
import tempfile
import numpy as np
import streamlit as st
import gdown
from pypdf import PdfReader
import docx
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
from groq import Groq

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AI Document Assistant", page_icon="📚", layout="wide")
st.title("📚 AI Document Assistant with Hybrid RAG")

# -----------------------------------------------------------------------------
# Initialize Session State
# -----------------------------------------------------------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "metadata" not in st.session_state:
    st.session_state.metadata = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "bm25" not in st.session_state:
    st.session_state.bm25 = None

# -----------------------------------------------------------------------------
# Caching Heavy Models
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# -----------------------------------------------------------------------------
# Secure Groq API Key Retrieval (No Sidebar Exposure)
# -----------------------------------------------------------------------------
secret_api_key = st.secrets.get("GROQ_API_KEY", "") if "GROQ_API_KEY" in st.secrets else ""

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Hide API Key input box completely if configured in secrets
    if secret_api_key:
        groq_api_key = secret_api_key
        st.success("✓ Groq API Key Loaded from Secrets")
    else:
        groq_api_key = st.text_input(
            "Groq API Key", 
            type="password", 
            help="Add GROQ_API_KEY in Streamlit Secrets to remove this box."
        )
    
    selected_model = st.selectbox(
        "Groq LLM Model",
        options=["openai/gpt-oss-120b", "qwen-3.6-27b", "llama-3.3-70b-versatile"],
        index=0
    )

    st.divider()
    st.header("📥 Document Sources")
    
    source_tab1, source_tab2 = st.tabs(["Local Upload", "Google Drive"])

# -----------------------------------------------------------------------------
# File Processing Functions
# -----------------------------------------------------------------------------
def extract_text_from_file(file_path, filename):
    ext = os.path.splitext(filename)[1].lower()
    text = ""
    
    try:
        if ext == ".pdf":
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages):
                extracted = page.extract_text()
                if extracted:
                    text += f"\n--- Page {page_num + 1} ---\n" + extracted
        elif ext == ".docx":
            doc = docx.Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        elif ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    except Exception as e:
        st.error(f"Error reading {filename}: {str(e)}")
        
    return text

def chunk_text(text, source_name, chunk_size=500, overlap=100):
    cleaned_text = re.sub(r'\n{3,}', '\n\n', text).strip()
    paragraphs = cleaned_text.split("\n\n")
    
    chunks = []
    metadata = []
    current_chunk = ""
    chunk_id = 0
    
    for para in paragraphs:
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                metadata.append({"source": source_name, "chunk_id": chunk_id})
                chunk_id += 1
            
            overlap_text = current_chunk[-overlap:] if len(current_chunk) >= overlap else current_chunk
            current_chunk = overlap_text + para + "\n\n"
            
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        metadata.append({"source": source_name, "chunk_id": chunk_id})
        
    return chunks, metadata

def process_documents(files_dict):
    all_chunks = []
    all_metadata = []
    
    with st.spinner("Extracting text and building chunks..."):
        for filename, filepath in files_dict.items():
            raw_text = extract_text_from_file(filepath, filename)
            if raw_text.strip():
                doc_chunks, doc_meta = chunk_text(raw_text, source_name=filename)
                all_chunks.extend(doc_chunks)
                all_metadata.extend(doc_meta)
                
    if not all_chunks:
        st.warning("No readable text found in the provided files.")
        return

    st.session_state.chunks = all_chunks
    st.session_state.metadata = all_metadata

    # 1. FAISS Indexing
    with st.spinner("Generating embeddings & building FAISS index..."):
        embeddings = embedding_model.encode(all_chunks, convert_to_numpy=True, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)
        
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        st.session_state.faiss_index = index

    # 2. BM25 Sparse Indexing
    with st.spinner("Building BM25 keyword index..."):
        tokenized_corpus = [re.findall(r'\w+', doc.lower()) for doc in all_chunks]
        st.session_state.bm25 = BM25Okapi(tokenized_corpus)

    st.success(f"Successfully processed {len(all_chunks)} unique document chunks!")

# -----------------------------------------------------------------------------
# Sidebar Actions: Document Ingestion
# -----------------------------------------------------------------------------
with source_tab1:
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT, MD",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True
    )
    if st.button("Process Uploaded Files") and uploaded_files:
        temp_files = {}
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in uploaded_files:
                path = os.path.join(temp_dir, file.name)
                with open(path, "wb") as f:
                    f.write(file.getbuffer())
                temp_files[file.name] = path
            
            process_documents(temp_files)

with source_tab2:
    drive_folder_url = st.text_input(
        "Google Drive Folder Link", 
        help="Make sure link sharing is set to 'Anyone with the link can view'"
    )
    if st.button("Download & Process Drive Folder") and drive_folder_url:
        with tempfile.TemporaryDirectory() as temp_dir:
            with st.spinner("Downloading files from Google Drive..."):
                try:
                    # Clean fixed gdown call without outdated parameters
                    gdown.download_folder(url=drive_folder_url, output=temp_dir, quiet=True)
                    drive_files = {}
                    
                    supported_exts = (".pdf", ".docx", ".txt", ".md")
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            if file.lower().endswith(supported_exts):
                                full_path = os.path.join(root, file)
                                drive_files[file] = full_path
                    
                    if drive_files:
                        process_documents(drive_files)
                    else:
                        st.error("No supported PDF, DOCX, TXT, or MD files found in the provided Drive folder.")
                except Exception as e:
                    st.error(f"Failed to fetch Google Drive folder: {str(e)}")

with st.sidebar:
    st.divider()
    if st.session_state.chunks:
        st.metric("Total Indexed Chunks", len(st.session_state.chunks))

# -----------------------------------------------------------------------------
# Hybrid Retrieval Pipeline
# -----------------------------------------------------------------------------
def hybrid_search(query, top_k=4, alpha=0.6):
    if not st.session_state.faiss_index or not st.session_state.bm25:
        return []

    num_chunks = len(st.session_state.chunks)
    actual_k = min(top_k * 3, num_chunks)

    # Dense Search
    query_vector = embedding_model.encode([query], convert_to_numpy=True)
    query_vector = np.array(query_vector, dtype=np.float32)
    faiss.normalize_L2(query_vector)
    
    dense_distances, dense_indices = st.session_state.faiss_index.search(query_vector, actual_k)
    
    dense_scores = np.zeros(num_chunks)
    for idx, dist in zip(dense_indices[0], dense_distances[0]):
        if idx != -1:
            dense_scores[idx] = dist

    if np.max(dense_scores) > np.min(dense_scores):
        dense_scores = (dense_scores - np.min(dense_scores)) / (np.max(dense_scores) - np.min(dense_scores) + 1e-8)

    # Sparse BM25 Search
    tokenized_query = re.findall(r'\w+', query.lower())
    sparse_scores = np.array(st.session_state.bm25.get_scores(tokenized_query))
    
    if np.max(sparse_scores) > np.min(sparse_scores):
        sparse_scores = (sparse_scores - np.min(sparse_scores)) / (np.max(sparse_scores) - np.min(sparse_scores) + 1e-8)

    # Combined Score
    hybrid_scores = alpha * dense_scores + (1 - alpha) * sparse_scores
    ranked_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    results = []
    for idx in ranked_indices:
        results.append({
            "chunk": st.session_state.chunks[idx],
            "metadata": st.session_state.metadata[idx],
            "score": float(hybrid_scores[idx]),
            "dense_score": float(dense_scores[idx]),
            "sparse_score": float(sparse_scores[idx])
        })
        
    return results

# -----------------------------------------------------------------------------
# Main Chat & Search Interface
# -----------------------------------------------------------------------------
if not st.session_state.chunks:
    st.info("👈 Upload your documents or enter a Google Drive folder link in the sidebar to get started.")
else:
    query = st.text_input("🔍 Ask a question about your documents:", placeholder="e.g., What are the key points of this document?")

    if query:
        results = hybrid_search(query, top_k=4, alpha=0.6)

        st.subheader("📑 Top Relevant Chunks")
        for idx, res in enumerate(results):
            with st.expander(f"Chunk {idx + 1} | Source: {res['metadata']['source']} (Hybrid Score: {res['score']:.3f})"):
                st.write(res['chunk'])
                st.caption(f"Dense Similarity: {res['dense_score']:.3f} | Sparse Keyword: {res['sparse_score']:.3f} | Chunk ID: {res['metadata']['chunk_id']}")

        st.divider()
        st.subheader("🤖 AI Answer")

        if not groq_api_key:
            st.warning("Please provide a valid Groq API Key in Streamlit Secrets.")
        else:
            try:
                client = Groq(api_key=groq_api_key)
                
                context = "\n\n---\n\n".join([f"Source: {r['metadata']['source']}\n{r['chunk']}" for r in results])
                
                system_prompt = (
                    "You are a helpful AI document assistant. Answer the question accurately using ONLY "
                    "the provided context chunks. If the answer cannot be found in the context, state that clearly."
                )
                user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

                with st.spinner("Generating answer with Groq..."):
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        model=selected_model,
                        temperature=0.2,
                        max_tokens=1024
                    )
                    
                    st.write(chat_completion.choices[0].message.content)
            except Exception as e:
                st.error(f"Groq API Error: {str(e)}")
            
