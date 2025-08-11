# main.py - Complete RAG System with File Upload and Q&A
import streamlit as st
import os
import tempfile
import warnings
warnings.filterwarnings('ignore')

import torch
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from ctransformers import AutoModelForCausalLM
import numpy as np
from typing import List, Dict, Any
import PyPDF2
from docx import Document
import io
from pathlib import Path
import time

# Page config
st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DocumentProcessor:
    """Handle document loading and text extraction"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt', '.docx']
    
    def load_pdf(self, file_bytes: bytes) -> str:
        """Extract text from PDF file bytes"""
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def load_docx(self, file_bytes: bytes) -> str:
        """Extract text from DOCX file bytes"""
        doc = Document(io.BytesIO(file_bytes))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def load_txt(self, file_bytes: bytes) -> str:
        """Load text from TXT file bytes"""
        return file_bytes.decode('utf-8')
    
    def load_document(self, file_name: str, file_bytes: bytes) -> str:
        """Load document based on file extension"""
        extension = Path(file_name).suffix.lower()
        
        if extension == '.pdf':
            return self.load_pdf(file_bytes)
        elif extension == '.docx':
            return self.load_docx(file_bytes)
        elif extension == '.txt':
            return self.load_txt(file_bytes)
        else:
            raise ValueError(f"Unsupported file format: {extension}")

class TextChunker:
    """Split documents into manageable chunks"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Break if we've reached the end
            if i + self.chunk_size >= len(words):
                break
                
        return chunks

@st.cache_resource
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load and cache embedding model"""
    return SentenceTransformer(model_name)

class ChromaVectorStore:
    """Chroma vector database for document storage and retrieval"""
    
    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        
        # Initialize Chroma client (in-memory for Streamlit)
        self.client = chromadb.Client()
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except:
            self.collection = self.client.create_collection(name=collection_name)
    
    def clear_collection(self):
        """Clear the collection for new documents"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(name=self.collection_name)
        except:
            pass
    
    def add_documents(self, texts: List[str], embeddings: np.ndarray, metadatas: List[Dict] = None):
        """Add documents to the vector store"""
        ids = [f"doc_{i}_{int(time.time())}" for i in range(len(texts))]
        
        if metadatas is None:
            metadatas = [{"source": f"document_{i}"} for i in range(len(texts))]
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        return len(texts)
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> Dict[str, Any]:
        """Search for similar documents"""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
        return results

@st.cache_resource
def load_local_llm():
    """Load and cache local LLM"""
    try:
        # Try different model configurations
        model_configs = [
            {
                "model_path": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                "model_file": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                "model_type": "mistral"
            },
            {
                "model_path": "TheBloke/Mistral-7B-Instruct-v0.1-GGUF", 
                "model_file": "mistral-7b-instruct-v0.1.Q4_0.gguf",
                "model_type": "mistral"
            },
            {
                "model_path": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                "model_file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", 
                "model_type": "llama"
            }
        ]
        
        for config in model_configs:
            try:
                llm = AutoModelForCausalLM.from_pretrained(
                    config["model_path"],
                    model_file=config["model_file"],
                    model_type=config["model_type"],
                    gpu_layers=0,  # CPU only for Streamlit Cloud
                    context_length=2048,
                    max_new_tokens=512,
                    temperature=0.7,
                    repetition_penalty=1.1
                )
                st.success(f"Loaded model: {config['model_path']}")
                return llm
            except Exception as e:
                st.warning(f"Failed to load {config['model_path']}: {str(e)}")
                continue
        
        raise Exception("All model configurations failed")
        
    except Exception as e:
        st.error(f"Failed to load any LLM model: {str(e)}")
        return None

class RAGPipeline:
    """Complete RAG pipeline for Streamlit"""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.text_chunker = TextChunker(chunk_size=800, overlap=100)
        self.embedding_model = load_embedding_model()
        self.vector_store = ChromaVectorStore()
        self.llm = None
    
    def process_uploaded_files(self, uploaded_files) -> int:
        """Process uploaded files and add to vector store"""
        if not uploaded_files:
            return 0
        
        # Clear previous documents
        self.vector_store.clear_collection()
        
        all_chunks = []
        all_metadatas = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing: {uploaded_file.name}")
            
            # Read file bytes
            file_bytes = uploaded_file.read()
            
            # Extract text
            try:
                text = self.doc_processor.load_document(uploaded_file.name, file_bytes)
                chunks = self.text_chunker.chunk_text(text)
                
                # Create metadata
                metadatas = [
                    {
                        "source": uploaded_file.name,
                        "chunk_id": j,
                        "file_type": Path(uploaded_file.name).suffix.lower()
                    } 
                    for j in range(len(chunks))
                ]
                
                all_chunks.extend(chunks)
                all_metadatas.extend(metadatas)
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        if all_chunks:
            status_text.text("Generating embeddings...")
            
            # Generate embeddings in batches to avoid memory issues
            batch_size = 50
            total_chunks = 0
            
            for i in range(0, len(all_chunks), batch_size):
                batch_chunks = all_chunks[i:i + batch_size]
                batch_metadatas = all_metadatas[i:i + batch_size]
                
                embeddings = self.embedding_model.encode(batch_chunks, convert_to_numpy=True)
                added = self.vector_store.add_documents(batch_chunks, embeddings, batch_metadatas)
                total_chunks += added
                
                progress_bar.progress(min(1.0, (i + batch_size) / len(all_chunks)))
            
            status_text.text(f"✅ Successfully processed {total_chunks} chunks from {len(uploaded_files)} files")
            return total_chunks
        
        return 0
    
    def retrieve_context(self, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant context for a query"""
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
        results = self.vector_store.similarity_search(query_embedding, k=k)
        contexts = results['documents'][0] if results['documents'] else []
        return contexts
    
    def generate_answer(self, query: str, context: List[str]) -> str:
        """Generate answer using retrieved context and local LLM"""
        if self.llm is None:
            with st.spinner("Loading language model... This may take a few minutes."):
                self.llm = load_local_llm()
        
        if self.llm is None:
            return "❌ Language model failed to load. Please try again or check your setup."
        
        # Create prompt with context
        context_text = "\n\n".join(context[:3])  # Limit context to avoid token limits
        
        prompt = f"""[INST] You are a helpful assistant that answers questions based on the provided context. 
Use only the information from the context to answer the question clearly and concisely. 
If the answer is not in the context, say "I cannot find this information in the provided documents."

Context:
{context_text}

Question: {query}

Answer: [/INST]"""
        
        try:
            response = self.llm(prompt)
            return response.strip()
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"
    
    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Complete RAG pipeline query"""
        # Retrieve context
        context = self.retrieve_context(question, k=k)
        
        if not context:
            return {
                "question": question,
                "answer": "No relevant context found. Please upload some documents first.",
                "context": []
            }
        
        # Generate answer
        answer = self.generate_answer(question, context)
        
        return {
            "question": question,
            "answer": answer,
            "context": context
        }

def main():
    st.title("🤖 RAG Document Q&A System")
    st.markdown("Upload documents and ask questions about their content!")
    
    # Initialize RAG pipeline
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Initialize document count
    if 'document_count' not in st.session_state:
        st.session_state.document_count = 0
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or DOCX files"
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    chunk_count = st.session_state.rag_pipeline.process_uploaded_files(uploaded_files)
                    st.session_state.document_count = chunk_count
                    if chunk_count > 0:
                        st.success(f"✅ Processed {chunk_count} text chunks!")
                        # Clear chat history when new documents are uploaded
                        st.session_state.messages = []
        
        # Document status
        if st.session_state.document_count > 0:
            st.info(f"📊 {st.session_state.document_count} chunks ready for Q&A")
        else:
            st.warning("No documents processed yet")
        
        # Settings
        st.header("⚙️ Settings")
        context_chunks = st.slider("Context chunks to retrieve", 1, 10, 5)
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    st.header("💬 Ask Questions About Your Documents")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "context" in message:
                with st.expander("📚 View Source Context"):
                    for i, ctx in enumerate(message["context"][:3]):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.markdown(ctx[:300] + "..." if len(ctx) > 300 else ctx)
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if st.session_state.document_count == 0:
            st.error("Please upload and process some documents first!")
            return
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.rag_pipeline.query(prompt, k=context_chunks)
                
                st.markdown(result["answer"])
                
                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result["answer"],
                    "context": result["context"]
                })
    
    # Footer
    st.markdown("---")
    st.markdown("🚀 Powered by Streamlit | 🧠 Local LLM | 🔒 Privacy-First")

if __name__ == "__main__":
    main()
