# main.py - Complete RAG System with File Upload and Q&A (Fixed SQLite Issue)
import streamlit as st
import os
import tempfile
import warnings
warnings.filterwarnings('ignore')

import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import List, Dict, Any, Tuple
import PyPDF2
from docx import Document
import io
from pathlib import Path
import time
import pickle
import json

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
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def load_docx(self, file_bytes: bytes) -> str:
        """Extract text from DOCX file bytes"""
        try:
            doc = Document(io.BytesIO(file_bytes))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return ""
    
    def load_txt(self, file_bytes: bytes) -> str:
        """Load text from TXT file bytes"""
        try:
            return file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return file_bytes.decode('latin-1')
            except Exception as e:
                st.error(f"Error reading TXT file: {str(e)}")
                return ""
    
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
        if not text.strip():
            return []
            
        # Split by sentences first to maintain coherence
        sentences = text.replace('\n', ' ').split('. ')
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk.split()) + len(sentence.split()) > self.chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    # Start new chunk with overlap
                    words = current_chunk.split()
                    if len(words) > self.overlap:
                        current_chunk = ' '.join(words[-self.overlap:]) + ' ' + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += '. ' + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out very short chunks
        chunks = [chunk for chunk in chunks if len(chunk.split()) > 10]
        return chunks

@st.cache_resource
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load and cache embedding model"""
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        st.error(f"Failed to load embedding model: {str(e)}")
        st.info("Trying alternative model...")
        try:
            return SentenceTransformer("paraphrase-MiniLM-L6-v2")
        except:
            st.error("Failed to load any embedding model")
            return None

class SimpleVectorStore:
    """Simple vector database using numpy and cosine similarity"""
    
    def __init__(self):
        self.embeddings = []
        self.documents = []
        self.metadatas = []
        self.ids = []
    
    def clear(self):
        """Clear all stored data"""
        self.embeddings = []
        self.documents = []
        self.metadatas = []
        self.ids = []
    
    def add_documents(self, texts: List[str], embeddings: np.ndarray, metadatas: List[Dict] = None):
        """Add documents to the vector store"""
        if metadatas is None:
            metadatas = [{"source": f"document_{i}"} for i in range(len(texts))]
        
        # Generate IDs
        new_ids = [f"doc_{len(self.ids) + i}_{int(time.time())}" for i in range(len(texts))]
        
        # Store everything
        if len(self.embeddings) == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
            
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)
        self.ids.extend(new_ids)
        
        return len(texts)
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> Dict[str, Any]:
        """Search for similar documents using cosine similarity"""
        if len(self.embeddings) == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        # Calculate cosine similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Return results in ChromaDB format for compatibility
        results = {
            "documents": [[self.documents[i] for i in top_indices]],
            "metadatas": [[self.metadatas[i] for i in top_indices]],
            "distances": [[1 - similarities[i] for i in top_indices]]  # Convert to distance
        }
        
        return results
    
    def count(self) -> int:
        """Return number of documents"""
        return len(self.documents)

# Simple LLM using Hugging Face Transformers (fallback if ctransformers fails)
class SimpleLLM:
    """Simple LLM interface for generating responses"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.loaded = False
    
    def load_model(self):
        """Load a simple generative model"""
        try:
            from transformers import pipeline
            self.model = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                return_full_text=False,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7
            )
            self.loaded = True
            return True
        except Exception as e:
            st.error(f"Failed to load transformers model: {str(e)}")
            return False
    
    def generate(self, prompt: str) -> str:
        """Generate response"""
        if not self.loaded:
            return "Model not loaded. Please try again."
        
        try:
            # Simple prompt formatting
            formatted_prompt = f"Question: {prompt}\nAnswer:"
            response = self.model(formatted_prompt)
            return response[0]['generated_text'].strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

class RAGPipeline:
    """Complete RAG pipeline for Streamlit"""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.text_chunker = TextChunker(chunk_size=800, overlap=100)
        self.embedding_model = load_embedding_model()
        self.vector_store = SimpleVectorStore()
        self.llm = None
    
    def process_uploaded_files(self, uploaded_files) -> int:
        """Process uploaded files and add to vector store"""
        if not uploaded_files:
            return 0
        
        if self.embedding_model is None:
            st.error("Embedding model not loaded. Cannot process documents.")
            return 0
        
        # Clear previous documents
        self.vector_store.clear()
        
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
                if not text.strip():
                    st.warning(f"No text extracted from {uploaded_file.name}")
                    continue
                    
                chunks = self.text_chunker.chunk_text(text)
                
                if not chunks:
                    st.warning(f"No valid chunks created from {uploaded_file.name}")
                    continue
                
                # Create metadata
                metadatas = [
                    {
                        "source": uploaded_file.name,
                        "chunk_id": j,
                        "file_type": Path(uploaded_file.name).suffix.lower(),
                        "file_size": len(text)
                    } 
                    for j in range(len(chunks))
                ]
                
                all_chunks.extend(chunks)
                all_metadatas.extend(metadatas)
                
                st.success(f"✅ Processed {uploaded_file.name}: {len(chunks)} chunks")
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        if all_chunks:
            status_text.text("Generating embeddings...")
            
            try:
                # Generate embeddings in batches
                batch_size = 32
                total_chunks = 0
                
                for i in range(0, len(all_chunks), batch_size):
                    batch_chunks = all_chunks[i:i + batch_size]
                    batch_metadatas = all_metadatas[i:i + batch_size]
                    
                    embeddings = self.embedding_model.encode(
                        batch_chunks, 
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
                    
                    added = self.vector_store.add_documents(batch_chunks, embeddings, batch_metadatas)
                    total_chunks += added
                    
                    progress_bar.progress(min(1.0, (i + batch_size) / len(all_chunks)))
                
                status_text.text(f"✅ Successfully processed {total_chunks} chunks from {len(uploaded_files)} files")
                return total_chunks
                
            except Exception as e:
                st.error(f"Error generating embeddings: {str(e)}")
                return 0
        
        return 0
    
    def retrieve_context(self, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant context for a query"""
        if self.embedding_model is None:
            return []
            
        try:
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
            results = self.vector_store.similarity_search(query_embedding, k=k)
            contexts = results['documents'][0] if results['documents'] else []
            return contexts
        except Exception as e:
            st.error(f"Error retrieving context: {str(e)}")
            return []
    
    def generate_simple_answer(self, query: str, context: List[str]) -> str:
        """Generate a simple answer based on context without external LLM"""
        if not context:
            return "No relevant information found in the documents."
        
        # Simple keyword-based answer generation
        context_text = " ".join(context[:3])  # Use top 3 contexts
        
        # Basic answer construction
        answer = f"Based on the uploaded documents:\n\n"
        
        # Add most relevant context
        if len(context) > 0:
            answer += f"Here's what I found: {context[0][:400]}..."
            
        return answer
    
    def generate_answer(self, query: str, context: List[str]) -> str:
        """Generate answer using retrieved context"""
        # Try simple answer first
        if not context:
            return "No relevant context found. Please upload some documents first."
        
        # For now, use simple context-based answering
        # You can extend this to use more sophisticated models later
        context_text = "\n\n".join(context[:3])
        
        answer = f"""Based on your documents, here's what I found:

**Relevant Information:**
{context_text[:800]}...

**Summary:** This information from your documents is most relevant to your question: "{query}"

*Note: This response is based on similarity matching with your uploaded documents.*"""
        
        return answer
    
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
    
    # Show system info
    with st.expander("ℹ️ System Information"):
        st.info("""
        **Fixed Version - No ChromaDB Dependency**
        - ✅ Uses simple vector storage (no SQLite issues)
        - ✅ Supports PDF, DOCX, and TXT files
        - ✅ Semantic search with sentence transformers
        - ✅ Privacy-first (runs locally)
        """)
    
    # Initialize RAG pipeline
    if 'rag_pipeline' not in st.session_state:
        with st.spinner("Initializing RAG system..."):
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
                        st.rerun()
        
        # Document status
        if st.session_state.document_count > 0:
            st.info(f"📊 {st.session_state.document_count} chunks ready for Q&A")
            
            # Show document stats
            vector_store = st.session_state.rag_pipeline.vector_store
            if hasattr(vector_store, 'metadatas') and vector_store.metadatas:
                sources = list(set([meta.get('source', 'Unknown') for meta in vector_store.metadatas]))
                st.write("**Uploaded files:**")
                for source in sources:
                    st.write(f"• {source}")
        else:
            st.warning("No documents processed yet")
        
        # Settings
        st.header("⚙️ Settings")
        context_chunks = st.slider("Context chunks to retrieve", 1, 10, 5)
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.rerun()
        
        # Clear all data
        if st.button("🗑️ Clear All Data"):
            if st.session_state.rag_pipeline.vector_store:
                st.session_state.rag_pipeline.vector_store.clear()
            st.session_state.document_count = 0
            st.session_state.messages = []
            st.success("All data cleared!")
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
                        st.markdown(ctx[:500] + "..." if len(ctx) > 500 else ctx)
                        if i < len(message["context"][:3]) - 1:
                            st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if st.session_state.document_count == 0:
            st.error("Please upload and process some documents first!")
        else:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching documents..."):
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
    st.markdown("🚀 **Powered by:** Streamlit | 🔍 **Search:** Sentence Transformers | 🔒 **Privacy:** Local Processing")

if __name__ == "__main__":
    main()
