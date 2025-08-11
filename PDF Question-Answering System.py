# PDF Question-Answering System with Fine-tuning
# Requirements: streamlit, langchain, PyPDF2, sentence-transformers, faiss-cpu, openai

import streamlit as st
import PyPDF2
import io
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
import tempfile

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

class PDFQASystem:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = None
        self.qa_chain = None
        
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return None
    
    def create_chunks(self, text):
        """Split text into chunks for better processing"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        return [Document(page_content=chunk) for chunk in chunks]
    
    def create_vector_store(self, documents):
        """Create FAISS vector store from documents"""
        try:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            return True
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return False
    
    def setup_qa_chain(self, openai_api_key=None):
        """Setup the question-answering chain"""
        try:
            if openai_api_key:
                llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
            else:
                # Use Hugging Face pipeline as fallback
                from transformers import pipeline
                qa_pipeline = pipeline(
                    "question-answering",
                    model="distilbert-base-cased-distilled-squad",
                    tokenizer="distilbert-base-cased-distilled-squad"
                )
                return qa_pipeline
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            return True
        except Exception as e:
            st.error(f"Error setting up QA chain: {str(e)}")
            return False
    
    def answer_question(self, question, use_openai=True, openai_api_key=None):
        """Answer question based on the PDF content"""
        if not self.vector_store:
            return "Please upload and process a PDF first."
        
        try:
            if use_openai and self.qa_chain:
                result = self.qa_chain({"query": question})
                return result["result"]
            else:
                # Use similarity search with simple answer generation
                relevant_docs = self.vector_store.similarity_search(question, k=3)
                context = "\n".join([doc.page_content for doc in relevant_docs])
                
                # Simple context-based answer (you can enhance this)
                return f"Based on the document content:\n\n{context[:1000]}..."
                
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def save_vector_store(self, filename="vector_store.pkl"):
        """Save the vector store for later use"""
        if self.vector_store:
            with open(filename, "wb") as f:
                pickle.dump(self.vector_store, f)
            return True
        return False
    
    def load_vector_store(self, filename="vector_store.pkl"):
        """Load a saved vector store"""
        try:
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    self.vector_store = pickle.load(f)
                return True
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
        return False

def main():
    st.set_page_config(
        page_title="PDF Question-Answering System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 PDF Question-Answering System")
    st.markdown("Upload a PDF and ask questions about its content!")
    
    # Initialize the QA system
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = PDFQASystem()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OpenAI API Key input
        openai_api_key = st.text_input(
            "OpenAI API Key (Optional)", 
            type="password",
            help="Enter your OpenAI API key for better answers. Leave empty to use free alternatives."
        )
        
        # Model selection
        use_openai = st.checkbox(
            "Use OpenAI GPT", 
            value=bool(openai_api_key),
            disabled=not bool(openai_api_key)
        )
        
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. Upload a PDF file
        2. Wait for processing to complete
        3. Ask questions about the content
        4. Get precise answers based on the document
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 PDF Upload & Processing")
        
        # PDF upload
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type="pdf",
            help="Upload the PDF document you want to query"
        )
        
        if uploaded_file is not None:
            # Display PDF info
            st.info(f"📁 File: {uploaded_file.name}")
            st.info(f"📊 Size: {len(uploaded_file.read())/1024:.1f} KB")
            uploaded_file.seek(0)  # Reset file pointer
            
            # Process PDF button
            if st.button("🔄 Process PDF", type="primary"):
                with st.spinner("Processing PDF..."):
                    # Extract text
                    text = st.session_state.qa_system.extract_text_from_pdf(uploaded_file)
                    
                    if text:
                        # Create chunks
                        documents = st.session_state.qa_system.create_chunks(text)
                        st.success(f"Created {len(documents)} text chunks")
                        
                        # Create vector store
                        if st.session_state.qa_system.create_vector_store(documents):
                            st.success("✅ Vector store created successfully!")
                            
                            # Setup QA chain
                            st.session_state.qa_system.setup_qa_chain(openai_api_key)
                            
                            # Save vector store
                            if st.session_state.qa_system.save_vector_store():
                                st.success("💾 Vector store saved!")
                        else:
                            st.error("❌ Failed to create vector store")
    
    with col2:
        st.header("❓ Ask Questions")
        
        # Question input
        question = st.text_area(
            "Enter your question about the PDF:",
            height=100,
            placeholder="e.g., What is the main topic discussed in this document?"
        )
        
        # Answer button
        if st.button("🔍 Get Answer", type="primary"):
            if question.strip():
                if st.session_state.qa_system.vector_store is not None:
                    with st.spinner("Generating answer..."):
                        answer = st.session_state.qa_system.answer_question(
                            question, 
                            use_openai=use_openai,
                            openai_api_key=openai_api_key
                        )
                        
                        st.markdown("### 💡 Answer:")
                        st.markdown(answer)
                else:
                    st.warning("⚠️ Please upload and process a PDF first!")
            else:
                st.warning("⚠️ Please enter a question!")
        
        # Example questions
        if st.session_state.qa_system.vector_store is not None:
            st.markdown("### 💭 Example Questions:")
            example_questions = [
                "What is the main topic of this document?",
                "Can you summarize the key points?",
                "What are the conclusions mentioned?",
                "Are there any specific dates or numbers mentioned?"
            ]
            
            for i, eq in enumerate(example_questions):
                if st.button(f"📝 {eq}", key=f"example_{i}"):
                    with st.spinner("Generating answer..."):
                        answer = st.session_state.qa_system.answer_question(
                            eq,
                            use_openai=use_openai,
                            openai_api_key=openai_api_key
                        )
                        st.markdown("### 💡 Answer:")
                        st.markdown(answer)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tip:** For better results, use specific questions and consider providing an OpenAI API key."
    )

if __name__ == "__main__":
    main()