"""
app.py — Streamlit Web Interface
=================================

HOW IT WORKS:
1. On startup, loads or creates the vector store
2. Shows a search box for the user to type their question
3. When user clicks Search:
   - Finds similar tickets from ChromaDB
   - Sends them to GPT-4o with the query
   - Displays the AI answer + relevant tickets

FLOW:
App starts → load/create vector store → initialize RAG chain
User types query → search → display AI answer + relevant tickets

STREAMLIT:
Streamlit reruns the whole script on every interaction.
We use st.session_state to keep the RAG chain in memory
so we don't reload it on every search.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import streamlit as st

from src.document_loader import SupportDocumentLoader
from src.rag_chain import SupportRAGChain
from src.vector_store import SupportVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("support_rag")

# Constants
VECTOR_STORE_DIR = "vector_store"
DATA_PATH = "data"

# Streamlit UI placeholders
status_placeholder = st.empty()
progress_bar = st.progress(0)


def log_error(e: Exception) -> str:
    """Log error and return formatted message for display."""
    logger.error(e, exc_info=True)
    return f"❌ Error: {str(e)}"


def get_documents():
    """Load all support tickets from the data folder."""
    try:
        status_placeholder.info("📂 Loading support documents...")
        loader = SupportDocumentLoader(DATA_PATH)
        documents = loader.create_documents()
        status_placeholder.success("✅ Support documents loaded successfully!")
        return documents
    except Exception as e:
        status_placeholder.error(log_error(e))
        return None


def create_new_vector_store() -> Optional[SupportVectorStore]:
    """
    Creates a brand new vector store from scratch.
    Embeds all tickets and saves to disk.
    Runs on first launch or when data changes.
    """
    try:
        vector_store = SupportVectorStore(vecstore_path=VECTOR_STORE_DIR)

        status_placeholder.info("⚙️ Creating new vector store...")
        progress_bar.progress(40)

        # Load documents
        documents = get_documents()
        if not documents:
            return None
        progress_bar.progress(60)

        # Embed and store
        status_placeholder.info("🔄 Generating embeddings...")
        vector_store.create_vector_store(documents)
        progress_bar.progress(80)

        status_placeholder.info("💾 Saving vector store...")
        progress_bar.progress(100)

        status_placeholder.success("✅ Vector store created and saved successfully!")
        return vector_store

    except Exception as e:
        status_placeholder.error(log_error(e))
        return None


def load_existing_vector_store() -> Optional[SupportVectorStore]:
    """
    Loads vector store from disk.
    Fast — no re-embedding needed.
    Used on every restart after first launch.
    """
    try:
        status_placeholder.info("📂 Loading existing vector store...")
        progress_bar.progress(30)
        vector_store = SupportVectorStore.load_local(VECTOR_STORE_DIR)
        progress_bar.progress(100)
        status_placeholder.success("✅ Vector store loaded successfully!")
        return vector_store
    except Exception as e:
        status_placeholder.error(log_error(e))
        return None


def initialize_rag_system() -> Optional[SupportRAGChain]:
    """
    Initializes the full RAG system on app startup.
    Tries to load existing vector store first.
    If not found, creates a new one from scratch.
    """
    try:
        # Try loading existing vector store from disk
        vector_store = load_existing_vector_store()
        if not vector_store:
            # First run — create everything from scratch
            vector_store = create_new_vector_store()
            if not vector_store:
                return None

        # Initialize RAG chain with the vector store
        status_placeholder.info("🤖 Initializing RAG chain...")
        rag_chain = SupportRAGChain(vector_store)

        status_placeholder.empty()
        return rag_chain

    except Exception as e:
        error_msg = log_error(e)
        status_placeholder.error(error_msg)
        return None


def display_system_status():
    """Shows error message if RAG system failed to initialize."""
    if not st.session_state.rag_chain:
        st.error("⚠️ System initialization failed")
        st.info("""
Please ensure:
1. The data directory contains valid support ticket files
2. OpenAI API key is properly configured in .env file
3. All required packages are installed

Check the logs for detailed error information.
        """)
        return False
    return True


def render_search_results(query: str, rag_chain: SupportRAGChain):
    """
    Runs the full RAG pipeline and displays results.
    Shows AI answer first, then the relevant tickets used as context.
    """
    try:
        # Get AI response
        with st.spinner("🔍 Searching for relevant tickets..."):
            relevant_docs = rag_chain.get_relevant_documents(query)

        with st.spinner("🤖 Generating AI response..."):
            response = asyncio.run(rag_chain.query(query))

        # Display AI answer
        st.subheader("🤖 AI Response")
        st.write(response)

        # Display relevant tickets
        st.subheader("📋 Relevant Support Tickets")
        for i, doc in enumerate(relevant_docs, 1):
            with st.expander(
                f"{i}. Ticket {doc['metadata']['ticket_id']} "
                f"— {doc['metadata'].get('support_type', 'Unknown').title()}"
            ):
                st.write(f"**Priority:** {doc['metadata'].get('priority', 'N/A')}")
                st.write(f"**Tags:** {', '.join(doc['metadata'].get('tags', []))}")
                st.write(f"**Content:**\n{doc['content']}")
                st.write(f"**Similarity Score:** {doc['similarity']:.2f}")

    except ValueError as e:
        # Handle validation errors (empty query, too short)
        st.warning(f"⚠️ {str(e)}")
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")


def main():
    """Main Streamlit app function."""

    # Clear progress indicators
    progress_bar.empty()

    # Page setup
    st.title("🎫 Support Ticket Search & Assistant")
    st.write("""
    Welcome to the Support Ticket Assistant! 
    Ask questions about common issues or search for similar support tickets 
    to help resolve your problem.
    """)

    # Initialize RAG system once and store in session state
    # session_state persists across Streamlit reruns
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = initialize_rag_system()

    # Stop if initialization failed
    if not display_system_status():
        return

    # Optional filter by support type
    support_types = ["All"] + st.session_state.rag_chain.vector_store.get_support_types()
    selected_type = st.selectbox("Filter by Support Type:", support_types)
    support_type_filter = None if selected_type == "All" else selected_type

    # Search input
    query = st.text_input(
        "Enter your question or describe your issue:",
        placeholder="e.g., 'How do I fix the login error on Chrome browser?'"
    )

    # Search button
    if st.button("🔍 Search") and query:
        render_search_results(query, st.session_state.rag_chain)


if __name__ == "__main__":
    main()