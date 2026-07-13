"""
vector_store.py — Vector Database Manager
==========================================

HOW IT WORKS:
1. Takes loaded documents from document_loader.py
2. Sends each ticket text to OpenAI to get embeddings (vectors = lists of numbers)
3. Stores those vectors in ChromaDB on disk (so we don't re-embed every restart)
4. When user searches, converts query to vector and finds nearest tickets

FLOW:
Documents → embed_documents() → ChromaDB (saved to disk)
User query → embed_query() → similarity search → top K results

COLLECTIONS:
ChromaDB organizes data in collections — one per support type:
- "technical"  → technical support tickets
- "product"    → product support tickets  
- "customer"   → customer support tickets

FIRST RUN:  create_vector_store() — embeds and saves everything
NEXT RUNS:  load_local() — loads from disk, no re-embedding needed
SEARCHING:  similarity_search() — finds most relevant tickets for a query
"""

from typing import List, Dict, Any
import os
import chromadb  # vector database
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings  # converts text to vectors using OpenAI
import logging
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
try:
    import streamlit as st
    openai_api = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
except:
    openai_api = os.getenv("OPENAI_API_KEY")
logger = logging.getLogger(__name__)


class SupportVectorStore:
    """
    Manages the vector database for support tickets.
    Converts tickets to embeddings and stores them in ChromaDB.
    Each support type (technical/product/customer) gets its own collection.
    """

    def __init__(self, vecstore_path: str):
        # Path where ChromaDB will save data on disk
        self.vecstore_path = vecstore_path

        # OpenAI embedding model — converts text to numbers (vectors)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=openai_api
        )

        # ChromaDB client — persists data to disk so we don't re-embed every time
        self.client = chromadb.PersistentClient(path=vecstore_path)

        # Dictionary to hold references to each collection
        self.collections = {}

    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        ChromaDB only accepts primitive types (str, int, float, bool).
        This converts lists to comma-separated strings and handles None values.
        """
        result = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                # Convert list to string: ["vpn", "network"] -> "vpn, network"
                result[key] = ", ".join(str(v) for v in value)
            elif value is None:
                result[key] = ""
            else:
                result[key] = value
        return result

    def _process_metadata_for_return(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reverses _prepare_metadata when retrieving from ChromaDB.
        Converts comma-separated tag strings back to lists.
        """
        result = dict(metadata)
        if "tags" in result and isinstance(result["tags"], str):
            # Convert back: "vpn, network" -> ["vpn", "network"]
            result["tags"] = [t.strip() for t in result["tags"].split(",") if t.strip()]
        return result

    def create_vector_store(self, documents_by_type: Dict[str, List[Document]]) -> None:
        """
        Takes all loaded documents, embeds them with OpenAI,
        and stores them in ChromaDB — one collection per support type.
        This only needs to run once — data is saved to disk.
        """
        for support_type, docs in documents_by_type.items():
            # Create or get existing collection for this support type
            collection = self.client.get_or_create_collection(name=support_type)

            # Extract text content, metadata and generate unique IDs
            texts = [doc.page_content for doc in docs]
            metadatas = [self._prepare_metadata(doc.metadata) for doc in docs]
            ids = [str(i) for i in range(len(docs))]

            # Send all texts to OpenAI to get embeddings (vectors)
            embeddings = self.embeddings.embed_documents(texts)

            # Store everything in ChromaDB
            collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            self.collections[support_type] = collection
            logger.info(f"Created collection '{support_type}' with {len(docs)} tickets")

    @classmethod
    def load_local(cls, directory: str) -> 'SupportVectorStore':
        """
        Loads an existing vector store from disk.
        Used when the app restarts — no need to re-embed everything.
        """
        instance = cls(directory)
        # Load all existing collections from ChromaDB
        for col in instance.client.list_collections():
            instance.collections[col.name] = instance.client.get_collection(col.name)
        logger.info(f"Loaded {len(instance.collections)} collections from disk")
        return instance

    def query_similar(
        self,
        query: str,
        support_type: str = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Finds the most similar tickets to the user query.
        Converts query to a vector, then finds nearest vectors in ChromaDB.
        Returns results sorted by similarity score (highest first).
        """
        # Reject empty queries silently
        if not query or not query.strip():
            logger.warning("Empty query received")
            return []

        # Convert user query to a vector using OpenAI
        query_embedding = self.embeddings.embed_query(query)

        results = []

        # Decide which collections to search
        if support_type:
            if support_type not in self.collections:
                logger.warning(f"Support type '{support_type}' not found")
                return []
            cols = {support_type: self.collections[support_type]}
        else:
            # Search all collections if no type specified
            cols = self.collections

        for stype, collection in cols.items():
            # Query ChromaDB for k most similar documents
            res = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, collection.count())
            )

            # Package results with similarity score
            for content, metadata, distance in zip(
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0]
            ):
                results.append({
                    "content": content,
                    "metadata": self._process_metadata_for_return(metadata),
                    "similarity": 1 - distance  # convert distance to similarity score
                })

        # Return sorted by highest similarity first
        return sorted(results, key=lambda x: x["similarity"], reverse=True)

    def similarity_search(
        self,
        query: str,
        support_type: str = None,
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Public wrapper for query_similar.
        Called by rag_chain.py to retrieve relevant tickets.
        """
        return self.query_similar(query, support_type=support_type, k=k)

    def get_support_types(self) -> List[str]:
        """Returns list of available support types in the vector store."""
        return list(self.collections.keys())