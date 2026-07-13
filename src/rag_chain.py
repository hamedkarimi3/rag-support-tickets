"""
rag_chain.py — GraphRAG Chain (The Brain of the System)
========================================================

HOW IT WORKS:
1. Takes user query and validates it
2. Searches vector store for most similar tickets (semantic search)
3. Takes those ticket IDs and traverses Neo4j graph for related tickets
4. Combines both results — vector results + graph results
5. Formats everything into a rich context block
6. Sends context + query to GPT-4o
7. Returns answer to the user

FLOW:
User Query → validate
    → ChromaDB similarity search (semantic)
    → Neo4j graph traversal (relational)
    → combine results
    → GPT-4o (query + rich context)
    → answer

WHY GRAPHRAG OVER BASIC RAG:
Basic RAG: finds tickets similar to query
GraphRAG:  finds similar tickets PLUS tickets related through shared
           tags, priorities, and support types — much richer context
"""

from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import logging
from .vector_store import SupportVectorStore
from .graph_store import SupportGraphStore
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

openai_api = os.getenv("OPENAI_API_KEY")
logger = logging.getLogger(__name__)


class SupportRAGChain:
    """
    Combines ChromaDB vector search + Neo4j graph traversal with GPT-4o
    to answer support queries with maximum context and accuracy.
    """

    def __init__(self, vector_store: SupportVectorStore, graph_store: SupportGraphStore = None):
        # Vector store for semantic similarity search
        self.vector_store = vector_store

        # Graph store for relationship traversal (optional)
        self.graph_store = graph_store

        # GPT-4o for generating the final answer
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=openai_api
        )

        # Prompt template — tells GPT-4o how to behave
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful support assistant.
Use the following support tickets to answer the user's question.
Tickets marked as [GRAPH RESULT] were found through knowledge graph traversal
and may provide additional related context.
If the context contains relevant tickets, use them to provide a helpful response.
If no relevant tickets are found, provide general guidance.

Context:
{context}"""),
            ("human", "{query}")
        ])

    def get_relevant_documents(
        self,
        query: str,
        support_type: str = None,
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        GraphRAG retrieval — combines vector search + graph traversal.
        Step 1: Find similar tickets via ChromaDB
        Step 2: Expand results via Neo4j graph relationships
        Step 3: Combine and return unique results
        """
        # Validate query
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if len(query.strip()) < 10:
            raise ValueError("Query too short. Please provide more details.")

        # Step 1 — Vector similarity search
        vector_results = self.vector_store.similarity_search(
            query,
            k=k,
            support_type=support_type
        )

        # Step 2 — Graph traversal if graph store available
        graph_results = []
        if self.graph_store:
            # Get IDs of vector results to expand from
            vector_ids = [
                doc["metadata"]["ticket_id"]
                for doc in vector_results
                if "ticket_id" in doc["metadata"]
            ]

            # Find related tickets through graph relationships
            graph_results = self.graph_store.get_related_tickets(vector_ids)

            # Mark graph results so GPT-4o knows their source
            for doc in graph_results:
                doc["metadata"]["from_graph"] = True

        # Step 3 — Combine results, vector results first
        all_results = vector_results + graph_results

        # Remove duplicates by ticket_id
        seen_ids = set()
        unique_results = []
        for doc in all_results:
            tid = doc["metadata"].get("ticket_id")
            if tid not in seen_ids:
                seen_ids.add(tid)
                unique_results.append(doc)

        return unique_results

    def _prepare_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved tickets into context for GPT-4o.
        Marks graph results differently so GPT-4o understands their source.
        """
        if not documents:
            return "No relevant support tickets found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            # Mark if this came from graph traversal
            source_label = ""
            if doc["metadata"].get("from_graph"):
                source_label = " [GRAPH RESULT]"

            context_parts.append(
                f"Ticket {i}{source_label}:\n"
                f"Support Type: {doc['metadata'].get('support_type', 'Unknown')}\n"
                f"Tags: {', '.join(doc['metadata'].get('tags', []))}\n"
                f"Content: {doc['content']}"
            )

        return "\n\n".join(context_parts)

    async def query(
        self,
        query: str,
        support_type: str = None
    ) -> str:
        """
        Full GraphRAG pipeline:
        1. Validate query
        2. Get relevant docs (vector + graph)
        3. Prepare context
        4. Send to GPT-4o
        5. Return answer
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if len(query.strip()) < 10:
            raise ValueError("Query too short. Please provide more details.")

        # Get combined vector + graph results
        documents = self.get_relevant_documents(query, support_type=support_type)

        # Format context
        context = self._prepare_context(documents)

        # Send to GPT-4o
        chain = self.prompt | self.llm
        response = await chain.ainvoke({
            "context": context,
            "query": query
        })

        return response.content