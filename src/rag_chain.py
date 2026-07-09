"""
rag_chain.py — RAG Chain (The Brain of the System)
====================================================

HOW IT WORKS:
1. Takes user query and validates it
2. Searches vector store for most similar tickets
3. Formats retrieved tickets into a context block
4. Sends context + query to GPT-4o
5. Returns GPT-4o's answer to the user

FLOW:
User Query → validate → similarity_search() → _prepare_context()
           → GPT-4o (query + context) → answer

WHY RAG:
Instead of asking GPT-4o from memory, we give it real relevant tickets
as context — this makes answers more accurate and grounded in real data.

ENTRY POINTS:
- get_relevant_documents() → just retrieves similar tickets, no LLM
- query()                  → full RAG: retrieves tickets + generates answer
"""

from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate  # structures the prompt for GPT-4o
from langchain_openai import ChatOpenAI               # connects to OpenAI GPT-4o
import logging
from .vector_store import SupportVectorStore
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

openai_api = os.getenv("OPENAI_API_KEY")
logger = logging.getLogger(__name__)


class SupportRAGChain:
    """
    Combines vector similarity search with GPT-4o to answer support queries.
    Retrieves relevant tickets from ChromaDB and uses them as context for the LLM.
    """

    def __init__(self, vector_store: SupportVectorStore):
        # Store reference to vector store for similarity search
        self.vector_store = vector_store

        # GPT-4o — used to generate the final answer
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,      # 0 = consistent, factual answers (no creativity)
            api_key=openai_api
        )

        # Prompt template — tells GPT-4o how to behave and structures the input
        # {context} = retrieved tickets, {query} = user question
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful support assistant. 
Use the following support tickets to answer the user's question.
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
        Retrieves most similar tickets from vector store.
        Validates query before searching.
        Used when you only need tickets, not a GPT-4o answer.
        """
        # Reject empty queries
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Reject very short queries — not enough context to search
        if len(query.strip()) < 10:
            raise ValueError("Query too short. Please provide more details.")

        return self.vector_store.similarity_search(
            query,
            k=k,
            support_type=support_type
        )

    def _prepare_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved tickets into a clean context block for GPT-4o.
        Each ticket is numbered and shows its type, tags and content.
        """
        if not documents:
            return "No relevant support tickets found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(
                f"Ticket {i}:\n"
                f"Support Type: {doc['metadata'].get('support_type', 'Unknown')}\n"
                f"Tags: {', '.join(doc['metadata'].get('tags', []))}\n"
                f"Content: {doc['content']}"
            )

        # Join all tickets with blank line between them
        return "\n\n".join(context_parts)

    async def query(
        self,
        query: str,
        support_type: str = None
    ) -> str:
        """
        Full RAG pipeline — retrieves tickets and generates an answer.
        This is the main function called by the Streamlit app.

        async because LLM calls are slow — async lets the app stay responsive.
        """
        # Validate query
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if len(query.strip()) < 10:
            raise ValueError("Query too short. Please provide more details.")

        # Step 1 — find similar tickets
        documents = self.get_relevant_documents(query, support_type=support_type)

        # Step 2 — format tickets as context
        context = self._prepare_context(documents)

        # Step 3 — send to GPT-4o and get answer
        # prompt | llm is LangChain syntax for chaining components
        chain = self.prompt | self.llm
        response = await chain.ainvoke({
            "context": context,
            "query": query
        })

        return response.content