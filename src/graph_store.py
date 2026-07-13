"""
graph_store.py — Neo4j Knowledge Graph Manager
===============================================

HOW IT WORKS:
1. Takes loaded tickets and creates a knowledge graph in Neo4j
2. Each ticket becomes a node with its properties
3. Tickets are connected by shared tags, priority, and support type
4. When searching, traverses the graph to find related tickets
   that vector search alone might miss

GRAPH STRUCTURE:
(:Ticket {id, subject, content, priority, type})
    -[:HAS_TAG]->(:Tag {name})
    -[:SAME_TYPE]->(:Ticket)
    -[:RELATED_TO]->(:Ticket)  # shared tags

FLOW:
Tickets → create nodes → create relationships → store in Neo4j
Query → vector results → expand via graph → richer context
"""

from typing import List, Dict, Any
from neo4j import GraphDatabase
import logging
import os
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())
try:
    import streamlit as st
    _secrets = st.secrets
except:
    _secrets = {}
logger = logging.getLogger(__name__)


class SupportGraphStore:
    """
    Manages the Neo4j knowledge graph for support tickets.
    Enriches RAG results by finding related tickets through graph traversal.
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(
            _secrets.get("NEO4J_URI") or os.getenv("NEO4J_URI", "bolt://localhost:7687"),
         auth=(
              _secrets.get("NEO4J_USERNAME") or os.getenv("NEO4J_USERNAME", "neo4j"),
             _secrets.get("NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD", "password123")
            )
        )
        logger.info("Connected to Neo4j")

    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()

    def clear_graph(self):
        """Clear all existing data in the graph."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Graph cleared")

    def create_graph(self, documents_by_type: Dict[str, List[Any]]) -> None:
        """
        Creates the knowledge graph from support tickets.
        Each ticket becomes a node, tags become separate nodes,
        and relationships connect related tickets.
        """
        with self.driver.session() as session:

            # Step 1 — Create ticket nodes
            for support_type, docs in documents_by_type.items():
                for doc in docs:
                    m = doc.metadata
                    session.run("""
                        MERGE (t:Ticket {ticket_id: $ticket_id})
                        SET t.subject = $subject,
                            t.content = $content,
                            t.support_type = $support_type,
                            t.priority = $priority,
                            t.queue = $queue,
                            t.source = $source
                    """, {
                        "ticket_id": m.get("ticket_id", ""),
                        "subject": m.get("subject", ""),
                        "content": doc.page_content,
                        "support_type": support_type,
                        "priority": m.get("priority", ""),
                        "queue": m.get("queue", ""),
                        "source": m.get("source", "")
                    })

                    # Step 2 — Create tag nodes and connect to ticket
                    for tag in m.get("tags", []):
                        if tag:
                            session.run("""
                                MERGE (tag:Tag {name: $tag})
                                WITH tag
                                MATCH (t:Ticket {ticket_id: $ticket_id})
                                MERGE (t)-[:HAS_TAG]->(tag)
                            """, {"tag": tag, "ticket_id": m.get("ticket_id", "")})

            # Step 3 — Connect tickets that share the same tags
            session.run("""
                MATCH (t1:Ticket)-[:HAS_TAG]->(tag:Tag)<-[:HAS_TAG]-(t2:Ticket)
                WHERE t1.ticket_id <> t2.ticket_id
                MERGE (t1)-[:RELATED_TO]->(t2)
            """)

            # Step 4 — Connect tickets of the same support type
            session.run("""
                MATCH (t1:Ticket), (t2:Ticket)
                WHERE t1.support_type = t2.support_type
                AND t1.ticket_id <> t2.ticket_id
                MERGE (t1)-[:SAME_TYPE]->(t2)
            """)

        logger.info("Knowledge graph created successfully")

    def get_related_tickets(
        self,
        ticket_ids: List[str],
        depth: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Given a list of ticket IDs from vector search,
        traverses the graph to find related tickets.
        depth=2 means: find tickets connected within 2 hops.
        This is the GraphRAG magic — finding tickets vector search missed.
        """
        if not ticket_ids:
            return []

        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:Ticket)
                WHERE t.ticket_id IN $ticket_ids
                MATCH (t)-[:RELATED_TO*1..2]-(related:Ticket)
                WHERE NOT related.ticket_id IN $ticket_ids
                RETURN DISTINCT related.ticket_id as ticket_id,
                       related.content as content,
                       related.support_type as support_type,
                       related.priority as priority
                LIMIT 5
            """, {"ticket_ids": ticket_ids})

            related = []
            for record in result:
                related.append({
                    "content": record["content"],
                    "metadata": {
                        "ticket_id": record["ticket_id"],
                        "support_type": record["support_type"],
                        "priority": record["priority"],
                        "tags": [],
                        "source": "graph"  # marks this came from graph traversal
                    },
                    "similarity": 0.0,  # no similarity score for graph results
                    "source": "graph"
                })

        return related

    def get_ticket_by_id(self, ticket_id: str) -> Dict[str, Any]:
        """Fetch a single ticket from the graph by its ID."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:Ticket {ticket_id: $ticket_id})
                OPTIONAL MATCH (t)-[:HAS_TAG]->(tag:Tag)
                RETURN t.ticket_id as ticket_id,
                       t.content as content,
                       t.support_type as support_type,
                       t.priority as priority,
                       collect(tag.name) as tags
            """, {"ticket_id": ticket_id})

            record = result.single()
            if record:
                return {
                    "content": record["content"],
                    "metadata": {
                        "ticket_id": record["ticket_id"],
                        "support_type": record["support_type"],
                        "priority": record["priority"],
                        "tags": record["tags"]
                    }
                }
        return {}

    def get_graph_stats(self) -> Dict[str, int]:
        """Returns basic stats about the graph — useful for the UI."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:Ticket) WITH count(t) as tickets
                MATCH (tag:Tag) WITH tickets, count(tag) as tags
                MATCH ()-[r:RELATED_TO]->() WITH tickets, tags, count(r) as relations
                RETURN tickets, tags, relations
            """)
            record = result.single()
            if record:
                return {
                    "tickets": record["tickets"],
                    "tags": record["tags"],
                    "relations": record["relations"]
                }
        return {"tickets": 0, "tags": 0, "relations": 0}