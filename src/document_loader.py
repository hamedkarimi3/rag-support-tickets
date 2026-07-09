"""
document_loader.py — Support Ticket Loader
===========================================

HOW IT WORKS:
1. Reads ticket files from the data folder
2. Converts each ticket into a LangChain Document object
3. Each Document has two parts:
   - page_content: the text that gets embedded (Subject, Description, Resolution...)
   - metadata: extra info stored alongside (ticket_id, priority, tags, source...)

FLOW:
data/technical/tickets.json  ─┐
data/product/tickets.json    ─┤→ load_tickets() → Dict of Documents by type
data/customer/tickets.json   ─┘
data/*/tickets.xml           ─┘

SUPPORTED FORMATS:
- JSON: list of ticket objects with fields like Subject, Body, Answer, tag_1...tag_8
- XML:  <ticket> elements with child tags like <subject>, <body>, <answer>

TICKET ID FORMAT:
- JSON tickets:  "{support_type}_{original_id}"      e.g. "technical_T001"
- XML tickets:   "{support_type}_xml_{original_id}"  e.g. "technical_xml_T001"

ENTRY POINT:
Call create_documents() to get all tickets ready for the vector store.
"""

from typing import List, Dict, Any
from pathlib import Path
import xml.etree.ElementTree as ET  # built-in Python XML parser
from langchain_core.documents import Document  # LangChain document format
import logging
import json
from uuid import uuid4  # generates unique IDs when ticket has no ID

logger = logging.getLogger(__name__)


class SupportDocumentLoader:
    """
    Loads support tickets from JSON and XML files and converts them
    into LangChain Document objects ready for embedding and storage.
    """

    def __init__(self, data_path: str):
        # Store the data folder path and make sure it actually exists
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")

    def get_json_content(self, data: Dict[str, Any]) -> str:
        """
        Converts raw ticket dictionary into a clean text block.
        This text is what gets embedded into vectors later.
        Format is fixed so all tickets look the same to the AI.
        """
        return (
            f"Subject: {data.get('Subject', '')}\n"
            f"Description: {data.get('Body', '')}\n"
            f"Resolution: {data.get('Answer', '')}\n"
            f"Type: {data.get('Type', '')}\n"
            f"Queue: {data.get('Queue', '')}\n"
            f"Priority: {data.get('Priority', '')}"
        )

    def get_json_metadata(self, record: Dict[str, Any], support_type: str = None) -> Dict[str, Any]:
        """
        Extracts metadata from a ticket record.
        Metadata is stored alongside the vector but not embedded —
        used for filtering, display, and ticket identification.
        ticket_id format: "{support_type}_{original_id}" e.g. "technical_T001"
        """
        if not support_type:
            raise ValueError("support_type must be provided")

        # Use existing ticket ID or generate a new unique one
        original_id = record.get("Ticket ID", str(uuid4()))

        # Collect tags from tag_1 through tag_8 (skip empty ones)
        tags = [record.get(f"tag_{i}", "") for i in range(1, 9) if record.get(f"tag_{i}")]

        return {
            "ticket_id": f"{support_type}_{original_id}",  # unique ID across all types
            "original_ticket_id": str(original_id),         # original ID from file
            "support_type": support_type,                    # technical / product / customer
            "type": record.get("Type", ""),
            "queue": record.get("Queue", ""),
            "priority": record.get("Priority", ""),
            "language": record.get("Language", ""),
            "tags": tags,
            "source": "json",                               # where it came from
            "subject": record.get("Subject", ""),
            "body": record.get("Body", ""),
            "answer": record.get("Answer", "")
        }

    def load_xml_tickets(self, file_path: Path, support_type: str) -> List[Document]:
        """
        Reads an XML file and converts each <ticket> element into a Document.
        Same content and metadata format as JSON tickets for consistency.
        ticket_id format for XML: "{support_type}_xml_{original_id}"
        """
        documents = []

        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        for ticket in root.findall("ticket"):
            # Helper to safely get text from an XML tag
            def get(tag, t=ticket):
                el = t.find(tag)
                return el.text.strip() if el is not None and el.text else ""

            original_id = get("ticket_id") or str(uuid4())

            # Extract tags from <tags><tag>value</tag></tags> structure
            tags = []
            tags_el = ticket.find("tags")
            if tags_el is not None:
                tags = [t.text.strip() for t in tags_el.findall("tag") if t.text]

            # Build same content format as JSON tickets
            content = (
                f"Subject: {get('subject')}\n"
                f"Description: {get('body')}\n"
                f"Resolution: {get('answer')}\n"
                f"Type: {get('type')}\n"
                f"Queue: {get('queue')}\n"
                f"Priority: {get('priority')}"
            )

            metadata = {
                "ticket_id": f"{support_type}_xml_{original_id}",
                "original_ticket_id": original_id,
                "support_type": support_type,
                "type": get("type"),
                "queue": get("queue"),
                "priority": get("priority"),
                "language": get("language"),
                "tags": tags,
                "source": "xml"  # marks this came from XML not JSON
            }

            documents.append(Document(page_content=content, metadata=metadata))

        return documents

    def load_tickets(self) -> Dict[str, List[Document]]:
        """
        Main loading function — loops through all 3 support type folders,
        loads all JSON and XML files, and returns organized by type.
        Also validates that no two tickets share the same ID.
        """
        result = {}
        all_ids = []  # collect all IDs to check for duplicates at the end

        for support_type in ["technical", "product", "customer"]:
            docs = []
            type_path = self.data_path / support_type

            # Skip if folder doesn't exist
            if not type_path.exists():
                continue

            # Load all JSON files in this folder
            for json_file in type_path.glob("*.json"):
                with open(json_file) as f:
                    records = json.load(f)
                if isinstance(records, list):
                    for record in records:
                        docs.append(Document(
                            page_content=self.get_json_content(record),
                            metadata=self.get_json_metadata(record, support_type=support_type)
                        ))

            # Load all XML files in this folder
            for xml_file in type_path.glob("*.xml"):
                docs.extend(self.load_xml_tickets(xml_file, support_type))

            result[support_type] = docs
            all_ids.extend([d.metadata["ticket_id"] for d in docs])

        # Make sure every ticket has a unique ID across all types
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("Duplicate ticket IDs found")

        return result

    def create_documents(self) -> Dict[str, List[Document]]:
        """
        Public entry point for the rest of the app.
        Calls load_tickets() and returns the result.
        """
        return self.load_tickets()