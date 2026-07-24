import os
import uuid
from typing import Any, List

import chromadb
import numpy as np

# --- Vector store settings ---
COLLECTION_NAME = "pdf_documents"
PERSIST_DIRECTORY = "../data/vector_store"
DISTANCE_METRIC = "cosine"


class VectorStore:
    """Manages document embeddings in a ChromaDB vector store."""

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_directory: str = PERSIST_DIRECTORY,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self) -> None:
        """Initialize ChromaDB client and collection with Cosine distance metric."""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": DISTANCE_METRIC},
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    @staticmethod
    def _sanitize_metadata(metadata: dict) -> dict:
        """Filter metadata to ensure compatibility with ChromaDB types."""
        clean_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean_metadata[k] = v
            elif isinstance(v, list) and all(isinstance(i, (str, int, float, bool)) for i in v):
                clean_metadata[k] = v
            else:
                clean_metadata[k] = str(v) if v is not None else ""
        return clean_metadata

    def add_documents(self, documents: List[Any], embeddings: np.ndarray) -> None:
        """Add documents and their embeddings to the vector store."""
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Document count ({len(documents)}) must match embeddings count ({len(embeddings)})"
            )

        print(f"Adding {len(documents)} documents to vector store...")

        ids, metadatas, documents_text, embeddings_list = [], [], [], []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            raw_metadata = dict(getattr(doc, "metadata", {}))
            raw_metadata["doc_index"] = i
            raw_metadata["content_length"] = len(doc.page_content)

            metadatas.append(self._sanitize_metadata(raw_metadata))
            documents_text.append(doc.page_content)
            embeddings_list.append(
                embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            )

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text,
            )
            print(f"Successfully added {len(documents)} documents.")
            print(f"Total documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise