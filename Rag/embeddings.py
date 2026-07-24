from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

# --- Embedding settings ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingManager:
    """Handles document embedding generation using sentence_transformers."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model: SentenceTransformer | None = None
        self.load_model()

    def load_model(self) -> None:
        """Load the SentenceTransformer model."""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"Error loading the model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not self.model:
            raise ValueError("Model is not loaded.")

        if not texts:
            print("Warning: Received an empty list of texts.")
            embedding_dim = self.model.get_embedding_dimension()
            return np.empty((0, embedding_dim))

        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")

        return embeddings