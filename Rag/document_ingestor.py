from pathlib import Path
from typing import Any, List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Ingestion settings ---
PDF_DIRECTORY = "../data/pdf_files"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


class DocumentIngestor:
    """Loads PDF files from a directory and splits them into chunks."""

    def __init__(
        self,
        pdf_directory: str = PDF_DIRECTORY,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.pdf_directory = pdf_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def load_pdfs(self) -> List[Any]:
        """Process all PDF files in `self.pdf_directory`."""
        all_documents = []
        pdf_dir = Path(self.pdf_directory)
        pdf_files = list(pdf_dir.glob("**/*.pdf"))

        print(f"Found {len(pdf_files)} PDF files to process")

        for pdf_file in pdf_files:
            print(f"\nProcessing: {pdf_file.name}")
            try:
                loader = PyMuPDFLoader(str(pdf_file))
                documents = loader.load()

                for doc in documents:
                    doc.metadata["source_file"] = pdf_file.name
                    doc.metadata["file_type"] = "pdf"

                all_documents.extend(documents)
                print(f"  ✓ Loaded {len(documents)} pages")

            except Exception as e:
                print(f"  ✗ Error: {e}")

        print(f"\nTotal documents loaded: {len(all_documents)}")
        return all_documents

    def split_documents(self, documents: List[Any]) -> List[Any]:
        """Split loaded documents into smaller overlapping chunks."""
        split_docs = self.text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
        return split_docs

    def run(self) -> List[Any]:
        """Convenience method: load + split in one call."""
        documents = self.load_pdfs()
        return self.split_documents(documents)