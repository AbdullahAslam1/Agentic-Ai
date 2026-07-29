# RAG (Retrieval-Augmented Generation) System - Analysis Report

## Executive Summary
The RAG system is a complete implementation of a **Retrieval-Augmented Generation** pipeline that combines document retrieval with LLM-based answer generation. It processes PDF documents, converts them to embeddings, stores them in a vector database, and retrieves relevant documents to answer user queries.

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Pipeline Flow                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PDF Files  →  Document    →  Embeddings   →  Vector      │
│   (Input)       Ingestor       Generation      Store       │
│                                                    ↓        │
│                                              ChromaDB       │
│                                                    ↓        │
│  User Query  →  Query        →  Retrieval   →  Retrieved  │
│                  Embedding       (Top-K)       Documents   │
│                                       ↓                    │
│                                    Context  →  LLM      →  │
│                                    + Query      Generation  │
│                                                    ↓        │
│                                            Final Answer    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. **document.py** - Data Setup & Examples
**Purpose:** Initializes sample documents and data directories

**Key Functions:**
- Creates sample text files (`python_intro.txt`, `machine_learning.txt`)
- Demonstrates document creation with metadata using LangChain's `Document` class
- Sets up directory structure for PDFs and text files
- Shows both `TextLoader` and `PyMuPDFLoader` usage

**Sample Document Structure:**
```python
Document(
    page_content="...",
    metadata={
        "source": "example.txt",
        "pages": 1,
        "author": "Krish Naik",
        "date_created": "2025-01-01"
    }
)
```

---

### 2. **embeddings.py** - Embedding Generation
**Purpose:** Converts text documents into numerical vector representations

**Class:** `EmbeddingManager`

**Key Features:**
| Feature | Details |
|---------|---------|
| **Model** | `all-MiniLM-L6-v2` (from SentenceTransformers) |
| **Model Size** | Lightweight, fast (~22M params) |
| **Vector Dimension** | 384 dimensions per embedding |
| **Framework** | SentenceTransformers library |

**Main Methods:**
- `__init__(model_name)` - Loads the embedding model
- `load_model()` - Initializes SentenceTransformer model
- `generate_embeddings(texts: List[str]) -> np.ndarray` - Creates embeddings for multiple texts

**Usage:**
```python
embedding_manager = EmbeddingManager()
embeddings = embedding_manager.generate_embeddings(["text1", "text2"])
# Returns: np.ndarray of shape (n_texts, 384)
```

---

### 3. **vector_store.py** - Vector Database Management
**Purpose:** Manages persistent storage and retrieval of document embeddings

**Class:** `VectorStore`

**Database Configuration:**
| Setting | Value |
|---------|-------|
| **Database** | ChromaDB |
| **Persistence** | `../data/vector_store` |
| **Collection** | `pdf_documents` |
| **Distance Metric** | Cosine Similarity |

**Key Features:**
- **Persistent Storage:** Uses ChromaDB's `PersistentClient` for data persistence
- **Metadata Sanitization:** Ensures metadata is ChromaDB-compatible (str, int, float, bool only)
- **Unique ID Generation:** Creates UUIDs for document tracking

**Main Methods:**
| Method | Purpose |
|--------|---------|
| `_initialize_store()` | Creates ChromaDB client and collection |
| `_sanitize_metadata()` | Converts metadata to compatible types |
| `add_documents()` | Stores document chunks and embeddings in vector DB |

**Document Storage Structure:**
```
{
  "id": "doc_a1b2c3d4_0",
  "embedding": [0.123, -0.456, ...],
  "metadata": {
    "source_file": "document.pdf",
    "file_type": "pdf",
    "doc_index": 0,
    "content_length": 1500
  },
  "content": "Document text chunk..."
}
```

---

### 4. **document_ingestor.py** - PDF Processing & Chunking
**Purpose:** Loads PDFs and splits them into manageable chunks

**Class:** `DocumentIngestor`

**Configuration:**
| Parameter | Value |
|-----------|-------|
| **PDF Directory** | `../data/pdf_files` |
| **Chunk Size** | 1000 characters |
| **Chunk Overlap** | 200 characters |

**Chunking Strategy:**
- **Splitter:** `RecursiveCharacterTextSplitter`
- **Separators (Priority Order):** 
  1. Double newline (`\n\n`)
  2. Single newline (`\n`)
  3. Space (` `)
  4. Character level (``)

**Main Methods:**
| Method | Purpose |
|--------|---------|
| `load_pdfs()` | Recursively loads all PDFs from directory, adds metadata |
| `split_documents()` | Splits documents into overlapping chunks |
| `run()` | Convenience method combining load + split |

**Processing Flow:**
1. Find all `.pdf` files in directory
2. Load each PDF using `PyMuPDFLoader`
3. Add source metadata (filename, file type)
4. Split documents into chunks (1000 chars with 200 char overlap)

---

### 5. **retriever.py** - Query-Based Retrieval
**Purpose:** Finds relevant documents for a user query

**Class:** `RAGRetriever`

**Retrieval Configuration:**
| Parameter | Default |
|-----------|---------|
| **Top K Results** | 5 |
| **Score Threshold** | 0.0 (no filtering) |

**Main Methods:**
- `retrieve(query, top_k, score_threshold)` - Retrieves most similar documents

**Retrieval Process:**
1. **Query Embedding:** Convert user query to embedding vector
2. **Similarity Search:** Query ChromaDB for closest embeddings (cosine distance)
3. **Score Conversion:** Convert distances to similarity scores (1 - distance)
4. **Filtering:** Filter results by similarity threshold
5. **Ranking:** Return results with rank, score, and metadata

**Retrieved Document Format:**
```python
{
    "id": "doc_unique_id",
    "content": "Document chunk text...",
    "metadata": {...},
    "similarity_score": 0.8234,  # 0-1, higher is better
    "distance": 0.1766,           # raw cosine distance
    "rank": 1                     # position in results
}
```

---

### 6. **generator.py** - LLM Response Generation
**Purpose:** Generates answers using an LLM based on retrieved context

**Class:** `GenericLLM`

**LLM Configuration:**
| Parameter | Value |
|-----------|-------|
| **Default Model** | `openai/gpt-oss-20b` |
| **Default Provider** | Groq (via Groq API) |
| **Temperature** | 0.1 (low randomness, factual) |
| **Max Tokens** | 1024 |

**API Integration:**
- Uses `GROQ_API_KEY` from `.env` file
- Supports multiple providers via LangChain's `init_chat_model()`

**Main Methods:**
- `__init__(model, model_provider)` - Initializes LLM with specified model
- `generate_response(query, context)` - Generates RAG answer

**Prompt Template:**
```
You are a helpful AI assistant. Use the following context to answer 
the question accurately and concisely.

Context: {context}
Question: {question}

Answer: Provide a clear and informative answer based on the context above. 
If the context doesn't contain enough information to answer the question, say so.
```

**Chain Architecture:**
```
PromptTemplate → LLM → StrOutputParser → Answer
```

---

### 7. **main.py** - RAG Pipeline Orchestration
**Purpose:** Ties all components together into a complete RAG system

**Main Functions:**

#### `build_pipeline()` - Initialization
1. Creates `EmbeddingManager` for embedding generation
2. Initializes `VectorStore` (ChromaDB)
3. **Checks if vector store is empty:**
   - If empty: Loads PDFs via `DocumentIngestor`, generates embeddings, stores in ChromaDB
   - If populated: Uses existing embeddings
4. Creates `RAGRetriever` for query handling
5. Returns retriever instance

#### `build_context(retrieved_docs)` - Context Formatting
- Joins retrieved document chunks into a single text block
- Includes similarity scores for each chunk
- Format: `[Document Chunk (Score: 0.8234)]:\n{content}`

#### `answer_query(query, retriever, llm)` - Complete RAG Loop
1. **Retrieval:** Retrieves top K documents matching the query
2. **Context Building:** Formats retrieved docs into context string
3. **Generation:** Passes query + context to LLM
4. **Output:** Returns LLM-generated answer

#### `main()` - Complete Execution
1. Builds the RAG pipeline (vector store + retriever)
2. Initializes the LLM
3. Executes query: `"What is attention is all you need"`
4. Returns and prints the final answer

---

## Data Flow Diagram

```
Step 1: INGESTION
├─ PDF Files
│  └─ DocumentIngestor
│     ├─ Load PDFs (PyMuPDFLoader)
│     └─ Split into chunks (1000 chars, 200 overlap)

Step 2: EMBEDDING
├─ Document Chunks
│  └─ EmbeddingManager
│     └─ Generate vectors (384-dim, all-MiniLM-L6-v2)

Step 3: STORAGE
├─ Embeddings + Metadata
│  └─ VectorStore
│     └─ ChromaDB (Persistent at ../data/vector_store)

Step 4: RETRIEVAL (on Query)
├─ User Query
│  └─ RAGRetriever
│     ├─ Embed query (384-dim)
│     ├─ Cosine similarity search
│     └─ Return Top-5 documents with scores

Step 5: GENERATION
├─ Retrieved Context + Query
│  └─ GenericLLM (Groq)
│     ├─ Format prompt template
│     ├─ Send to LLM
│     └─ Parse response

Step 6: OUTPUT
└─ Final Answer
```

---

## Configuration & Settings

### Embedding Model
```python
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# - 22M parameters (lightweight)
# - 384-dimensional vectors
# - Good for semantic similarity
```

### Vector Store
```python
COLLECTION_NAME = "pdf_documents"
PERSIST_DIRECTORY = "../data/vector_store"
DISTANCE_METRIC = "cosine"
```

### Ingestion
```python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PDF_DIRECTORY = "../data/pdf_files"
```

### Retrieval
```python
DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.0
```

### LLM
```python
MODEL = "openai/gpt-oss-20b"
MODEL_PROVIDER = "groq"
TEMPERATURE = 0.1
MAX_TOKENS = 1024
```

---

## Key Features & Capabilities

✅ **Full RAG Pipeline**
- End-to-end document ingestion to answer generation

✅ **Persistent Storage**
- ChromaDB-based vector persistence across sessions

✅ **Semantic Search**
- Cosine similarity-based document retrieval

✅ **Metadata Tracking**
- Maintains document source, type, index, and content length

✅ **Flexible LLM Support**
- Can switch between Groq, OpenAI, Anthropic, etc.

✅ **Chunking with Overlap**
- Prevents context loss at chunk boundaries

✅ **Score-based Filtering**
- Configurable similarity threshold for retrieved documents

✅ **Error Handling**
- Try-catch blocks for robust document loading and processing

---

## Dependencies

### Core Libraries
| Library | Purpose |
|---------|---------|
| `langchain` | LLM orchestration and chains |
| `langchain-community` | Document loaders (PDF, text) |
| `chromadb` | Vector database |
| `sentence-transformers` | Embedding model |
| `numpy` | Numerical operations |
| `python-dotenv` | Environment variable loading |

### External APIs
| Service | Purpose |
|---------|---------|
| **Groq API** | LLM inference (primary provider) |
| **OpenAI, Anthropic** | Alternative LLM providers (supported) |

---

## Strengths

1. **Modular Design** - Each component is independent and reusable
2. **Production-Ready** - Error handling, metadata management, persistence
3. **Scalable** - Can handle large PDF collections with ChromaDB
4. **Flexible LLM Support** - Easy to switch between different model providers
5. **Semantic Understanding** - Uses sentence-transformers for contextual embeddings
6. **Configurable** - All parameters are easily adjustable

---

## Potential Improvements

1. **Batch Processing** - Add support for batch query operations
2. **Query Optimization** - Implement query expansion techniques
3. **Caching** - Cache embeddings to avoid recomputation
4. **Advanced Chunking** - Implement semantic-based chunking
5. **Reranking** - Add cross-encoder for result reranking
6. **Monitoring** - Add logging and performance metrics
7. **Async Operations** - Support asynchronous embedding/retrieval
8. **Document Updates** - Handle document versioning and updates

---

## Usage Example

```python
# Initialize the system
retriever = build_pipeline()
llm = GenericLLM()

# Ask a question
query = "What is attention is all you need"
answer = answer_query(query, retriever, llm)

print(answer)
```

---

## File Structure

```
Rag/
├── document.py           # Document examples and data setup
├── embeddings.py         # EmbeddingManager class
├── vector_store.py       # VectorStore (ChromaDB) class
├── document_ingestor.py  # DocumentIngestor class
├── retriever.py          # RAGRetriever class
├── generator.py          # GenericLLM class
└── main.py              # Pipeline orchestration
```

---

## Conclusion

The RAG system is a **well-structured, production-ready implementation** that demonstrates best practices for building retrieval-augmented generation systems. It effectively combines:

- **Document Processing** (ingestion and chunking)
- **Semantic Understanding** (embeddings)
- **Vector Search** (ChromaDB)
- **LLM Integration** (Groq/OpenAI/Anthropic)

This architecture enables accurate, context-aware answers by combining retrieved documents with LLM reasoning capabilities.
