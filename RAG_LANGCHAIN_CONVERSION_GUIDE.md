# RAG Implementation Conversion to Proper LangChain Architecture

## Overview

The current RAG implementation is **functional but non-idiomatic**. It manually handles components that LangChain abstracts away. This guide shows how to refactor it into a proper LangChain RAG system.

---

## Current vs LangChain Architecture

### Current Implementation (Manual Approach)
```
Document → Manual Splitting → Manual Embedding → Manual Vector Store
  ↓
Query → Manual Embedding → Manual Similarity Search → Manual Context Building
  ↓
LLM Chain (Basic)
```

### Proper LangChain RAG (Built-in Abstractions)
```
Document → LangChain Loaders → Text Splitters → Embeddings
  ↓
Retrievers (Vector DB abstraction)
  ↓
RAG Chains (LangChain's pre-built retrieval chains)
  ↓
LLM Chains with LCEL (LangChain Expression Language)
```

---

## Component-by-Component Conversion Guide

### 1. DOCUMENT LOADING

#### Current Implementation
```python
# document_ingestor.py (manual approach)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyMuPDFLoader(pdf_file)
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split_documents(documents)
```

#### Why Convert?
- ❌ Current approach is low-level and repetitive
- ❌ No abstraction for different document types
- ❌ No built-in error handling or logging
- ❌ Hard to compose multiple loaders

#### LangChain Approach
```python
# Proper LangChain way
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Single abstraction for loading entire directories
loader = DirectoryLoader(
    "../data/pdf_files",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=True,
    loader_kwargs={"extract_images": True}
)

documents = loader.load()

# Using LangChain's text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

split_docs = text_splitter.split_documents(documents)
```

#### LangChain Components Used
| Component | Purpose |
|-----------|---------|
| `DirectoryLoader` | Batch load all documents from directory |
| `PyMuPDFLoader` | Extract text and images from PDFs |
| `RecursiveCharacterTextSplitter` | Intelligent text chunking |
| `Document` | Unified document representation |

#### Improvements
✅ Handles multiple document types with single interface
✅ Batch processing built-in
✅ Image extraction support
✅ Consistent metadata handling

---

### 2. EMBEDDINGS & VECTOR STORE

#### Current Implementation
```python
# embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingManager:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def generate_embeddings(self, texts):
        return self.model.encode(texts, show_progress_bar=True)

# vector_store.py (manual ChromaDB handling)
import chromadb
class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="...")
        self.collection = self.client.get_or_create_collection(...)
    
    def add_documents(self, documents, embeddings):
        # Manual ID generation, metadata sanitization, etc.
        self.collection.add(ids=..., embeddings=..., documents=...)
```

#### Why Convert?
- ❌ Low-level embedding API handling
- ❌ Manual vector store configuration
- ❌ No abstraction layer for swapping vector DBs
- ❌ Hard to use advanced features (metadata filtering, etc.)
- ❌ No built-in retrieval chain integration

#### LangChain Approach

**Option A: LangChain Embeddings Wrapper**
```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# LangChain's embedding abstraction
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

# LangChain's vector store abstraction
vector_store = Chroma(
    collection_name="pdf_documents",
    embedding_function=embeddings,
    persist_directory="../data/vector_store"
)

# Add documents directly (handles embedding internally)
vector_store.add_documents(split_docs)
```

**Option B: Direct Integration with Retriever**
```python
from langchain_core.vectorstores import VectorStore
from langchain_chroma import Chroma

vector_store = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory="../data/vector_store"
)
```

#### LangChain Components Used
| Component | Purpose |
|-----------|---------|
| `HuggingFaceEmbeddings` | Wrapper around sentence-transformers |
| `Chroma` | LangChain's ChromaDB integration |
| `Embedding` (base class) | Standard embedding interface |
| `VectorStore` (base class) | Standard retriever interface |

#### Advantages
✅ Abstraction layer for swapping embeddings (OpenAI, Cohere, etc.)
✅ Abstraction layer for swapping vector stores (Pinecone, Weaviate, etc.)
✅ Automatic embedding generation
✅ Built-in metadata filtering
✅ Seamless retriever integration

---

### 3. RETRIEVAL (Query & Search)

#### Current Implementation
```python
# retriever.py (manual retrieval)
class RAGRetriever:
    def retrieve(self, query, top_k=5):
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        # Manual result formatting
        return [{"id": ..., "content": ..., "score": ...}]
```

#### Why Convert?
- ❌ Manual query embedding generation
- ❌ Low-level vector DB queries
- ❌ No standard interface for different retrieval methods
- ❌ Can't use advanced retrieval strategies (hybrid search, reranking, etc.)
- ❌ Not compatible with LangChain chains

#### LangChain Approach

**Option A: Vector Store as Retriever**
```python
from langchain_chroma import Chroma

vector_store = Chroma(...)

# Use vector store as retriever directly
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Usage
retrieved_docs = retriever.invoke("What is attention?")
```

**Option B: With Advanced Options**
```python
# Similarity search with score threshold
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.7}
)

# MMR (Maximal Marginal Relevance) - diverse results
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

# Metadata filtering
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"source": {"$eq": "document.pdf"}}
    }
)
```

**Option C: MultiVector Retriever** (Advanced)
```python
from langchain_core.documents import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.storage import InMemoryStore

# Store original documents separately
docstore = InMemoryStore()

# Retrieve by summary or parent doc
retriever = MultiVectorRetriever(
    vectorstore=vector_store,
    docstore=docstore,
    id_key="doc_id"
)
```

#### LangChain Components Used
| Component | Purpose |
|-----------|---------|
| `VectorStore.as_retriever()` | Convert vector store to retriever |
| `Retriever` (base class) | Standard retrieval interface |
| `MultiVectorRetriever` | Retrieve by summaries/metadata |
| `ParentDocumentRetriever` | Retrieve parent doc given child chunk |
| `ContextualCompressionRetriever` | Filter results with reranker |

#### Advantages
✅ Unified interface for all retriever types
✅ Built-in support for MMR, score filtering, metadata filtering
✅ Chainable with other LangChain components
✅ Easy to swap retrieval strategies
✅ Support for advanced retrieval patterns

---

### 4. PROMPT & LLM CHAIN

#### Current Implementation
```python
# generator.py (basic prompt + LLM)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model="openai/gpt-oss-20b",
    model_provider="groq",
    temperature=0.1
)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Context: {context}\nQuestion: {question}\nAnswer: ..."
)

chain = prompt | llm | StrOutputParser()
response = chain.invoke({"context": context, "question": query})
```

#### Why Convert?
- ⚠️ Basic implementation, but not RAG-specific
- ⚠️ No built-in history/conversation support
- ⚠️ No standard RAG prompt format
- ⚠️ Manual context building and formatting

#### LangChain Approach

**Option A: Proper RAG Chain (Recommended)**
```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Define RAG prompt (LangChain standard)
system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know.

Context: {context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.1)

# Create document processing chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Combine with retriever
rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)

# Usage
response = rag_chain.invoke({"input": "What is attention?"})
print(response["answer"])
```

**Option B: With Conversation History**
```python
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

# Contextualize question with history
contextualize_q_system_prompt = """
Given a chat history and the latest user question, 
formulate a standalone question that can be understood 
without the chat history. Don't change the wording.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# QA chain with history
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on context..."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# Usage with history
response = rag_chain.invoke({
    "input": "What is attention?",
    "chat_history": [...]
})
```

**Option C: Using LCEL (LangChain Expression Language)**
```python
from langchain_core.runnables import RunnablePassthrough

# Format context
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build chain explicitly with LCEL
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("What is attention?")
```

#### LangChain Components Used
| Component | Purpose |
|-----------|---------|
| `create_retrieval_chain()` | Pre-built RAG chain |
| `create_stuff_documents_chain()` | Combine docs into single prompt |
| `create_history_aware_retriever()` | Retriever with conversation history |
| `ChatPromptTemplate` | Multi-message prompts |
| `MessagesPlaceholder` | Dynamic message insertion |
| `LCEL (Runnables)` | Compose chains declaratively |

#### Advantages
✅ Standard RAG prompt engineering
✅ Conversation history support built-in
✅ Multiple document combination strategies (stuff, map_reduce, refine)
✅ Better error handling and logging
✅ Composable with other chains

---

### 5. MAIN PIPELINE ORCHESTRATION

#### Current Implementation
```python
# main.py (manual orchestration)
def build_pipeline():
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()
    
    if vector_store.collection.count() == 0:
        ingestor = DocumentIngestor()
        chunks = ingestor.run()
        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)
        vector_store.add_documents(chunks, embeddings)
    
    retriever = RAGRetriever(vector_store, embedding_manager)
    return retriever

def build_context(retrieved_docs):
    # Manual formatting
    return "\n\n".join([f"[{doc['score']}]: {doc['content']}" ...])

def answer_query(query, retriever, llm):
    docs = retriever.retrieve(query)
    context = build_context(docs)
    return llm.generate_response(query=query, context=context)
```

#### Why Convert?
- ❌ Lots of boilerplate code
- ❌ Manual state management
- ❌ Hard to test individual components
- ❌ Difficult to add new features
- ❌ No built-in error handling

#### LangChain Approach

**Complete Refactored Pipeline**
```python
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI  # or ChatGroq, ChatAnthropic, etc.

class RAGPipeline:
    def __init__(self, pdf_directory: str, vector_store_path: str):
        self.pdf_directory = pdf_directory
        self.vector_store_path = vector_store_path
        self.retriever = None
        self.rag_chain = None
    
    def build_vector_store(self):
        """Load documents and create vector store"""
        # Load documents
        loader = DirectoryLoader(
            self.pdf_directory,
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader,
            show_progress=True
        )
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create and persist vector store
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=self.vector_store_path,
            collection_name="pdf_documents"
        )
        
        return vector_store
    
    def build_retriever(self, vector_store):
        """Create retriever from vector store"""
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
    
    def build_rag_chain(self, retriever):
        """Create RAG chain"""
        # Define system prompt
        system_prompt = """You are a helpful assistant for question-answering tasks.
        
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.

Context: {context}"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        # Initialize LLM (LangChain supports multiple providers)
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            api_key="your-api-key"
        )
        
        # Create document chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Create retrieval chain
        rag_chain = create_retrieval_chain(retriever, document_chain)
        
        return rag_chain
    
    def initialize(self):
        """Initialize entire pipeline"""
        print("Building vector store...")
        vector_store = self.build_vector_store()
        
        print("Creating retriever...")
        self.retriever = self.build_retriever(vector_store)
        
        print("Building RAG chain...")
        self.rag_chain = self.build_rag_chain(self.retriever)
        
        print("✅ RAG pipeline initialized!")
    
    def query(self, question: str) -> str:
        """Answer a question using the RAG chain"""
        if not self.rag_chain:
            raise ValueError("Pipeline not initialized. Call initialize() first.")
        
        response = self.rag_chain.invoke({"input": question})
        return response["answer"]

# Usage
if __name__ == "__main__":
    rag = RAGPipeline(
        pdf_directory="../data/pdf_files",
        vector_store_path="../data/vector_store"
    )
    
    rag.initialize()
    
    # Query
    answer = rag.query("What is attention is all you need?")
    print(f"\n{'='*50}")
    print(f"Answer: {answer}")
    print(f"{'='*50}")
```

#### Advantages
✅ Clean, modular class structure
✅ Easy to extend and test
✅ Clear separation of concerns
✅ Built-in error handling
✅ Follows LangChain best practices

---

## Missing Components & Features

### 1. **Advanced Retrieval Strategies** ❌

Currently Missing:
- Only basic similarity search
- No reranking
- No metadata filtering
- No query expansion

**Add This:**
```python
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever

# Reranking
reranker = CohereRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vector_store.as_retriever()
)

# Query expansion
from langchain.retrievers import BM25Retriever
from langchain_core.retrievers import EnsembleRetriever

# Hybrid search (semantic + keyword)
bm25_retriever = BM25Retriever.from_documents(split_docs)
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_store.as_retriever(), bm25_retriever],
    weights=[0.6, 0.4]
)
```

---

### 2. **Conversation Memory** ❌

Currently Missing:
- No chat history
- Each query is stateless
- No context carryover

**Add This:**
```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Store conversation history
chat_history = ChatMessageHistory()

# Create history-aware RAG chain
history_aware_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)
```

---

### 3. **LLM Providers Flexibility** ⚠️

Currently: Only Groq

**Add Support For:**
```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatGroq, ChatCohere
from langchain_mistralai import ChatMistralAI

# Easy provider switching
providers = {
    "openai": lambda: ChatOpenAI(model="gpt-4"),
    "anthropic": lambda: ChatAnthropic(model="claude-3-opus"),
    "groq": lambda: ChatGroq(model="mixtral-8x7b"),
    "cohere": lambda: ChatCohere(model="command"),
    "mistral": lambda: ChatMistralAI(model="mistral-large")
}

llm = providers["openai"]()
```

---

### 4. **Query Analysis & Routing** ❌

Currently Missing:
- No query classification
- No routing to different retrieval strategies
- No query decomposition

**Add This:**
```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class QueryAnalysis(BaseModel):
    query_type: str = Field(description="Type: factual, comparison, synthesis, etc.")
    entities: list[str] = Field(description="Key entities in query")
    should_search: bool = Field(description="Whether to search documents")

# Route queries based on analysis
analyze_prompt = PromptTemplate.from_template("""
Analyze this query and determine its type.
Query: {query}
""")

analyzer = analyze_prompt | llm | PydanticOutputParser(pydantic_object=QueryAnalysis)
analysis = analyzer.invoke({"query": user_query})
```

---

### 5. **Response Evaluation & Quality Metrics** ❌

Currently Missing:
- No answer quality scoring
- No retrieval quality metrics
- No prompt injection detection

**Add This:**
```python
from langchain.evaluation import load_evaluator, EvaluatorType

# Evaluate retrieved documents
retrieval_evaluator = load_evaluator(EvaluatorType.PAIRWISE_STRING)

# Evaluate final answer
answer_evaluator = load_evaluator(EvaluatorType.QA)

# Check for hallucinations
def evaluate_hallucination(query, context, answer):
    hallucination_prompt = f"""
    Based on the context, is the answer hallucinated?
    Context: {context}
    Answer: {answer}
    """
    # Use LLM to score
```

---

### 6. **Caching & Performance** ⚠️

Currently Missing:
- No embedding caching
- No LLM response caching
- No performance monitoring

**Add This:**
```python
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache

# Cache LLM responses
set_llm_cache(SQLiteCache(database_path="../cache/llm_cache.db"))

# Cache embeddings
from langchain_core.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

store = LocalFileStore("../cache/embeddings")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    HuggingFaceEmbeddings(),
    store
)
```

---

### 7. **Logging & Monitoring** ❌

Currently Missing:
- No structured logging
- No performance tracking
- No debug information

**Add This:**
```python
import logging
from langchain.callbacks import StdOutCallbackHandler, FileCallbackHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add callbacks for debugging
callbacks = [
    StdOutCallbackHandler(),
    FileCallbackHandler(filename="../logs/rag_debug.log")
]

# Use in chain
response = rag_chain.invoke(
    {"input": question},
    config={"callbacks": callbacks}
)
```

---

### 8. **Error Handling & Fallbacks** ⚠️

Currently: Basic try-catch

**Add This:**
```python
from langchain.output_parsers import OutputFixingParser
from langchain_core.exceptions import OutputParserException

# Fallback chain
fallback_prompt = PromptTemplate.from_template(
    "Generate a helpful response about: {input}"
)

fallback_chain = fallback_prompt | llm | StrOutputParser()

# Main chain with fallback
main_chain = rag_chain.with_fallbacks([fallback_chain])

# Try different providers on failure
def get_response_with_fallback(question):
    try:
        return rag_chain.invoke({"input": question})
    except Exception as e:
        logger.warning(f"Main chain failed: {e}. Trying fallback...")
        return fallback_chain.invoke({"input": question})
```

---

## Comparison Table: Current vs LangChain

| Feature | Current | LangChain |
|---------|---------|-----------|
| **Document Loading** | Manual PDF loading | DirectoryLoader + multiple formats |
| **Text Splitting** | Recursive splitter | Multiple splitter options |
| **Embeddings** | SentenceTransformers direct | 20+ embedding providers abstracted |
| **Vector Store** | ChromaDB manual | 15+ vector DB integrations |
| **Retrieval** | Manual similarity search | 10+ retrieval strategies |
| **Prompt Management** | Simple string template | ChatPromptTemplate + message history |
| **LLM Integration** | Basic LangChain usage | 30+ LLM providers supported |
| **Chains** | Manual composition | LCEL for declarative chains |
| **RAG Chains** | DIY implementation | Pre-built `create_retrieval_chain()` |
| **Conversation** | Stateless | Built-in message history |
| **Caching** | None | LLM + embedding cache support |
| **Callbacks** | None | Full callback system |
| **Error Handling** | Basic | With fallbacks and error handlers |
| **Evaluation** | None | LLMEvaluator for quality scoring |
| **Monitoring** | None | Callbacks + logging integration |

---

## Step-by-Step Conversion Plan

### Phase 1: Core RAG (Week 1)
```
1. Replace DocumentIngestor with DirectoryLoader
2. Replace EmbeddingManager with HuggingFaceEmbeddings
3. Replace VectorStore with Chroma wrapper
4. Replace RAGRetriever with vector_store.as_retriever()
5. Implement create_retrieval_chain()
```

### Phase 2: Enhanced Features (Week 2)
```
6. Add conversation history support
7. Implement multiple LLM provider support
8. Add reranking (ContextualCompressionRetriever)
9. Add metadata filtering
10. Implement ensemble retrieval (hybrid search)
```

### Phase 3: Production Features (Week 3)
```
11. Add comprehensive logging and callbacks
12. Implement LLM + embedding caching
13. Add query analysis and routing
14. Implement answer evaluation
15. Add error handling and fallbacks
```

### Phase 4: Advanced Features (Week 4)
```
16. Add RAG evaluation metrics
17. Implement persistence for conversation history
18. Add web UI (Streamlit/Gradio)
19. Deploy to production
20. Add monitoring and analytics
```

---

## Migration Code Template

```python
# OLD WAY (Current Implementation)
from embeddings import EmbeddingManager
from vector_store import VectorStore
from retriever import RAGRetriever
from generator import GenericLLM
from document_ingestor import DocumentIngestor

# NEW WAY (LangChain)
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# TRANSITION: Can mix old and new during migration
# 1. Create new components alongside old ones
# 2. Gradually replace old imports
# 3. Keep tests for both implementations
# 4. Switch over component by component
```

---

## Key Takeaways

### Why Use LangChain's Built-in RAG?

1. **Abstraction** - Switch components without rewriting code
2. **Best Practices** - Follows ML community standards
3. **Composability** - Combine with other LangChain tools
4. **Features** - Built-in conversation, caching, evaluation
5. **Community** - Large ecosystem of integrations
6. **Maintainability** - Less code to maintain
7. **Debugging** - Callbacks and logging built-in

### Quick Wins (Easiest to Implement First)

✅ Replace manual embedding with HuggingFaceEmbeddings
✅ Replace custom retriever with vector_store.as_retriever()
✅ Replace manual chain with create_retrieval_chain()
✅ Add ChatPromptTemplate for better prompt management
✅ Enable logging with callbacks

### Biggest Impact Improvements

🎯 Add conversation history (create_history_aware_retriever)
🎯 Add reranking (ContextualCompressionRetriever)
🎯 Support multiple LLM providers
🎯 Implement query analysis and routing
🎯 Add comprehensive error handling

---

## Next Steps

1. **Start with Phase 1** - Convert core components first
2. **Keep current version** - Run both in parallel during transition
3. **Add tests** - For each new component
4. **Document changes** - Update docstrings and README
5. **Benchmark** - Compare performance before/after

This conversion will make the RAG system more robust, maintainable, and feature-rich!
