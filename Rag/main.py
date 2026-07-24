from embeddings import EmbeddingManager
from generator import GenericLLM  # Updated import from GroqLLM to GenericLLM
from document_ingestor import DocumentIngestor
from retriever import RAGRetriever
from vector_store import VectorStore


def build_pipeline():
    """
    Step 1: Set up the database and retriever.
    If the database is empty, it processes documents and saves them first.
    """
    print("--- 1. Setting up Embeddings & Vector Store ---")
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()

    # If database has 0 documents, load PDFs and add them
    if vector_store.collection.count() == 0:
        print("Vector store is empty! Ingesting PDFs...")
        ingestor = DocumentIngestor()
        chunks = ingestor.run()

        # Extract text from chunks and convert them into vector numbers
        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)
        
        # Save both text chunks and vector embeddings into ChromaDB
        vector_store.add_documents(chunks, embeddings)

    # Initialize retriever with the populated vector store
    retriever = RAGRetriever(
        vector_store=vector_store, 
        embedding_manager=embedding_manager
    )
    
    return retriever


def build_context(retrieved_docs: list) -> str:
    """
    Step 2: Take the list of retrieved document chunks and join 
    them into a single text block for the LLM.
    """
    formatted_chunks = []
    for doc in retrieved_docs:
        chunk_text = f"[Document Chunk (Score: {doc['similarity_score']})]:\n{doc['content']}"
        formatted_chunks.append(chunk_text)
        
    return "\n\n".join(formatted_chunks)


def answer_query(query: str, retriever: RAGRetriever, llm: GenericLLM,) -> str:
    """
    Step 3: Complete RAG Loop
    - Find closest document chunks (Retrieval)
    - Pass query + chunks to LLM (Generation)
    """
    print(f"\nSearching for chunks matching: '{query}'...")
    retrieved_docs = retriever.retrieve(query)

    # Fallback if no relevant documents were found
    if not retrieved_docs:
        return "I couldn't find any relevant information in the documents to answer that question."

    # Format retrieved document chunks into a single context string
    context = build_context(retrieved_docs)
    
    # Send formatted context + user query to the LLM
    print("Generating answer using LLM...")
    return llm.generate_response(query=query, context=context)


def main():
    # 1. Build database and retrieve system
    retriever = build_pipeline()

    # You can easily change provider to 'openai', 'anthropic', etc. here!
    llm = GenericLLM()

    # 3. Ask your question!
    user_query = "What is attention is all you need"
    final_answer = answer_query(user_query, retriever, llm)

    # 4. Print output
    print("\n================ FINAL ANSWER ================")
    print(final_answer)
    print("==============================================")


if __name__ == "__main__":
    main()