import argparse
from pathlib import Path
import sys

from rag_loader import load_rag_helpers


def main():
    parser = argparse.ArgumentParser(description="Run a quick RAG smoke test (no Streamlit)")
    parser.add_argument("-q", "--query", type=str, help="Question to ask the RAG", required=False)
    args = parser.parse_args()

    build_pipeline, answer_query, GenericLLM = load_rag_helpers()

    print("Building pipeline (may ingest documents)...")
    retriever = build_pipeline()
    print("Pipeline ready")

    llm = GenericLLM()

    if args.query:
        query = args.query
    else:
        query = input("Enter a question to test: ")

    print(f"\nQuery: {query}\n")

    try:
        retrieved = retriever.retrieve(query)
    except Exception as e:
        print(f"Retrieval error: {e}")
        sys.exit(1)

    if not retrieved:
        print("No relevant documents found.")
    else:
        print("Retrieved chunks:")
        for i, doc in enumerate(retrieved, start=1):
            score = doc.get("similarity_score")
            print(f"--- Chunk {i} (score={score}) ---")
            print(doc.get("content")[:1000])

    print("\nGenerating answer...")
    try:
        answer = answer_query(query=query, retriever=retriever, llm=llm)
        print("\nAnswer:\n")
        print(answer)
    except Exception as e:
        print(f"Generation error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
