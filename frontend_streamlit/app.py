import streamlit as st
from pathlib import Path
import traceback

from rag_loader import load_rag_helpers

st.set_page_config(page_title="RAG Streamlit Frontend", layout="wide")

def ensure_pipeline():
    if "retriever" not in st.session_state:
        build_pipeline, _, GenericLLM = load_rag_helpers()
        with st.spinner("Building RAG pipeline (may ingest docs first)..."):
            retriever = build_pipeline()
        # instantiate LLM
        llm = GenericLLM()
        st.session_state.retriever = retriever
        st.session_state.llm = llm
        st.session_state.built = True

def main():
    st.title("RAG — Streamlit Frontend")

    st.sidebar.header("Controls")
    if st.sidebar.button("Build Pipeline / (Re)Load"):
        try:
            ensure_pipeline()
            st.sidebar.success("Pipeline ready")
        except Exception as e:
            st.sidebar.error(f"Error building pipeline: {e}")
            st.sidebar.text(traceback.format_exc())

    query = st.text_area("Your question", height=120)

    col1, col2 = st.columns([3, 2])

    with col1:
        if st.button("Ask"):
            if "retriever" not in st.session_state:
                try:
                    ensure_pipeline()
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.stop()

            retriever = st.session_state.retriever
            llm = st.session_state.llm

            if not query.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Retrieving relevant documents..."):
                    try:
                        # show retrieved docs and final answer
                        retrieved = retriever.retrieve(query)
                    except Exception as e:
                        st.error(f"Retrieval error: {e}")
                        st.stop()

                if not retrieved:
                    st.info("No relevant documents found.")
                else:
                    with st.expander("Retrieved Chunks", expanded=True):
                        for i, doc in enumerate(retrieved, start=1):
                            st.write(f"**Chunk {i} — Score:** {doc.get('similarity_score')}")
                            st.write(doc.get("content"))

                with st.spinner("Generating answer..."):
                    try:
                        answer = load_rag_helpers()[1](query=query, retriever=retriever, llm=llm)
                    except Exception as e:
                        st.error(f"Generation error: {e}")
                        st.stop()

                st.subheader("Answer")
                st.write(answer)

    with col2:
        st.subheader("Status")
        st.write("Pipeline built:" if st.session_state.get("built") else "Pipeline not built")
        st.write("Retriever:" )
        st.write(type(st.session_state.get("retriever")).__name__ if st.session_state.get("retriever") else "—")


if __name__ == "__main__":
    main()
