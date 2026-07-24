# RAG Streamlit Frontend

This folder contains a lightweight Streamlit frontend to interact with the RAG core located in the `Rag` folder.

Quick start

1. (Optional) Create a virtual environment and activate it.

2. Install frontend dependencies:

```bash
pip install -r frontend_streamlit/requirements.txt
```

3. Run the app:

```bash
streamlit run frontend_streamlit/app.py
```

Notes
- The frontend dynamically loads the core modules from the `Rag` folder using `rag_loader.py` so you don't need to modify the core code.
- The first pipeline build may ingest documents if your vector store is empty; this can take time.
