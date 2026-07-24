import os
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader , DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader

doc=Document(
    page_content="this is the main text content I am using to create RAG",
    metadata={
        "source":"exmaple.txt",
        "pages":1,
        "author":"Krish Naik",
        "date_created":"2025-01-01"
    }
)


## Create a simple txt file
os.makedirs("../data/text_files",exist_ok=True)
os.makedirs("../data/pdf_files",exist_ok=True)


sample_texts={
    "../data/text_files/python_intro.txt":"""Python Programming Introduction

Python is a high-level, interpreted programming language known for its simplicity and readability.
Created by Guido van Rossum and first released in 1991, Python has become one of the most popular
programming languages in the world.

Key Features:
- Easy to learn and use
- Extensive standard library
- Cross-platform compatibility
- Strong community support

Python is widely used in web development, data science, artificial intelligence, and automation.""",
    
    "../data/text_files/machine_learning.txt": """Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables systems to learn and improve
from experience without being explicitly programmed. It focuses on developing computer programs
that can access data and use it to learn for themselves.

Types of Machine Learning:
1. Supervised Learning: Learning with labeled data
2. Unsupervised Learning: Finding patterns in unlabeled data
3. Reinforcement Learning: Learning through rewards and penalties

Applications include image recognition, speech processing, and recommendation systems
    
    
    """

}

for filepath,content in sample_texts.items():
    with open(filepath,'w',encoding="utf-8") as f:
        f.write(content)

print("✅ Sample text files created!")

### TextLoader
loader=TextLoader("../data/text_files/python_intro.txt",encoding="utf-8")
document=loader.load()
print(document)

## load all the text files from the directory
dir_loader=DirectoryLoader(
    "../data/text_files",
    glob="**/*.txt", ## Pattern to match files  
    loader_cls= TextLoader, ##loader class to use
    loader_kwargs={'encoding': 'utf-8'},
    show_progress=False

)
documents=dir_loader.load()
documents


## load all the text files from the directory
dir_loader=DirectoryLoader(
    "../data/text_files",
    glob="**/*.txt", ## Pattern to match files  
    loader_cls= PyMuPDFLoader, ##loader class to use
    show_progress=False

)

pdf_documents=dir_loader.load()
pdf_documents


from pathlib import Path
import requests

DOWNLOAD_DIR = r"../data/pdf_files"

Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)

pdfs = {
    "Attention_Is_All_You_Need.pdf":
        "https://arxiv.org/pdf/1706.03762.pdf",

    "BERT.pdf":
        "https://arxiv.org/pdf/1810.04805.pdf",

    "GPT2.pdf":
        "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
}

headers = {
    "User-Agent": "Mozilla/5.0"
}

for filename, url in pdfs.items():
    filepath = Path(DOWNLOAD_DIR) / filename

    print(f"Downloading {filename}...")

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"✓ Saved to {filepath}")

    except Exception as e:
        print(f"✗ Failed: {filename}")
        print(e)

print("\nDone!")