import os
import json
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pydantic import BaseModel
from pinecone import Pinecone
from uuid import uuid4

# Config
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
SOURCE_DIRECTORY = "./DataSet"
PINECONE_ENVIRONMENT = "us-east-1"

pc = Pinecone(api_key="pcsk_3aziQH_FhYmdBpXgFTL6borDTmnUcMFSnsXd55CXq5YdYmge17qF14iQP7ZScvAXxnT261")
index_name = "lawyerlens-kb"
index = pc.Index(index_name)

def load_json_files(json_paths: List[str]) -> List[Document]:
    """Load legal laws from JSON files and convert them into LangChain Documents."""
    documents = []
    for path in json_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
                for law in data:
                    text = law.get("text", "")
                    notes = law.get("interpretation_notes", "")
                    metadata = {
                        "source_id": law.get("source_id", ""),
                        "jurisdiction": law.get("jurisdiction", ""),
                        "title": law.get("title", ""),
                        "section": law.get("statute_section", ""),
                        "topics": law.get("topics", []),
                        "keywords": law.get("keywords", []),
                        "risk_relevance": json.dumps(law.get("risk_relevance", {})),  # <-- FIX
                        "url": law.get("source_url", "")
                    }
                    content = f"{text}\n\nNotes: {notes}"
                    documents.append(Document(page_content=content, metadata=metadata))
        except Exception as e:
            print(f"Error reading {path}: {e}")
    return documents

def get_text_chunks(documents: List[Document]) -> List[Document]:
    """Split long text in documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    split_docs = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            split_docs.append(Document(page_content=chunk, metadata=doc.metadata))
    return split_docs

def main():
    print("Building knowledge base and uploading to Pinecone...")

    # 1. Load source JSON files
    json_paths = [os.path.join(SOURCE_DIRECTORY, f) for f in os.listdir(SOURCE_DIRECTORY) if f.endswith(".json")]
    if not json_paths:
        print(f"No JSON documents found in the '{SOURCE_DIRECTORY}' folder.")
        return
    print(f"Found {len(json_paths)} JSON files to process.")

    # 2. Load JSON as documents
    documents = load_json_files(json_paths)
    print(f"Loaded {len(documents)} laws from JSON.")

    # 3. Split into chunks
    documents = get_text_chunks(documents)
    print(f"Split into {len(documents)} chunks.")

    # 4. Initialize the embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )

    # 5. Upload documents to Pinecone
    print(f"Uploading {len(documents)} chunks to Pinecone index '{index_name}'...")
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)

    print("\n ✅ Knowledge base uploaded to Pinecone successfully!")

if __name__ == '__main__':
    main()
