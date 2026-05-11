
from pathlib import Path
from typing import List
from langchain_core.documents import Document

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    JSONLoader
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader


def load_all_documents(data_dir: str) -> List[Document]:
    """Load all documents recursively from a directory (PDF, CSV, TXT, MD, JSON, DOCX, XLSX)."""

    data_path = Path(data_dir).resolve()
    print(f"[INFO] Loading documents from: {data_path}")

    documents: List[Document] = []

    # --- PDF ---
    for file in data_path.glob("**/*.pdf"):
        try:
            loader = PyPDFLoader(str(file))
            docs = loader.load()
            print(f"[PDF] {file.name} → {len(docs)} pages")
            documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] PDF {file.name}: {e}")

    # --- CSV ---
    for file in data_path.glob("**/*.csv"):
        try:
            loader = CSVLoader(str(file))
            docs = loader.load()
            print(f"[CSV] {file.name} → {len(docs)} rows")
            documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] CSV {file.name}: {e}")

    # --- TEXT (.txt, .md) ---
    for ext in ["*.txt", "*.md"]:
        for file in data_path.glob(f"**/{ext}"):
            try:
                loader = TextLoader(str(file), encoding="utf-8")
                docs = loader.load()
                print(f"[TEXT] {file.name}")
                documents.extend(docs)
            except Exception as e:
                print(f"[ERROR] TEXT {file.name}: {e}")

    # --- JSON ---
    for file in data_path.glob("**/*.json"):
        try:
            loader = JSONLoader(
                file_path=str(file),
                jq_schema=".",   
                text_content=False
            )
            docs = loader.load()
            print(f"[JSON] {file.name}")
            documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] JSON {file.name}: {e}")

    # --- DOCX ---
    for file in data_path.glob("**/*.docx"):
        try:
            loader = Docx2txtLoader(str(file))
            docs = loader.load()
            print(f"[DOCX] {file.name}")
            documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] DOCX {file.name}: {e}")

    # --- EXCEL ---
    for file in data_path.glob("**/*.xlsx"):
        try:
            loader = UnstructuredExcelLoader(str(file))
            docs = loader.load()
            print(f"[EXCEL] {file.name}")
            documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] EXCEL {file.name}: {e}")

    print(f"\n✅ Total documents loaded: {len(documents)}")

    return documents