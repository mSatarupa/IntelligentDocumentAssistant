import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

from .vectorStore import FaissVectorStore
from .data_loader import load_all_documents
from langchain_groq import ChatGroq

load_dotenv()


class RAGSearch:
    def __init__(
        self,
        data_dir: str = "data",
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.3-70b-versatile",
        force_rebuild: bool = False,
    ):
        """
        Initialize RAG system with flexible configuration.
        """

        # --- Paths ---
        self.data_path = Path(data_dir).resolve()
        self.persist_path = Path(persist_dir).resolve()

        # Ensure persist directory exists
        self.persist_path.mkdir(parents=True, exist_ok=True)

        # --- Vector Store ---
        self.vector_store = FaissVectorStore(
            persist_dir=str(self.persist_path),
            embedding_model=embedding_model
        )

        faiss_path = self.persist_path / "faiss_index"
        meta_path = self.persist_path / "metadata.pkl"

        # --- Build or Load ---
        if force_rebuild or not (faiss_path.exists() and meta_path.exists()):
            print("[INFO] Building vector store...")

            if not self.data_path.exists():
                raise FileNotFoundError(f"Data directory not found: {self.data_path}")

            docs = load_all_documents(str(self.data_path))

            if not docs:
                raise ValueError("No documents found to build vector store.")

            self.vector_store.build_from_document(docs)

        else:
            print("[INFO] Loading existing vector store...")
            self.vector_store.load()

        # --- LLM Initialization ---
        groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")

        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model
        )

        print(f"[INFO] Groq LLM initialized with model: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        """
        Perform semantic search and summarize results using LLM.
        """

        results = self.vector_store.query(query, top_k=top_k)

        texts = [
            r.get("metadata", {}).get("text", "")
            for r in results
            if r.get("metadata")
        ]

        context = "\n\n".join(texts).strip()

        if not context:
            return "No relevant information found."

        prompt = f"""
You are a helpful AI assistant.

Answer strictly based on the provided context.
- Be concise and accurate
- No hallucination
- Clean UI-friendly output

Question:
{query}

Context:
{context}
"""

        response = self.llm.invoke(prompt)
        return response.content