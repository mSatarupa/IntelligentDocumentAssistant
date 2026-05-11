import os
import uuid
import shutil
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, status, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from util.search import RAGSearch
from util.vectorStore import FaissVectorStore
from util.data_loader import load_all_documents

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants (Single Source of Truth) ─────────────────────────────────────────
DATA_DIR = "data"
FAISS_STORE = "faiss_store"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"

ALLOWED_EXTS = {
    ".txt", ".pdf", ".md", ".json", ".csv",
    ".docx", ".xlsx"
}

# ── In-memory result store ─────────────────────────────────────────────────────
result_store: dict[str, dict] = {}
rag_search: Optional[RAGSearch] = None


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_search

    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        rag_search = RAGSearch(
            data_dir=DATA_DIR,
            persist_dir=FAISS_STORE,
            embedding_model=EMBED_MODEL,
            llm_model=LLM_MODEL
        )
        logger.info("✅ RAG system initialised.")
    except Exception as e:
        logger.warning(f"⚠️ RAG init skipped: {e}")
        rag_search = None

    yield

    logger.info("🛑 Shutting down ytRAG API.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="ytRAG API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔒 Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static Files ──────────────────────────────────────────────────────────────
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", tags=["Frontend"])
def serve_frontend():
    return FileResponse("static/index.html")


# ── Schemas ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str


class QuerySubmitResponse(BaseModel):
    query_id: str
    message: str


class OutputResponse(BaseModel):
    query_id: str
    query: str
    response: str
    status: str


class UploadResponse(BaseModel):
    filename: str
    message: str


# ── Reindex Helper (Single Control Point) ──────────────────────────────────────
def _reindex() -> int:
    """Rebuild FAISS index using RAGSearch as single source of truth."""
    global rag_search

    rag_search = RAGSearch(
        data_dir=DATA_DIR,
        persist_dir=FAISS_STORE,
        embedding_model=EMBED_MODEL,
        llm_model=LLM_MODEL,
        force_rebuild=True   # 🔥 critical
    )

    try:
        from util.data_loader import load_all_documents
        docs = load_all_documents(DATA_DIR)
        doc_count = len(docs)
    except Exception:
        doc_count = -1  # fallback

    logger.info("🔄 Re-index completed. Docs: %s", doc_count)
    return doc_count


# ── Upload API ────────────────────────────────────────────────────────────────
@app.post(
    "/api/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Upload"],
)
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[-1].lower()

    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTS}",
        )

    dest = os.path.join(DATA_DIR, file.filename)

    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info("📁 Saved: %s", dest)
    except Exception as e:
        raise HTTPException(500, f"Failed to save file: {e}")
    finally:
        await file.close()

    try:
        doc_count = _reindex()
    except Exception as e:
        logger.exception("Indexing failed.")
        raise HTTPException(500, f"File saved but indexing failed: {e}")

    return UploadResponse(
        filename=file.filename,
        message=f"✅ Uploaded & indexed. Total docs: {doc_count}",
    )


# ── Query API ─────────────────────────────────────────────────────────────────
@app.post(
    "/api/query",
    response_model=QuerySubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Query"],
)
async def submit_query(request: QueryRequest):

    if not request.query.strip():
        raise HTTPException(422, "Query must not be empty.")

    if rag_search is None:
        raise HTTPException(503, "RAG system not ready.")

    query_id = str(uuid.uuid4())

    try:
        response = rag_search.search_and_summarize(request.query)

        result_store[query_id] = {
            "query": request.query,
            "response": response,
            "status": "ready"
        }

        logger.info("🔍 Query processed: %s", query_id)

    except Exception as e:
        logger.exception("RAG failed.")
        raise HTTPException(500, f"Search failed: {e}")

    return QuerySubmitResponse(
        query_id=query_id,
        message="Query processed. Use GET /api/output"
    )


# ── Output API ────────────────────────────────────────────────────────────────
@app.get(
    "/api/output",
    response_model=OutputResponse,
    tags=["Output"],
)
async def get_output(query_id: str = Query(...)):

    result = result_store.get(query_id)

    if not result:
        raise HTTPException(404, "Query ID not found.")

    return OutputResponse(
        query_id=query_id,
        query=result["query"],
        response=result["response"],
        status=result["status"]
    )


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Meta"])
def health():
    return {
        "status": "ok",
        "rag_ready": rag_search is not None,
        "results_cached": len(result_store)
    }