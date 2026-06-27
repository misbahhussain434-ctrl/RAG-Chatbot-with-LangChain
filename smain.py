"""
standalone.py
=============
A minimal Retrieval-Augmented Generation (RAG) web service.

Pipeline overview:
  Phase 1 - Ingest:  Upload a file -> LangChain loads it and splits it into chunks
                     -> the chunks are embedded and stored in ChromaDB (a vector DB).
  Phase 2 - Query:   The user asks a question -> the Groq LLM refines the question
                     -> ChromaDB returns the most similar chunks
                     -> the Groq LLM writes the final answer using those chunks.

Tools used:
  - LangChain              : load files (PDF/TXT) and split them into chunks
  - ChromaDB               : vector database for semantic (meaning-based) search
  - Sentence-Transformers  : turns text into embeddings (vectors of numbers)
  - Groq (Llama-3)         : the Large Language Model that refines queries and writes answers
  - FastAPI                : exposes the /upload and /query HTTP endpoints
"""

# --- Standard library imports ---
import os                            # read environment variables (API keys, paths)
import shutil                        # copy the uploaded file stream to disk
from pathlib import Path             # safe, cross-platform file paths
from typing import List, Dict, Any   # type hints for clearer function signatures

# --- Third-party helpers ---
from dotenv import load_dotenv       # loads variables from the .env file into the environment
from loguru import logger            # simple, readable logging
from fastapi import FastAPI, UploadFile, File, HTTPException  # web framework + file upload
from pydantic import BaseModel       # request/response data validation models

# --- LangChain / Chroma imports (the RAG building blocks) ---
from langchain_community.document_loaders import PyPDFLoader, TextLoader   # read PDF / text files
from langchain_text_splitters import RecursiveCharacterTextSplitter       # split text into chunks
from langchain_chroma import Chroma                                       # the vector store
from langchain_community.embeddings import SentenceTransformerEmbeddings  # text -> vectors (embeddings)
from langchain_groq import ChatGroq                                       # the Groq LLM client

# Load the .env file so the os.getenv(...) calls below can see GROQ_API_KEY, etc.
load_dotenv()

# --- Configuration ---
# Every setting is read from the .env file. The second argument to os.getenv()
# is the default value used when that variable is not present in .env.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")                          # secret key for Groq (never hard-code it)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")   # which LLM to run on Groq
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")               # folder where ChromaDB saves vectors
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "redjet_collection")  # name of the vector collection
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")  # embedding model
DATA_DIR = Path("./data")                                        # folder where uploaded files are stored

# --- Prompts ---
# SYSTEM_PROMPT sets the AI's persona / overall behaviour.
SYSTEM_PROMPT = (
    "You are a precise, pragmatic assistant. You refine user queries to maximize retrieval "
    "quality, maintain factual grounding, and respond with clear, concise explanations "
    "based strictly on provided context. If context is insufficient, state limitations."
)

# DEVELOPER_PROMPT gives the strict working rules (answer using ONLY the retrieved chunks).
DEVELOPER_PROMPT = (
    "First, refine the user’s raw query to better target the corpus. Then answer using only "
    "the retrieved chunks. Prefer technical clarity, cite chunk IDs inline where relevant, "
    "and avoid speculation."
)

# REFINE_TEMPLATE is used to rewrite a messy question into a better search query.
REFINE_TEMPLATE = (
    "Refine the following query to maximize retrieval quality from a technical corpus.\n\n"
    "Raw Query:\n{query}\n\n"
    "Output only the refined query, no commentary."
)

# ANSWER_TEMPLATE glues the question + the retrieved context together for the final answer.
ANSWER_TEMPLATE = (
    "System:\n{system}\nDeveloper:\n{dev}\n\n"
    "User Query: {query}\nRefined Query: {refined}\n\n"
    "Context Chunks (id: content):\n{context}\n\n"
    "Task: Provide a direct, well-structured answer grounded strictly in the context. "
    "Prefer concise, technical clarity. If uncertain or insufficient context, say so."
)

# --- FastAPI App ---
# Create the web application object that will hold our endpoints.
app = FastAPI(title="Redjet RAG Service with Upload", version="1.1.0")

# --- Data Models ---
# Pydantic models describe the JSON shape of requests and responses,
# and they automatically validate the incoming data.

class QueryRequest(BaseModel):
    query: str               # the user's question
    top_k: int = 3           # how many chunks to retrieve from ChromaDB
    min_score: float = 0.0   # optional minimum similarity score to keep a chunk

class ChunkResult(BaseModel):
    chunk_id: str            # identifier like "file.pdf-chunk-0"
    score: float             # similarity score for this chunk
    content: str             # the chunk text
    metadata: Dict[str, Any] # extra info (source file, page, etc.)

class QueryResponse(BaseModel):
    refined_query: str             # the improved query Groq produced
    top_chunks: List[ChunkResult]  # the chunks used to build the answer
    answer: str                    # the final AI answer

class UploadResponse(BaseModel):
    filename: str            # name of the uploaded file
    status: str              # human-readable status message
    chunks_added: int        # how many chunks were stored

# --- Helper Functions (Core Logic) ---

def get_llm():
    """Create the Groq chat model. Fails fast if the API key is missing."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY missing in environment variables.")
    return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)

def get_vectorstore():
    """Open (or create) the ChromaDB collection using our embedding model."""
    # The embedding function turns text into vectors so Chroma can compare by meaning.
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        collection_name=CHROMA_COLLECTION,   # which collection to use
        persist_directory=CHROMA_DIR,        # where vectors are saved on disk
        embedding_function=embeddings,       # how to embed text
    )

def process_file(file_path: Path):
    """Parse a file, chunk it, and add it to the vector store. Returns the chunk count."""
    logger.info(f"Processing file: {file_path}")

    # 1. Load -- pick the right loader based on the file extension.
    docs = []
    if file_path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(file_path))                   # read a PDF
        docs.extend(loader.load())
    elif file_path.suffix.lower() in [".txt", ".md", ".log"]:
        loader = TextLoader(str(file_path), encoding="utf-8")  # read plain text
        docs.extend(loader.load())
    else:
        logger.warning(f"Unsupported file type: {file_path}")
        return 0

    if not docs:                                  # nothing could be read from the file
        logger.warning(f"No content found in {file_path}")
        return 0

    # 2. Chunk -- split big documents into ~1000-character pieces.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,                          # max size of each chunk
        chunk_overlap=150,                        # repeat 150 chars so sentences are not cut in half
        separators=["\n\n", "\n", " ", ""]        # try to split on paragraphs, then lines, then words
    )
    chunks = splitter.split_documents(docs)

    # 3. Add Metadata -- give every chunk a unique id and remember its source file.
    for i, c in enumerate(chunks):
        c.metadata = c.metadata or {}
        # Make a simple ID based on filename and index to help tracking
        c.metadata["chunk_id"] = f"{file_path.name}-chunk-{i}"
        c.metadata["source"] = str(file_path)

    # 4. Upsert to Chroma -- embed the chunks and store them.
    if chunks:
        vs = get_vectorstore()
        texts = [c.page_content for c in chunks]          # the raw text of each chunk
        metadatas = [c.metadata for c in chunks]          # the metadata of each chunk
        ids = [m.get("chunk_id") for m in metadatas]      # stable ids (so re-uploads update, not duplicate)

        logger.info(f"Upserting {len(chunks)} chunks from {file_path.name}")
        vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        # Note: newer langchain-chroma auto-persists, so no vs.persist() needed
        return len(chunks)

    return 0

def refine_query(llm: ChatGroq, raw_query: str) -> str:
    """Ask the LLM to rewrite the user's question into a better search query."""
    prompt = REFINE_TEMPLATE.format(query=raw_query)
    resp = llm.invoke(prompt)
    refined = resp.content.strip()
    logger.info(f"Refined query: {refined}")
    return refined

def retrieve_chunks(vs: Chroma, query: str, top_k: int) -> List[Dict[str, Any]]:
    """Search ChromaDB and return the top_k most similar chunks (with their scores)."""
    results = vs.similarity_search_with_relevance_scores(query, k=top_k)
    structured = []
    for doc, score in results:
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        structured.append({
            "chunk_id": chunk_id,
            "score": float(score),
            "content": doc.page_content,
            "metadata": doc.metadata
        })
    return structured

def answer_with_context(llm: ChatGroq, user_query: str, refined_query: str, chunks: List[Dict[str, Any]]) -> str:
    """Build the final prompt from the chunks and ask the LLM for the grounded answer."""
    # Turn each chunk into an "id: content" line, joined by dividers.
    context_lines = [f"{c['chunk_id']}: {c['content']}" for c in chunks]
    context_str = "\n---\n".join(context_lines)

    prompt = ANSWER_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        dev=DEVELOPER_PROMPT,
        query=user_query,
        refined=refined_query,
        context=context_str
    )
    resp = llm.invoke(prompt)
    return resp.content.strip()

# --- Endpoints ---

@app.on_event("startup")
async def startup_event():
    # Ensure the ./data folder exists before any upload happens.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Server ready. Data dir: {DATA_DIR.absolute()}")

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads a file (PDF/Text), saves it to ./data, and ingests it into ChromaDB.
    """
    try:
        # 1. Save File -- stream the upload to disk.
        file_path = DATA_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Ingest -- load, chunk and store it in the vector database.
        chunks_count = process_file(file_path)

        msg = f"Successfully processed and ingested {file.filename} into the vector database. Added {chunks_count} new chunks of data."
        return UploadResponse(
            filename=file.filename,
            status=msg,
            chunks_added=chunks_count
        )
    except Exception as e:
        logger.error(f"Error processing file upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(payload: QueryRequest):
    """
    Queries the vector store and returns an AI answer grounded in the documents.
    """
    llm = get_llm()
    vs = get_vectorstore()

    # 1. Refine -- improve the raw question for better retrieval.
    refined = refine_query(llm, payload.query)

    # 2. Retrieve -- get the top_k most similar chunks from ChromaDB.
    raw_chunks = retrieve_chunks(vs, refined, payload.top_k)

    # 3. Filter (optional) -- drop weak matches below min_score.
    if payload.min_score > 0:
        filtered = [c for c in raw_chunks if c["score"] >= payload.min_score]
    else:
        filtered = raw_chunks

    # 4. Answer -- ask the LLM using the question + the retrieved chunks.
    answer = answer_with_context(llm, payload.query, refined, filtered)

    # Build the structured response objects.
    top_chunks = [
        ChunkResult(
            chunk_id=c["chunk_id"],
            score=c["score"],
            content=c["content"],
            metadata=c["metadata"]
        ) for c in filtered
    ]

    return QueryResponse(
        refined_query=refined,
        top_chunks=top_chunks,
        answer=answer
    )

@app.get("/health")
def health():
    # Simple health-check endpoint to confirm the server is running.
    return {"status": "ok", "collection": CHROMA_COLLECTION}