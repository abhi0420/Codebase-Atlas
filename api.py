"""
Codebase Atlas API
FastAPI endpoints for hybrid code search and AI-powered code assistance.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn

from chroma_setup import (
    read_jsonl_file,
    create_collection_from_data,
    create_bm25_store,
    hybrid_search,
    vector_search,
    bm25_search,
    call_model,
    chroma_client,
    embedding
)

# ============================================================================
# App Configuration
# ============================================================================

app = FastAPI(
    title="Codebase Atlas API",
    description="Hybrid search API for code exploration using vector embeddings and BM25",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================================
# Global State (initialized on startup)
# ============================================================================

class AppState:
    data: List[dict] = []
    collection = None
    bm25 = None
    bm25_ids: List[str] = []
    initialized: bool = False

state = AppState()

# ============================================================================
# Request/Response Models
# ============================================================================

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The search query")
    top_n: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    alpha: float = Field(default=0.6, ge=0, le=1, description="Weight for vector search (0-1)")
    beta: float = Field(default=0.4, ge=0, le=1, description="Weight for BM25 search (0-1)")

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The question about the codebase")
    top_n: int = Field(default=3, ge=1, le=10, description="Number of context results to use")
    alpha: float = Field(default=0.6, ge=0, le=1, description="Weight for vector search")
    beta: float = Field(default=0.4, ge=0, le=1, description="Weight for BM25 search")

class CodeNode(BaseModel):
    id: str
    name: str
    node_type: str
    file: str
    line_no: int
    end_line_no: int
    score: float
    docstring: Optional[str] = None
    source_code: Optional[str] = None
    args: Optional[List[str]] = None

class SearchResponse(BaseModel):
    query: str
    results: List[CodeNode]
    total_results: int

class AskResponse(BaseModel):
    query: str
    answer: str
    sources: List[CodeNode]

class HealthResponse(BaseModel):
    status: str
    initialized: bool
    total_nodes: int
    collection_count: int

# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the search indices on startup."""
    try:
        print("\n" + "="*60)
        print("Initializing Codebase Atlas API...")
        print("="*60 + "\n")
        
        # Load data
        state.data = read_jsonl_file("parsed_python_files.jsonl")
        if not state.data:
            print("WARNING: No data loaded from JSONL file")
            return
        
        # Create ChromaDB collection
        state.collection = create_collection_from_data(state.data)
        
        # Create BM25 index
        state.bm25, state.bm25_ids = create_bm25_store(state.data)
        
        state.initialized = True
        print("\n" + "="*60)
        print("✓ API initialized successfully!")
        print(f"  - Loaded {len(state.data)} code nodes")
        print(f"  - Collection has {state.collection.count()} documents")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"ERROR during startup: {e}")
        state.initialized = False

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Serve the frontend."""
    return FileResponse("static/index.html")

@app.get("/api", tags=["Info"])
async def api_info():
    """API info endpoint."""
    return {
        "name": "Codebase Atlas API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Check API health and initialization status."""
    return HealthResponse(
        status="healthy" if state.initialized else "not_initialized",
        initialized=state.initialized,
        total_nodes=len(state.data),
        collection_count=state.collection.count() if state.collection else 0
    )

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest):
    """
    Perform hybrid search combining vector embeddings and BM25.
    
    - **query**: The search query (e.g., "upload file to GCS")
    - **top_n**: Number of results to return (default: 5)
    - **alpha**: Weight for semantic/vector search (default: 0.6)
    - **beta**: Weight for keyword/BM25 search (default: 0.4)
    """
    if not state.initialized:
        raise HTTPException(status_code=503, detail="Service not initialized. Check /health endpoint.")
    
    if abs(request.alpha + request.beta - 1.0) > 0.01:
        raise HTTPException(status_code=400, detail=f"alpha + beta must equal 1.0 (got {request.alpha + request.beta})")
    
    try:
        # Perform hybrid search
        results = hybrid_search(
            state.collection,
            state.bm25,
            state.bm25_ids,
            request.query,
            top_n=request.top_n,
            alpha=request.alpha,
            beta=request.beta
        )
        
        # Build response
        code_nodes = []
        for node_id, score in results:
            node = next((n for n in state.data if n["id"] == node_id), None)
            if node:
                code_nodes.append(CodeNode(
                    id=node_id,
                    name=node["name"],
                    node_type=node["node_type"],
                    file=node["id"].split("::")[0],
                    line_no=node["line_no"],
                    end_line_no=node["end_line_no"],
                    score=round(score, 4),
                    docstring=node.get("node_docstring"),
                    source_code=node.get("source_code"),
                    args=node.get("args", [])
                ))
        
        return SearchResponse(
            query=request.query,
            results=code_nodes,
            total_results=len(code_nodes)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/ask", response_model=AskResponse, tags=["AI"])
async def ask(request: AskRequest):
    """
    Ask a question about the codebase and get an AI-powered answer.
    
    Uses hybrid search to find relevant code, then sends it to GPT-4 for analysis.
    
    - **query**: Your question (e.g., "How do I execute a BigQuery query?")
    - **top_n**: Number of code snippets to include as context (default: 3)
    """
    if not state.initialized:
        raise HTTPException(status_code=503, detail="Service not initialized. Check /health endpoint.")
    
    try:
        # Perform hybrid search for context
        results = hybrid_search(
            state.collection,
            state.bm25,
            state.bm25_ids,
            request.query,
            top_n=request.top_n,
            alpha=request.alpha,
            beta=request.beta
        )
        
        if not results:
            return AskResponse(
                query=request.query,
                answer="No relevant code found for your query.",
                sources=[]
            )
        
        # Get AI response
        answer = call_model(request.query, results, state.data)
        
        # Build sources
        sources = []
        for node_id, score in results:
            node = next((n for n in state.data if n["id"] == node_id), None)
            if node:
                sources.append(CodeNode(
                    id=node_id,
                    name=node["name"],
                    node_type=node["node_type"],
                    file=node["id"].split("::")[0],
                    line_no=node["line_no"],
                    end_line_no=node["end_line_no"],
                    score=round(score, 4),
                    docstring=node.get("node_docstring"),
                    args=node.get("args", [])
                ))
        
        return AskResponse(
            query=request.query,
            answer=answer,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ask failed: {str(e)}")

@app.get("/search/vector", response_model=SearchResponse, tags=["Search"])
async def search_vector_only(
    query: str = Query(..., min_length=1, description="The search query"),
    top_n: int = Query(default=5, ge=1, le=20, description="Number of results")
):
    """Perform vector-only semantic search."""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="Service not initialized.")
    
    try:
        results = vector_search(state.collection, query, top_n=top_n)
        
        code_nodes = []
        for node_id, score in results.items():
            node = next((n for n in state.data if n["id"] == node_id), None)
            if node:
                code_nodes.append(CodeNode(
                    id=node_id,
                    name=node["name"],
                    node_type=node["node_type"],
                    file=node["id"].split("::")[0],
                    line_no=node["line_no"],
                    end_line_no=node["end_line_no"],
                    score=round(score, 4),
                    docstring=node.get("node_docstring"),
                    args=node.get("args", [])
                ))
        
        # Sort by score descending
        code_nodes.sort(key=lambda x: x.score, reverse=True)
        
        return SearchResponse(
            query=query,
            results=code_nodes,
            total_results=len(code_nodes)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")

@app.get("/search/bm25", response_model=SearchResponse, tags=["Search"])
async def search_bm25_only(
    query: str = Query(..., min_length=1, description="The search query"),
    top_n: int = Query(default=5, ge=1, le=20, description="Number of results")
):
    """Perform BM25 keyword search only."""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="Service not initialized.")
    
    try:
        results = bm25_search(state.bm25, state.bm25_ids, query, top_n=top_n)
        
        code_nodes = []
        for node_id, score in results.items():
            node = next((n for n in state.data if n["id"] == node_id), None)
            if node:
                code_nodes.append(CodeNode(
                    id=node_id,
                    name=node["name"],
                    node_type=node["node_type"],
                    file=node["id"].split("::")[0],
                    line_no=node["line_no"],
                    end_line_no=node["end_line_no"],
                    score=round(score, 4),
                    docstring=node.get("node_docstring"),
                    args=node.get("args", [])
                ))
        
        # Sort by score descending
        code_nodes.sort(key=lambda x: x.score, reverse=True)
        
        return SearchResponse(
            query=query,
            results=code_nodes,
            total_results=len(code_nodes)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BM25 search failed: {str(e)}")

@app.get("/nodes", tags=["Data"])
async def list_nodes(
    file: Optional[str] = Query(default=None, description="Filter by filename"),
    node_type: Optional[str] = Query(default=None, description="Filter by type (function/class)")
):
    """List all indexed code nodes with optional filtering."""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="Service not initialized.")
    
    nodes = state.data
    
    if file:
        nodes = [n for n in nodes if file.lower() in n["id"].split("::")[0].lower()]
    
    if node_type:
        nodes = [n for n in nodes if n["node_type"].lower() == node_type.lower()]
    
    return {
        "total": len(nodes),
        "nodes": [
            {
                "id": n["id"],
                "name": n["name"],
                "type": n["node_type"],
                "file": n["id"].split("::")[0],
                "line_no": n["line_no"],
                "end_line_no": n["end_line_no"]
            }
            for n in nodes
        ]
    }

@app.get("/nodes/{node_id:path}", tags=["Data"])
async def get_node(node_id: str):
    """Get full details of a specific code node by ID."""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="Service not initialized.")
    
    node = next((n for n in state.data if n["id"] == node_id), None)
    
    if not node:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    
    return node

# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
