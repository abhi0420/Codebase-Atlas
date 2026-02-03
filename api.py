"""
Codebase Atlas API - Multi-Repository Code Search
FastAPI server with dynamic repository indexing
"""

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
import uvicorn
import hashlib
import json
import os
import shutil
import tempfile
import subprocess
import re
import stat


def force_remove_readonly(func, path, excinfo):
    """Error handler for shutil.rmtree to handle read-only files on Windows."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

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
from repo_parser import parse_repository, get_repo_hash

# Upload directory for repos
UPLOAD_DIR = Path(__file__).parent / "uploaded_repos"
UPLOAD_DIR.mkdir(exist_ok=True)

# ============================================================================
# Request/Response Models
# ============================================================================

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The search query")
    top_n: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    alpha: float = Field(default=0.6, ge=0, le=1, description="Weight for vector search (0-1)")
    beta: float = Field(default=0.4, ge=0, le=1, description="Weight for BM25 search (0-1)")
    repo_id: Optional[str] = Field(default=None, description="Repository to search (uses active if not specified)")

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The question about the codebase")
    top_n: int = Field(default=2, ge=1, le=10, description="Number of context results to use")
    alpha: float = Field(default=0.6, ge=0, le=1, description="Weight for vector search")
    beta: float = Field(default=0.4, ge=0, le=1, description="Weight for BM25 search")
    repo_id: Optional[str] = Field(default=None, description="Repository to search")

class IndexRepoRequest(BaseModel):
    repo_path: str = Field(..., description="Path to the repository to index")
    force_reindex: bool = Field(default=False, description="Force re-indexing even if unchanged")

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
    repo_id: Optional[str] = None
    repo_name: Optional[str] = None

class AskResponse(BaseModel):
    query: str
    answer: str
    sources: List[CodeNode]
    repo_id: Optional[str] = None
    repo_name: Optional[str] = None

# ============================================================================
# Repository State Management
# ============================================================================

class RepoState:
    """State for a single indexed repository"""
    def __init__(self, repo_id: str, repo_name: str, repo_path: str, 
                 nodes: List[Dict], content_hash: str):
        self.repo_id = repo_id
        self.repo_name = repo_name
        self.repo_path = repo_path
        self.nodes = nodes
        self.content_hash = content_hash
        self.collection = None
        self.bm25 = None
        self.bm25_ids: List[str] = []
        self.file_count = 0
        self.indexed_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            "repo_id": self.repo_id,
            "repo_name": self.repo_name,
            "repo_path": self.repo_path,
            "node_count": len(self.nodes),
            "file_count": self.file_count,
            "content_hash": self.content_hash,
            "indexed_at": self.indexed_at
        }


class AppState:
    """Global application state for multi-repo management"""
    def __init__(self):
        self.repos: Dict[str, RepoState] = {}
        self.active_repo_id: Optional[str] = None
        self.index_file = Path("./repo_index.json")
        self.initialized = False
    
    def get_active_repo(self) -> Optional[RepoState]:
        if self.active_repo_id and self.active_repo_id in self.repos:
            return self.repos[self.active_repo_id]
        return None
    
    def save_index(self):
        """Persist repo index to disk"""
        index_data = {
            "active_repo_id": self.active_repo_id,
            "repos": {rid: r.to_dict() for rid, r in self.repos.items()}
        }
        with open(self.index_file, "w") as f:
            json.dump(index_data, f, indent=2)
    
    def load_index(self) -> Optional[Dict]:
        """Load repo index from disk"""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load index: {e}")
        return None


state = AppState()

# ============================================================================
# Repository Indexing Functions
# ============================================================================

def generate_repo_id(repo_path: str) -> str:
    """Generate unique ID from repo path"""
    return hashlib.md5(repo_path.encode()).hexdigest()[:12]


def index_repository(repo_path: str, force: bool = False) -> Dict[str, Any]:
    """Parse and index a repository"""
    repo_path = os.path.abspath(repo_path)
    
    if not os.path.exists(repo_path):
        raise ValueError(f"Path does not exist: {repo_path}")
    
    repo_id = generate_repo_id(repo_path)
    content_hash = get_repo_hash(repo_path)
    repo_name = os.path.basename(repo_path)
    
    print(f"\n{'='*60}")
    print(f"Indexing: {repo_name}")
    print(f"Path: {repo_path}")
    print(f"ID: {repo_id} | Hash: {content_hash[:16]}...")
    print(f"{'='*60}\n")
    
    # Check if already indexed with same content
    if repo_id in state.repos and not force:
        existing = state.repos[repo_id]
        if existing.content_hash == content_hash:
            print(f"✓ Already indexed, no changes detected")
            state.active_repo_id = repo_id
            return {
                "status": "already_indexed",
                "repo_id": repo_id,
                "repo_name": repo_name,
                "node_count": len(existing.nodes),
                "message": "Repository already indexed, no changes detected"
            }
        print(f"! Content changed, re-indexing...")
    
    # Parse repository
    print(f"Parsing Python files...")
    parse_result = parse_repository(repo_path)
    
    if parse_result["total_nodes"] == 0:
        raise ValueError(f"No Python code found in: {repo_path}")
    
    nodes = parse_result["nodes"]
    print(f"✓ Found {len(nodes)} nodes in {parse_result['files_parsed']} files")
    
    # Create ChromaDB collection
    collection_name = f"repo_{repo_id}"
    print(f"Creating collection: {collection_name}")
    
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass
    
    collection = create_collection_from_data(nodes, collection_name=collection_name)
    
    # Create BM25 index
    print(f"Creating BM25 index...")
    bm25, bm25_ids = create_bm25_store(nodes)
    
    # Create repo state
    repo_state = RepoState(
        repo_id=repo_id,
        repo_name=repo_name,
        repo_path=repo_path,
        nodes=nodes,
        content_hash=content_hash
    )
    repo_state.collection = collection
    repo_state.bm25 = bm25
    repo_state.bm25_ids = bm25_ids
    repo_state.file_count = parse_result["files_parsed"]
    
    # Store and activate
    state.repos[repo_id] = repo_state
    state.active_repo_id = repo_id
    state.initialized = True
    state.save_index()
    
    print(f"\n✓ Indexed successfully: {len(nodes)} nodes from {parse_result['files_parsed']} files\n")
    
    return {
        "status": "indexed",
        "repo_id": repo_id,
        "repo_name": repo_name,
        "node_count": len(nodes),
        "file_count": parse_result["files_parsed"],
        "message": "Repository indexed successfully"
    }


def restore_repo(repo_info: Dict) -> bool:
    """Restore a repo from saved index"""
    try:
        repo_id = repo_info["repo_id"]
        repo_path = repo_info["repo_path"]
        is_legacy = repo_id == "legacy"
        
        # Skip legacy repos - they'll be loaded from JSONL fallback
        if is_legacy:
            print(f"  → Skipping (will reload from JSONL)")
            return False
        
        # Check if it's a GitHub repo (URL stored as path)
        is_github = repo_path.startswith("https://github.com/") or repo_path.startswith("git@github.com:")
        
        # For GitHub repos, we can't re-parse (files are deleted after indexing)
        # We rely solely on the ChromaDB collection
        if is_github:
            collection_name = f"repo_{repo_id}"
            try:
                collection = chroma_client.get_collection(name=collection_name)
                if collection.count() == 0:
                    print(f"  ! Collection is empty: {collection_name}")
                    return False
            except Exception:
                print(f"  ! Collection not found: {collection_name}")
                return False
            
            # For GitHub repos, we need to get nodes from collection metadata
            # Since we can't re-parse, create minimal state with collection data
            all_data = collection.get(include=["metadatas", "documents"])
            nodes = []
            for i, meta in enumerate(all_data["metadatas"]):
                # Get source from metadata first (truncated), then from documents if available
                source_code = meta.get("source_code", "")
                nodes.append({
                    "id": all_data["ids"][i],
                    "name": meta.get("name", ""),
                    "node_type": meta.get("type", ""),  # stored as 'type' in metadata
                    "line_no": meta.get("line_no", 0),
                    "end_line_no": meta.get("end_line_no", 0),
                    "node_docstring": "",
                    "source_code": source_code,
                    "args": meta.get("args", "").split(",") if meta.get("args") else []
                })
            
            # Create BM25 from recovered nodes
            bm25, bm25_ids = create_bm25_store(nodes)
            
            repo_state = RepoState(
                repo_id=repo_id,
                repo_name=repo_info["repo_name"],
                repo_path=repo_path,
                nodes=nodes,
                content_hash=repo_info.get("content_hash", "")
            )
            repo_state.collection = collection
            repo_state.bm25 = bm25
            repo_state.bm25_ids = bm25_ids
            repo_state.file_count = repo_info.get("file_count", 0)
            repo_state.indexed_at = repo_info.get("indexed_at", "")
            
            state.repos[repo_id] = repo_state
            return True
        
        if not os.path.exists(repo_path):
            print(f"  ! Path no longer exists: {repo_path}")
            return False
        
        # Check if it's an uploaded repo
        is_uploaded = "uploaded_repos" in repo_path or str(UPLOAD_DIR) in repo_path
        
        if not is_uploaded:
            # Check content hash only for regular indexed repos
            current_hash = get_repo_hash(repo_path)
            if current_hash != repo_info.get("content_hash", ""):
                print(f"  ! Content changed, needs re-indexing")
                return False
        
        # Get existing collection
        collection_name = f"repo_{repo_id}"
        try:
            collection = chroma_client.get_collection(name=collection_name)
            if collection.count() == 0:
                print(f"  ! Collection is empty: {collection_name}")
                return False
        except Exception:
            print(f"  ! Collection not found: {collection_name}")
            return False
        
        # Re-parse for BM25 (not persisted)
        parse_result = parse_repository(repo_path)
        nodes = parse_result["nodes"]
        
        if not nodes:
            print(f"  ! No nodes found in path")
            return False
        
        bm25, bm25_ids = create_bm25_store(nodes)
        
        # Create state
        repo_state = RepoState(
            repo_id=repo_id,
            repo_name=repo_info["repo_name"],
            repo_path=repo_path,
            nodes=nodes,
            content_hash=repo_info.get("content_hash", "")
        )
        repo_state.collection = collection
        repo_state.bm25 = bm25
        repo_state.bm25_ids = bm25_ids
        repo_state.file_count = repo_info.get("file_count", 0)
        repo_state.indexed_at = repo_info.get("indexed_at", "")
        
        state.repos[repo_id] = repo_state
        return True
        
    except Exception as e:
        print(f"  ! Error restoring: {e}")
        return False


def get_repo_for_search(repo_id: Optional[str] = None) -> RepoState:
    """Get repository for search operations"""
    if repo_id:
        if repo_id not in state.repos:
            raise HTTPException(status_code=404, detail=f"Repository not found: {repo_id}")
        return state.repos[repo_id]
    
    active = state.get_active_repo()
    if not active:
        raise HTTPException(
            status_code=400, 
            detail="No repository indexed. Please index a repository first using POST /repos/index"
        )
    return active


# ============================================================================
# Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown"""
    print("\n" + "="*60)
    print("Initializing Codebase Atlas API...")
    print("="*60 + "\n")
    
    # Restore saved repos
    saved = state.load_index()
    if saved:
        print("Restoring indexed repositories...")
        for rid, rinfo in saved.get("repos", {}).items():
            print(f"\n  {rinfo['repo_name']}...")
            if restore_repo(rinfo):
                print(f"  ✓ Restored")
            else:
                print(f"  ✗ Failed")
        
        if saved.get("active_repo_id") in state.repos:
            state.active_repo_id = saved["active_repo_id"]
    
    # Fallback: load legacy JSONL
    if not state.repos:
        jsonl_path = "parsed_python_files.jsonl"
        if os.path.exists(jsonl_path):
            print(f"\nLoading legacy JSONL: {jsonl_path}")
            try:
                nodes = read_jsonl_file(jsonl_path)
                if nodes:
                    repo_id = "legacy"
                    collection = create_collection_from_data(nodes)
                    bm25, bm25_ids = create_bm25_store(nodes)
                    
                    repo_state = RepoState(
                        repo_id=repo_id,
                        repo_name="Legacy JSONL",
                        repo_path=jsonl_path,
                        nodes=nodes,
                        content_hash=""
                    )
                    repo_state.collection = collection
                    repo_state.bm25 = bm25
                    repo_state.bm25_ids = bm25_ids
                    
                    state.repos[repo_id] = repo_state
                    state.active_repo_id = repo_id
                    print(f"✓ Loaded {len(nodes)} nodes")
            except Exception as e:
                print(f"Warning: Could not load JSONL: {e}")
    
    state.initialized = len(state.repos) > 0
    
    print("\n" + "="*60)
    if state.repos:
        print(f"✓ Ready with {len(state.repos)} repository(ies)")
        for r in state.repos.values():
            active = " (active)" if r.repo_id == state.active_repo_id else ""
            print(f"  - {r.repo_name}: {len(r.nodes)} nodes{active}")
    else:
        print("✓ Ready (no repositories indexed)")
    print("="*60 + "\n")
    
    yield
    
    print("\nShutting down...")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Codebase Atlas API",
    description="Multi-repository hybrid code search",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ============================================================================
# Repository Management Endpoints
# ============================================================================

@app.post("/repos/index", tags=["Repositories"])
async def api_index_repo(request: IndexRepoRequest):
    """Index a repository for searching (local path)"""
    try:
        return index_repository(request.repo_path, force=request.force_reindex)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


class GitHubRepoRequest(BaseModel):
    repo_url: str = Field(..., description="GitHub repository URL (e.g., https://github.com/owner/repo)")
    force_reindex: bool = Field(default=False, description="Force re-indexing even if already indexed")


def clone_github_repo(repo_url: str, dest_path: str) -> bool:
    """Clone a GitHub repository to the specified path."""
    try:
        subprocess.run(
            ['git', 'clone', '--depth', '1', repo_url, dest_path],
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Git clone failed: {e.stderr}")
        return False


def extract_repo_name_from_url(repo_url: str) -> str:
    """Extract repository name from GitHub URL."""
    # Handle various URL formats
    # https://github.com/owner/repo.git
    # https://github.com/owner/repo
    # git@github.com:owner/repo.git
    match = re.search(r'[/:]([^/]+)/([^/.]+?)(\.git)?$', repo_url)
    if match:
        return match.group(2)
    return "github-repo"


@app.post("/repos/github", tags=["Repositories"])
async def index_github_repo(request: GitHubRepoRequest):
    """
    Clone and index a GitHub repository.
    The repository is cloned temporarily, indexed, and then the clone is removed.
    """
    repo_url = request.repo_url.strip()
    
    # Validate URL format
    if not re.match(r'^(https://github\.com/|git@github\.com:)[\w.-]+/[\w.-]+(\.git)?$', repo_url):
        raise HTTPException(
            status_code=400, 
            detail="Invalid GitHub URL. Please use format: https://github.com/owner/repo"
        )
    
    # Extract repo name and generate ID
    repo_name = extract_repo_name_from_url(repo_url)
    repo_id = hashlib.md5(repo_url.encode()).hexdigest()[:12]
    
    # Check if already indexed
    if repo_id in state.repos and not request.force_reindex:
        existing = state.repos[repo_id]
        state.active_repo_id = repo_id
        return {
            "status": "already_indexed",
            "repo_id": repo_id,
            "repo_name": existing.repo_name,
            "node_count": len(existing.nodes),
            "message": f"Repository '{repo_name}' already indexed. Enable force re-index to update."
        }
    
    # Create temp directory for cloning
    clone_dir = UPLOAD_DIR / f"github_{repo_id}"
    
    # Clean up if exists
    if clone_dir.exists():
        shutil.rmtree(clone_dir, onerror=force_remove_readonly)
    
    print(f"\n{'='*60}")
    print(f"Cloning GitHub Repository: {repo_name}")
    print(f"URL: {repo_url}")
    print(f"ID: {repo_id}")
    print(f"{'='*60}\n")
    
    # Clone the repository
    print(f"Cloning repository...")
    if not clone_github_repo(repo_url, str(clone_dir)):
        raise HTTPException(
            status_code=400,
            detail="Failed to clone repository. Please check the URL and ensure the repository is public."
        )
    
    print(f"✓ Repository cloned successfully")
    
    try:
        # Index the cloned repository
        result = index_github_repo_internal(str(clone_dir), repo_name, repo_id, repo_url)
        return result
    finally:
        # Always clean up the cloned directory after indexing
        print(f"Cleaning up cloned files...")
        if clone_dir.exists():
            shutil.rmtree(clone_dir, onerror=force_remove_readonly)
        print(f"✓ Cleanup complete")


def index_github_repo_internal(repo_path: str, repo_name: str, repo_id: str, repo_url: str) -> Dict[str, Any]:
    """Index a cloned GitHub repository."""
    from repo_parser import parse_repository
    
    print(f"Parsing repository files...")
    parse_result = parse_repository(repo_path)
    
    if parse_result["total_nodes"] == 0:
        raise ValueError("No Python code nodes found in repository")
    
    nodes = parse_result["nodes"]
    print(f"✓ Found {len(nodes)} nodes in {parse_result['files_parsed']} files")
    
    # Create ChromaDB collection
    collection_name = f"repo_{repo_id}"
    print(f"Creating collection: {collection_name}")
    
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass
    
    collection = create_collection_from_data(nodes, collection_name=collection_name)
    
    # Create BM25 index
    print(f"Creating BM25 index...")
    bm25, bm25_ids = create_bm25_store(nodes)
    
    # Generate content hash
    content_hash = hashlib.md5(
        "|".join(n.get("source_code", "") for n in nodes).encode()
    ).hexdigest()
    
    # Create repo state - store URL as path for reference
    repo_state = RepoState(
        repo_id=repo_id,
        repo_name=repo_name,
        repo_path=repo_url,  # Store GitHub URL as the path
        nodes=nodes,
        content_hash=content_hash
    )
    repo_state.collection = collection
    repo_state.bm25 = bm25
    repo_state.bm25_ids = bm25_ids
    repo_state.file_count = parse_result["files_parsed"]
    
    # Store and activate
    state.repos[repo_id] = repo_state
    state.active_repo_id = repo_id
    state.initialized = True
    state.save_index()
    
    print(f"\n✓ Indexed successfully: {len(nodes)} nodes from {parse_result['files_parsed']} files\n")
    
    return {
        "status": "indexed",
        "repo_id": repo_id,
        "repo_name": repo_name,
        "repo_url": repo_url,
        "node_count": len(nodes),
        "file_count": parse_result["files_parsed"],
        "message": "GitHub repository cloned and indexed successfully"
    }


@app.post("/repos/upload", tags=["Repositories"])
async def upload_repo(
    files: List[UploadFile] = File(...),
    repo_name: str = Form(...),
    force_reindex: bool = Form(False)
):
    """
    Upload a repository folder for indexing.
    Accepts multiple files with their relative paths.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # Generate repo ID from name
    repo_id = hashlib.md5(repo_name.encode()).hexdigest()[:12]
    repo_dir = UPLOAD_DIR / repo_id
    
    # Check if repo already exists
    if repo_id in state.repos and not force_reindex:
        existing = state.repos[repo_id]
        state.active_repo_id = repo_id
        return {
            "status": "already_indexed",
            "repo_id": repo_id,
            "repo_name": existing.repo_name,
            "node_count": len(existing.nodes),
            "message": f"Repository '{repo_name}' already indexed. Enable force re-index to update."
        }
    
    # Clean existing upload if re-indexing
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Uploading Repository: {repo_name}")
    print(f"ID: {repo_id}")
    print(f"Files: {len(files)}")
    print(f"{'='*60}\n")
    
    # Save uploaded files
    python_files = 0
    for file in files:
        # Get relative path from filename (browsers send webkitRelativePath)
        filename = file.filename
        if not filename:
            continue
            
        # Skip non-Python files and common ignored directories
        skip_dirs = {'__pycache__', '.git', '.venv', 'venv', 'env', 'node_modules', '.idea', '.vscode'}
        path_parts = Path(filename).parts
        if any(part in skip_dirs for part in path_parts):
            continue
            
        # Only process Python files
        if not filename.endswith('.py'):
            continue
        
        # Create file path
        file_path = repo_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save file
        try:
            content = await file.read()
            with open(file_path, 'wb') as f:
                f.write(content)
            python_files += 1
        except Exception as e:
            print(f"Error saving {filename}: {e}")
    
    print(f"✓ Saved {python_files} Python files")
    
    if python_files == 0:
        shutil.rmtree(repo_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="No Python files found in upload")
    
    # Now index the uploaded repo
    try:
        result = index_uploaded_repo(str(repo_dir), repo_name, repo_id)
        return result
    except Exception as e:
        shutil.rmtree(repo_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


def index_uploaded_repo(repo_path: str, repo_name: str, repo_id: str) -> Dict[str, Any]:
    """Index an uploaded repository"""
    from repo_parser import parse_repository
    
    print(f"Parsing uploaded files...")
    parse_result = parse_repository(repo_path)
    
    if parse_result["total_nodes"] == 0:
        raise ValueError("No Python code nodes found in uploaded files")
    
    nodes = parse_result["nodes"]
    print(f"✓ Found {len(nodes)} nodes in {parse_result['files_parsed']} files")
    
    # Create ChromaDB collection
    collection_name = f"repo_{repo_id}"
    print(f"Creating collection: {collection_name}")
    
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass
    
    collection = create_collection_from_data(nodes, collection_name=collection_name)
    
    # Create BM25 index
    print(f"Creating BM25 index...")
    bm25, bm25_ids = create_bm25_store(nodes)
    
    # Generate content hash
    content_hash = hashlib.md5(
        "|".join(n.get("source_code", "") for n in nodes).encode()
    ).hexdigest()
    
    # Create repo state
    repo_state = RepoState(
        repo_id=repo_id,
        repo_name=repo_name,
        repo_path=repo_path,
        nodes=nodes,
        content_hash=content_hash
    )
    repo_state.collection = collection
    repo_state.bm25 = bm25
    repo_state.bm25_ids = bm25_ids
    repo_state.file_count = parse_result["files_parsed"]
    
    # Store and activate
    state.repos[repo_id] = repo_state
    state.active_repo_id = repo_id
    state.initialized = True
    state.save_index()
    
    print(f"\n✓ Indexed successfully: {len(nodes)} nodes from {parse_result['files_parsed']} files\n")
    
    return {
        "status": "indexed",
        "repo_id": repo_id,
        "repo_name": repo_name,
        "node_count": len(nodes),
        "file_count": parse_result["files_parsed"],
        "message": "Repository uploaded and indexed successfully"
    }


@app.get("/repos", tags=["Repositories"])
async def list_repos():
    """List all indexed repositories"""
    return {
        "repos": [r.to_dict() for r in state.repos.values()],
        "active_repo_id": state.active_repo_id,
        "count": len(state.repos)
    }


@app.get("/repos/{repo_id}", tags=["Repositories"])
async def get_repo(repo_id: str):
    """Get repository details"""
    if repo_id not in state.repos:
        raise HTTPException(status_code=404, detail="Repository not found")
    return state.repos[repo_id].to_dict()


@app.post("/repos/{repo_id}/activate", tags=["Repositories"])
async def activate_repo(repo_id: str):
    """Set a repository as active for searches"""
    if repo_id not in state.repos:
        raise HTTPException(status_code=404, detail="Repository not found")
    state.active_repo_id = repo_id
    state.save_index()
    return {"status": "ok", "active_repo_id": repo_id}


@app.delete("/repos/{repo_id}", tags=["Repositories"])
async def delete_repo(repo_id: str):
    """Remove a repository from the index"""
    if repo_id not in state.repos:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    try:
        chroma_client.delete_collection(f"repo_{repo_id}")
    except Exception:
        pass
    
    del state.repos[repo_id]
    
    if state.active_repo_id == repo_id:
        state.active_repo_id = next(iter(state.repos.keys()), None)
    
    state.save_index()
    return {"status": "deleted", "repo_id": repo_id}

# ============================================================================
# Search Endpoints
# ============================================================================

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest):
    """Hybrid search combining vector + BM25"""
    repo = get_repo_for_search(request.repo_id)
    
    try:
        results = hybrid_search(
            repo.collection,
            repo.bm25,
            repo.bm25_ids,
            request.query,
            top_n=request.top_n,
            alpha=request.alpha,
            beta=request.beta
        )
        
        code_nodes = []
        for node_id, score in results:
            node = next((n for n in repo.nodes if n["id"] == node_id), None)
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
            total_results=len(code_nodes),
            repo_id=repo.repo_id,
            repo_name=repo.repo_name
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/ask", response_model=AskResponse, tags=["AI"])
async def ask(request: AskRequest):
    """Ask a question about the codebase"""
    repo = get_repo_for_search(request.repo_id)
    
    try:
        results = hybrid_search(
            repo.collection,
            repo.bm25,
            repo.bm25_ids,
            request.query,
            top_n=request.top_n,
            alpha=request.alpha,
            beta=request.beta
        )
        
        if not results:
            return AskResponse(
                query=request.query,
                answer="No relevant code found for your query.",
                sources=[],
                repo_id=repo.repo_id,
                repo_name=repo.repo_name
            )
        
        # Get AI response with full node data
        answer = call_model(request.query, results, repo.nodes)
        
        # Build sources
        sources = []
        for node_id, score in results:
            node = next((n for n in repo.nodes if n["id"] == node_id), None)
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
            sources=sources,
            repo_id=repo.repo_id,
            repo_name=repo.repo_name
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ask failed: {str(e)}")


@app.get("/search/vector", response_model=SearchResponse, tags=["Search"])
async def search_vector_only(
    query: str = Query(..., min_length=1),
    top_n: int = Query(default=5, ge=1, le=20),
    repo_id: Optional[str] = None
):
    """Vector-only search"""
    repo = get_repo_for_search(repo_id)
    
    try:
        results = vector_search(repo.collection, query, top_n=top_n)
        
        code_nodes = []
        for node_id, score in results.items():
            node = next((n for n in repo.nodes if n["id"] == node_id), None)
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
        
        code_nodes.sort(key=lambda x: x.score, reverse=True)
        
        return SearchResponse(
            query=query,
            results=code_nodes,
            total_results=len(code_nodes),
            repo_id=repo.repo_id,
            repo_name=repo.repo_name
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")


@app.get("/search/bm25", response_model=SearchResponse, tags=["Search"])
async def search_bm25_only(
    query: str = Query(..., min_length=1),
    top_n: int = Query(default=5, ge=1, le=20),
    repo_id: Optional[str] = None
):
    """BM25 keyword search only"""
    repo = get_repo_for_search(repo_id)
    
    try:
        results = bm25_search(repo.bm25, repo.bm25_ids, query, top_n=top_n)
        
        code_nodes = []
        for node_id, score in results.items():
            node = next((n for n in repo.nodes if n["id"] == node_id), None)
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
        
        code_nodes.sort(key=lambda x: x.score, reverse=True)
        
        return SearchResponse(
            query=query,
            results=code_nodes,
            total_results=len(code_nodes),
            repo_id=repo.repo_id,
            repo_name=repo.repo_name
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BM25 search failed: {str(e)}")

# ============================================================================
# Utility Endpoints
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Serve the frontend"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Codebase Atlas API", "docs": "/docs"}


@app.get("/health", tags=["Info"])
async def health():
    """Health check"""
    active = state.get_active_repo()
    return {
        "status": "healthy" if state.initialized else "no_repos",
        "initialized": state.initialized,
        "repos_count": len(state.repos),
        "active_repo": active.repo_name if active else None,
        "total_nodes": len(active.nodes) if active else 0
    }


@app.get("/nodes", tags=["Data"])
async def list_nodes(
    file: Optional[str] = None,
    node_type: Optional[str] = None,
    repo_id: Optional[str] = None
):
    """List all indexed nodes"""
    repo = get_repo_for_search(repo_id)
    nodes = repo.nodes
    
    if file:
        nodes = [n for n in nodes if file.lower() in n["id"].split("::")[0].lower()]
    if node_type:
        nodes = [n for n in nodes if n["node_type"].lower() == node_type.lower()]
    
    return {
        "repo_id": repo.repo_id,
        "repo_name": repo.repo_name,
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
async def get_node(node_id: str, repo_id: Optional[str] = None):
    """Get a specific node by ID"""
    repo = get_repo_for_search(repo_id)
    node = next((n for n in repo.nodes if n["id"] == node_id), None)
    
    if not node:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")
    
    return node

# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
