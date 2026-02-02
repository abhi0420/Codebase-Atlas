import chromadb
from chromadb.utils import embedding_functions
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import re


load_dotenv()

model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

embedding = embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-3-small")

chroma_client = chromadb.PersistentClient(path="./chroma_db")

def build_embedding_text(node):
    parts = [
        f"Name: {node['name']}",
        f"Type: {node['node_type']}",
        f"File: {node['id'].split('::')[0]}"
        ]
    if node["module_docstring"]:
        parts.append(f"Docstring: {node['module_docstring']}")

    if node["node_docstring"]:
        parts.append(f"Node Docstring: {node['node_docstring']}")

    parts.append(f"Source Code: {node['source_code']}")

    return "\n".join(parts)



def read_jsonl_file(jsonl_path):   
    data = []
    
    try:
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
        
        print(f"Reading JSONL file: {jsonl_path}")
        
        # Read JSONL format (multi-line pretty-printed JSON objects)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        if not content.strip():
            raise ValueError("JSONL file is empty")
            
        # Split by closing brace followed by newline and opening brace
        json_objects = content.strip().split('\n}\n{')
        print(f"Found {len(json_objects)} JSON objects")
        
        for i, obj in enumerate(json_objects):
            try:
                # Add back the braces that were removed during split
                if i > 0:
                    obj = '{' + obj
                if i < len(json_objects) - 1:
                    obj = obj + '\n}'
                
                obj = obj.strip()
                if obj:
                    file_data = json.loads(obj)
                    # Each object contains a dict with filename as key and list of nodes as value
                    for filename, nodes in file_data.items():
                        if not isinstance(nodes, list):
                            print(f"Warning: Expected list for {filename}, got {type(nodes)}")
                            continue
                        data.extend(nodes)
                        print(f"Loaded {len(nodes)} nodes from {filename}")
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON object {i}: {e}")
                continue
                
        print(f"Total nodes loaded: {len(data)}")
        return data
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        raise
    except Exception as e:
        print(f"ERROR reading JSONL file: {e}")
        raise

def tokenize(text):
     return re.findall(r"[A-Za-z_\.]+", text.lower())

def build_bm25_text(node):
    parts = [
        node['name'],
        node['node_type'],
        node['id'].split('::')[0],
        " ".join(node.get('imports', [])),
        " ".join(node.get('args', [])),
        ]

    return " ".join(filter(None, parts))

def create_bm25_store(data):
    """
    Creates a BM25 index from parsed code nodes.
    """
    try:
        if not data:
            raise ValueError("Cannot create BM25 store: data is empty")
        
        print(f"Creating BM25 store from {len(data)} nodes...")
        bm25_docs = []
        bm25_ids = []
        
        for node in data:
            if "id" not in node:
                print(f"Warning: Node missing 'id' field: {node.get('name', 'unknown')}")
                continue
            bm25_docs.append(tokenize(build_bm25_text(node)))
            bm25_ids.append(node["id"])
        
        bm25 = BM25Okapi(bm25_docs)
        print(f"✓ BM25 store created with {len(bm25_docs)} documents")
        if bm25_docs:
            print(f"  Sample tokens: {bm25_docs[0][:20]}")
        return bm25, bm25_ids
        
    except Exception as e:
        print(f"ERROR creating BM25 store: {e}")
        raise

def bm25_search(bm25, bm25_ids, query, top_n=2):
    """
    Performs a BM25 search.
    Args:
        bm25 (BM25Okapi): The BM25 index.
        bm25_ids (list): List of document IDs corresponding to the BM25 index.
        query (str): The search query.
        top_n (int): Number of top results to return.

    Returns:
        dict: Mapping of IDs to BM25 scores.

    """
    try:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
            
        query_tokens = tokenize(query)
        print(f"BM25 query tokens: {query_tokens}")
        
        scores = bm25.get_scores(query_tokens)
        print(f"BM25 score range: [{min(scores):.4f}, {max(scores):.4f}]")

        top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]

        results = {
            bm25_ids[i]: scores[i]
            for i in top_n_indices
            if scores[i] > 0
        }
        
        print(f"✓ BM25 search returned {len(results)} results")
        return results
        
    except Exception as e:
        print(f"ERROR in BM25 search: {e}")
        return {}  

def create_collection_from_data(data, collection_name: str = "code_atlas"):
    """
    Creates a ChromaDB collection from parsed code nodes.
    Args:
        data (list): List of parsed code nodes.
        collection_name (str): Name for the collection (default: "code_atlas")
    Returns:
        chroma.Collection: The created ChromaDB collection.
    """

    try:
        if not data:
            raise ValueError("Cannot create collection: data is empty")
            
        print(f"Creating ChromaDB collection from {len(data)} nodes...")
        documents = []
        metadata = []
        ids = []
        
        for i, node in enumerate(data):
            try:
                # Validate required fields
                required_fields = ["id", "name", "node_type", "line_no", "end_line_no", "source_code"]
                missing = [f for f in required_fields if f not in node]
                if missing:
                    print(f"Warning: Node {i} missing fields: {missing}, skipping")
                    continue
                
                text = build_embedding_text(node)
                documents.append(text)
                ids.append(node["id"])
                metadata.append({
                    "name": node["name"],
                    "type": node["node_type"],
                    "file": node['id'].split('::')[0],
                    "line_no": node["line_no"],
                    "end_line_no": node["end_line_no"],
                    "imports": ",".join(node.get("imports", [])),
                    "args": ",".join(node.get("args", [])),
                    "source_code": node["source_code"][:1000]  # Truncate source code for metadata
                })
            except Exception as e:
                print(f"Warning: Error processing node {i} ({node.get('name', 'unknown')}): {e}")
                continue

        if not documents:
            raise ValueError("No valid documents to add to collection")

        print(f"Prepared {len(documents)} documents for indexing")
        collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=embedding)
        
        # Check existing count
        existing_count = collection.count()
        print(f"Existing collection has {existing_count} documents")
        
        collection.add(documents=documents, metadatas=metadata, ids=ids)
        new_count = collection.count()
        print(f"✓ Collection created/updated: {new_count} total documents")

        return collection
        
    except Exception as e:
        print(f"ERROR creating collection: {e}")
        raise

def vector_search(collection, query, top_n=2):
    """
    Performs a vector search on the ChromaDB collection.
    Args:
        collection (chroma.Collection): The ChromaDB collection to search.
        query (str): The search query.
        top_n (int): Number of top results to return.
    Returns:
        dict: Mapping of IDs to similarity scores.
    """

    try:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
            
        results = collection.query(query_texts=[query], n_results=top_n, include=['distances'])
        
        if not results["ids"] or not results["ids"][0]:
            print("Warning: Vector search returned no results")
            return {}
        
        scores = {
            results["ids"][0][i]: 1 - results["distances"][0][i]
            for i in range(len(results["ids"][0]))
        }
        
        print(f"✓ Vector search returned {len(scores)} results")
        return scores
        
    except Exception as e:
        print(f"ERROR in vector search: {e}")
        return {}

def normalise_scores(scores):
    """Normalises scores to [0, 1] range using min-max scaling.
    Args: 
        scores (dict): Mapping of IDs to scores.

    Returns:    
        dict: Mapping of IDs to normalised scores.

    """
    if not scores:
        return {}
    
    values = list(scores.values())

    min_score = min(values)
    max_score = max(values)

    if min_score == max_score:
        return {k: 1.0 for k in scores.keys()}
    
    return {
        k: (v - min_score) / (max_score - min_score)
        for k, v in scores.items()
    }


def hybrid_search(collection, bm25, bm25_ids, query, top_n=2, alpha=0.7, beta=0.3):
    """
    Performs a hybrid search combining vector search and BM25.
    Args:
        collection (chroma.Collection): The ChromaDB collection for vector search.
        bm25 (BM25Okapi): The BM25 index.
        bm25_ids (list): List of document IDs corresponding to the BM25 index.
        query (str): The search query.
        top_n (int): Number of top results to return.
        alpha (float): Weight for vector search scores.
        beta (float): Weight for BM25 scores.
    """

    try:
        if abs(alpha + beta - 1.0) > 0.01:
            print(f"Warning: alpha + beta = {alpha + beta}, should equal 1.0")
        
        print(f"\n{'='*60}")
        print(f"Hybrid Search: '{query}'")
        print(f"Weights: Vector={alpha}, BM25={beta}, Top N={top_n}")
        print(f"{'='*60}")

        vector_results = vector_search(collection, query, top_n=top_n*2)  # Get more for fusion
        bm25_results = bm25_search(bm25, bm25_ids, query, top_n=top_n*2)

        vector_scores = normalise_scores(vector_results)
        bm25_scores = normalise_scores(bm25_results)

        all_ids = set(vector_scores) | set(bm25_scores)
        print(f"Unique results to fuse: {len(all_ids)}")

        fused = []
        for node_id in all_ids:
            v_score = vector_scores.get(node_id, 0)
            b_score = bm25_scores.get(node_id, 0)
            final_score = (alpha * v_score) + (beta * b_score)
            fused.append((node_id, final_score, v_score, b_score))

        fused.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n{'─'*60}")
        print("Top Results:")
        for i, (node_id, final, vec, bm25) in enumerate(fused[:top_n], 1):
            print(f"{i}. {node_id}")
            print(f"   Final: {final:.4f} (Vector: {vec:.4f}, BM25: {bm25:.4f})")
        print(f"{'─'*60}\n")
        
        return [(node_id, score) for node_id, score, _, _ in fused[:top_n]]
        
    except Exception as e:
        print(f"ERROR in hybrid search: {e}")
        return []

def call_model(query, search_results, data):
    """
    Calls the language model with the query and search results.
    Args:
        query (str): The user query.
        search_results (list): List of tuples (node_id, score).
        data (list): Full list of parsed code nodes.
    Returns:
        str: The model's response.
    """
    try:
        if not search_results:
            return "No relevant code nodes found."

        # Build rich context from top results
        context_parts = []
        for i, (node_id, score) in enumerate(search_results, 1):
            node = next((n for n in data if n["id"] == node_id), None)
            if node:
                context_parts.append(f"""
[Result {i}] {node['name']} (Relevance Score: {score:.4f})
File: {node['id'].split('::')[0]}
Lines: {node['line_no']}-{node['end_line_no']}
Type: {node['node_type']}
Arguments: {', '.join(node.get('args', []))}
Docstring: {node.get('node_docstring', 'No documentation')}

Source Code:
{node['source_code']}
""")
        
        context = "\n" + "="*80 + "\n".join(context_parts)

        prompt = f"""
You are a helpful code assistant. Analyze the following code snippets from a codebase and answer the user's query.

CODE CONTEXT:
{context}

USER QUERY: {query}

Provide a clear, concise answer based on the code above. Include function names, file locations, and line numbers when relevant."""
        
        response = model.chat.completions.create(
            model="gpt-4o", 
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.2
        )
        return response.choices[0].message.content 
    except Exception as e:
        print(f"ERROR calling model: {e}")
        return "Error generating response from model."


if __name__ == "__main__":
    try:
        print("\n" + "="*80)
        print("CODEBASE ATLAS - Hybrid Search System")
        print("="*80 + "\n")
        
        # Load data
        data = read_jsonl_file("parsed_python_files.jsonl")
        if not data:
            print("ERROR: No data loaded, exiting")
            exit(1)
        
        # Create stores
        collection = create_collection_from_data(data)
        bm25, bm25_ids = create_bm25_store(data)
        
        # Example query
        query = "Is there any funtion related to creation of stored procedures?"
        search_results = hybrid_search(collection, bm25, bm25_ids, query, top_n=2, alpha=0.6, beta=0.4)

        if not search_results:
            print("No search results found")
        else:
            print("\n" + "="*80)
            print("FINAL RESULTS")
            print("="*80)
            for i, (node_id, score) in enumerate(search_results, 1):
                node = next((n for n in data if n["id"] == node_id), None)
                if node:
                    print(f"\n[{i}] {node['name']} (Score: {score:.4f})")
                    print(f"    File: {node['id'].split('::')[0]}:{node['line_no']}-{node['end_line_no']}")
                    print(f"    Type: {node['node_type']}")
                    if node.get('node_docstring'):
                        print(f"    Doc: {node['node_docstring'][:100]}...")

                    print(f"    Source Code:\n{node['source_code'][:300]}...\n")
                    print("Line nos:", node['line_no'], "-", node['end_line_no'])
                else:
                    print(f"\n[{i}] {node_id} (Score: {score:.4f}) - Node not found in data")
        
        print("\n" + "="*80)
        print("✓ Search complete")
        print("="*80 + "\n")

        query_response = call_model(query, search_results, data)
        print("MODEL RESPONSE:")
        print("-" * 40)
        print(query_response)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

