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
    
    # Read JSONL format (multi-line pretty-printed JSON objects)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        content = f.read()
        # Split by closing brace followed by newline and opening brace
        json_objects = content.strip().split('\n}\n{')
        
        for i, obj in enumerate(json_objects):
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
                    data.extend(nodes) 
    return data

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
    bm25_docs = []
    bm25_ids = []
    for node in data:
        bm25_docs.append(tokenize(build_bm25_text(node)))
        bm25_ids.append(node["id"])
    
    bm25 = BM25Okapi(bm25_docs)
    print("BM25 store created.")
    print("Sample BM25 document tokens:", bm25_docs[0][:20])  # Print first 20 tokens of the first document
    return bm25, bm25_ids

def bm25_search(bm25, bm25_ids, query, top_n=2):
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    print("BM25 Scores:", scores)  # Debug: Print all scores

    top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]

    results = [(bm25_ids[i], scores[i]) for i in top_n_indices if scores[i] > 0]

    return results  

def create_collection_from_data(data):
    documents = []
    metadata = []
    
    for node in data:
        text = build_embedding_text(node)
        documents.append(text)
        metadata.append({
        "name": node["name"],
        "type": node["node_type"],
        "file": node['id'].split('::')[0],
        "line_no": node["line_no"],
        "end_line_no": node["end_line_no"],
        "imports": ",".join(node["imports"]),
        "args": ",".join(node.get("args", [])),
        "source_code": node["source_code"][:1000]  # Truncate source code for metadata
    })



    collection  = chroma_client.get_or_create_collection(name="code_atlas", embedding_function=embedding)

    collection.add(documents=documents, metadatas=metadata, ids = [node["id"] for node in data]
)

    query = "Where can I find function to upload to GCS?"

    results = collection.query(query_texts=[query], n_results=2)

    print("Query Results:")
    for i, metadata in enumerate(results["metadatas"][0]):
        print(f"\n--- Result {i+1} ---")
        print(metadata)

    response = model.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful programming assistant."},
            {"role": "user", "content": f"{query}:\n\n{results['documents'][0][0]}\n\n{results['documents'][0][1]}\n\nProvide the function name and its file location."}
        ],
        max_tokens=300,
    )
    print("\nModel Response:")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    data = read_jsonl_file("parsed_python_files.jsonl")
    create_collection_from_data(data)
    bm25, bm25_ids = create_bm25_store(data)
    # Example BM25 query
    query = "bigquery_assistant.py"
    # results = bm25_search(bm25, bm25_ids, query, top_n=2)
    # print(f"\nBM25 Search Results for '{query}':")
    # for node_id, score in results:
    #     print(f"ID: {node_id}, Score: {score}")

    