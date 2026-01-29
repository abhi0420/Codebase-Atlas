import chromadb
from chromadb.utils import embedding_functions
import os
import json

from dotenv import load_dotenv

load_dotenv()
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






def create_collection_from_json(json_path):
    documents = []
    metadata = []
    data = []
    
    # Read JSONL format (multi-line pretty-printed JSON objects)
    with open(json_path, "r", encoding="utf-8") as f:
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
        "args": ",".join(node.get("args", []))
    })




    collection  = chroma_client.get_or_create_collection(name="code_atlas", embedding_function=embedding)

    collection.add(documents=documents, metadatas=metadata, ids = [node["id"] for node in data]
)

    query = "Where can I find function to upload to GCS?"

    results = collection.query(query_texts=[query], n_results=3)

    print("Query Results:")
    print(results["documents"][0][0])

if __name__ == "__main__":
    create_collection_from_json("parsed_python_files.jsonl")


