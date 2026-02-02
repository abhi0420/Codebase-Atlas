"""
Repository Parser and Indexer
Parses Python files from a repository and creates searchable indices.
"""

import ast
import os
import json
import hashlib
from typing import List, Dict, Optional
from pathlib import Path


def extract_imports(tree) -> List[str]:
    """Extract all imports from an AST tree."""
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.add(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for n in node.names:
                    imports.add(f"{node.module}.{n.name}")
    return list(imports)


def parse_python_file(file_path: str, project_folder: str) -> List[Dict]:
    """
    Parse a Python file and extract functions and classes.
    
    Args:
        file_path: Absolute path to the Python file
        project_folder: Root folder of the project (for relative paths)
    
    Returns:
        List of parsed node dictionaries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    try:
        parsed_ast = ast.parse(file_content, filename=file_path)
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}")
        return []
    
    parsed_nodes = []
    module_imports = extract_imports(parsed_ast)
    module_docstring = ast.get_docstring(parsed_ast)
    
    # Get relative filename
    try:
        filename = os.path.relpath(file_path, project_folder)
    except ValueError:
        filename = os.path.basename(file_path)
    
    for node in parsed_ast.body:
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            parsed_nodes.append({
                'id': f"{filename}::{node.name}",
                'name': node.name,
                'node_type': 'function',
                'args': [arg.arg for arg in node.args.args],
                'imports': module_imports,
                'line_no': node.lineno,
                'end_line_no': node.end_lineno,
                'module_docstring': module_docstring,
                'node_docstring': ast.get_docstring(node),
                'source_code': ast.get_source_segment(file_content, node) or ""
            })
        
        elif isinstance(node, ast.ClassDef):
            base_classes = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    base_classes.append(base.id)
                elif isinstance(base, ast.Attribute):
                    base_classes.append(f"{base.value.id}.{base.attr}" if hasattr(base.value, 'id') else base.attr)
            
            parsed_nodes.append({
                'id': f"{filename}::{node.name}",
                'name': node.name,
                'node_type': 'class',
                'base_class': base_classes,
                'args': [],
                'imports': module_imports,
                'line_no': node.lineno,
                'end_line_no': node.end_lineno,
                'module_docstring': module_docstring,
                'node_docstring': ast.get_docstring(node),
                'source_code': ast.get_source_segment(file_content, node) or ""
            })
            
            # Also parse methods within the class
            for item in node.body:
                if isinstance(item, ast.FunctionDef) or isinstance(item, ast.AsyncFunctionDef):
                    parsed_nodes.append({
                        'id': f"{filename}::{node.name}.{item.name}",
                        'name': f"{node.name}.{item.name}",
                        'node_type': 'method',
                        'args': [arg.arg for arg in item.args.args],
                        'imports': module_imports,
                        'line_no': item.lineno,
                        'end_line_no': item.end_lineno,
                        'module_docstring': module_docstring,
                        'node_docstring': ast.get_docstring(item),
                        'source_code': ast.get_source_segment(file_content, item) or ""
                    })
    
    return parsed_nodes


def parse_repository(repo_path: str, recursive: bool = True) -> Dict:
    """
    Parse all Python files in a repository.
    
    Args:
        repo_path: Path to the repository root
        recursive: Whether to search subdirectories
    
    Returns:
        Dictionary with repo info and parsed nodes
    """
    repo_path = os.path.abspath(repo_path)
    
    if not os.path.exists(repo_path):
        raise FileNotFoundError(f"Repository path not found: {repo_path}")
    
    # Collect all Python files
    python_files = []
    
    if recursive:
        for root, dirs, files in os.walk(repo_path):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if d not in {
                '__pycache__', '.git', '.venv', 'venv', 'env',
                'node_modules', '.tox', '.eggs', '*.egg-info',
                'build', 'dist', '.pytest_cache', '.mypy_cache'
            }]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(repo_path):
            if file.endswith('.py'):
                python_files.append(os.path.join(repo_path, file))
    
    # Parse all files
    all_nodes = []
    files_parsed = 0
    files_failed = 0
    
    for file_path in python_files:
        nodes = parse_python_file(file_path, repo_path)
        if nodes:
            all_nodes.extend(nodes)
            files_parsed += 1
        else:
            files_failed += 1
    
    # Generate repo ID from path
    repo_id = hashlib.md5(repo_path.encode()).hexdigest()[:12]
    repo_name = os.path.basename(repo_path)
    
    return {
        'repo_id': repo_id,
        'repo_name': repo_name,
        'repo_path': repo_path,
        'files_parsed': files_parsed,
        'files_failed': files_failed,
        'total_nodes': len(all_nodes),
        'nodes': all_nodes
    }


def get_repo_hash(repo_path: str) -> str:
    """Generate a hash based on repo path and file modification times."""
    repo_path = os.path.abspath(repo_path)
    
    # Collect modification times of all Python files
    mod_times = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in {'__pycache__', '.git', '.venv', 'venv', 'env', 'node_modules'}]
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                mod_times.append(f"{file_path}:{os.path.getmtime(file_path)}")
    
    # Create hash from path + all modification times
    content = repo_path + "|" + "|".join(sorted(mod_times))
    return hashlib.md5(content.encode()).hexdigest()


if __name__ == "__main__":
    # Test parsing
    import sys
    
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
    else:
        repo_path = "."
    
    print(f"Parsing repository: {repo_path}")
    result = parse_repository(repo_path)
    
    print(f"\nRepository: {result['repo_name']}")
    print(f"Files parsed: {result['files_parsed']}")
    print(f"Files failed: {result['files_failed']}")
    print(f"Total nodes: {result['total_nodes']}")
    
    # Show first few nodes
    for node in result['nodes'][:5]:
        print(f"\n- {node['name']} ({node['node_type']}) in {node['id'].split('::')[0]}")
