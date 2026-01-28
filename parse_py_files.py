import ast
import os


def extract_imports(tree):
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


def parse_python_file(file_path):
    """
    Parses a Python file and returns its Abstract Syntax Tree (AST).

    Args:
        file_path (str): The path to the Python file to be parsed.
    Returns:
        ast.Module: The AST of the parsed Python file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
    parsed_nodes = []
    parsed_ast = ast.parse(file_content, filename=file_path)
    module_imports = extract_imports(parsed_ast)
    module_docstring = ast.get_docstring(parsed_ast)
    filename = os.path.relpath(file_path, project_folder)
    for node in parsed_ast.body:
        if isinstance(node, ast.FunctionDef):
            id = f"{filename}::{node.name}"
            name = node.name
            imports = module_imports
            type = 'function'
            args = [arg.arg for arg in node.args.args]
            line_no = node.lineno
            end_line_no = node.end_lineno
            node_docstring = ast.get_docstring(node)
            source_code = ast.get_source_segment(file_content, node)
            parsed_nodes.append({
                'id' : id,
                'name': name,
                'type': type,
                'args': args,
                'imports': imports,
                'line_no': line_no,
                'end_line_no': end_line_no,
                'module_docstring': module_docstring,
                'node_docstring': node_docstring,
                'source_code': source_code
            })

        elif isinstance(node, ast.ClassDef):
            name = node.name
            id = f"{filename}::{name}"
            type = 'class'
            base_class = [base.id for base in node.bases if isinstance(base, ast.Name)]
            line_no = node.lineno
            end_line_no = node.end_lineno
            node_docstring = ast.get_docstring(node)    
            parsed_nodes.append({
                'id': id,
                'name': name,
                'type': type,
                'base_class': base_class,
                'imports': module_imports,
                'line_no': line_no,
                'end_line_no': end_line_no,
                'module_docstring': module_docstring,
                'node_docstring': node_docstring,
                'source_code': ast.get_source_segment(file_content, node)
            })


        
            
            

    return parsed_nodes


if __name__ == "__main__":
    project_folder = "C:\\Users\\Abhinand\\OneDrive\\VS Code\\Data-Engineer-Agent-\\agents"

    for file in os.listdir(project_folder):
        if file.endswith(".py"):
            file_path = os.path.join(project_folder, file)
            print("-" * 40)
            print(f"Parsing file: {file_path}")
            ast_tree = parse_python_file(file_path)

            print(f"Nodes in {file}:\n{ast_tree}\n")


