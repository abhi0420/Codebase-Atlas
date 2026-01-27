import ast
import os

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
    parsed_functions = []
    parsed_ast = ast.parse(file_content, filename=file_path)
    for node in ast.walk(parsed_ast):
        if isinstance(node, ast.FunctionDef):
            parsed_functions.append(node.body)

    return parsed_functions


if __name__ == "__main__":
    project_folder = "C:\\Users\\Abhinand\\OneDrive\\VS Code\\Data-Engineer-Agent-\\agents"

    for file in os.listdir(project_folder):
        if file.endswith(".py"):
            file_path = os.path.join(project_folder, file)
            print("-" * 40)
            print(f"Parsing file: {file_path}")
            ast_tree = parse_python_file(file_path)
            print(f"Functions in {file}:\n{ast_tree}\n")


