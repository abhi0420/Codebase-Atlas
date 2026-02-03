import subprocess

def clone_repo(repo_url : str, dest_path : str) -> bool:
    """
    Parses a GitHub repository URL and returns the owner and repository name.

    Args:
        repo_url (str): The GitHub repository URL.
    Returns:
        bool: True if the repository was cloned successfully, False otherwise.
    """
    try:
        subprocess.run(['git', 'clone', repo_url, dest_path], check=True)
        return True
    except subprocess.CalledProcessError:
        return False
    

if __name__ == "__main__":
    repo_url = "https://github.com/minimice/quaizr.git"
    dest_path = "./quaizr"
    success = clone_repo(repo_url, dest_path)
    if success:
        print(f"Repository cloned successfully to {dest_path}")
    else:
        print("Failed to clone the repository.")
        
