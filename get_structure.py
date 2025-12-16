# Script to run locally (no file path provided so you can save/run where you want)
import os, json

def dir_to_dict(path):
    result = {}
    try:
        with os.scandir(path) as it:
            for entry in sorted(it, key=lambda e: e.name.lower()):
                if entry.is_dir() and entry.name not in (".git", "__pycache__", ".vscode",".pytest_cache"):
                    result[entry.name] = dir_to_dict(entry.path)
                else:
                    result[entry.name] = None  # or use {} or file metadata if you prefer
    except PermissionError:
        result["__error__"] = "PermissionError"
    return result

if __name__ == "__main__":
    root = r"c:\Users\MARIE\Desktop\PROJECTS\skillhub"
    tree_dict = {os.path.basename(root.rstrip("\\/")): dir_to_dict(root)}
    print(json.dumps(tree_dict, indent=2, ensure_ascii=False))