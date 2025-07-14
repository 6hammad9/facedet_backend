import sys
import pickle
import os
from face2025old import getEmbed

if len(sys.argv) < 3:
    print("Usage: script.py <name> <image_path>")
    sys.exit(1)

name = sys.argv[1]
image_path = sys.argv[2]

pkl_path = r"D:\New folder\OCR\users.pkl"

os.makedirs(os.path.dirname(pkl_path), exist_ok=True)

try:
    with open(pkl_path, "rb") as f:
        users = pickle.load(f)
    print("[INFO] Loaded existing users.pkl")
except FileNotFoundError:
    users = {}
    print("[INFO] users.pkl not found. Creating new one.")

try:
    embedding = getEmbed(image_path)
    if embedding is None:
        raise ValueError("Embedding is None. Check getEmbed function.")
    users[name] = embedding
    with open(pkl_path, "wb") as f:
        pickle.dump(users, f)
    print(f"[SUCCESS] Embedding for {name} updated.")
except Exception as e:
    print(f"[ERROR] Failed to get embedding: {e}")
    sys.exit(1)
