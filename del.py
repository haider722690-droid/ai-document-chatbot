# delete_chromadb.py
import shutil
import os

chroma_path = "./chroma_db"
if os.path.exists(chroma_path):
    shutil.rmtree(chroma_path)
    print(f"Deleted {chroma_path}")
else:
    print("ChromaDB folder not found")