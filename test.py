import chromadb
import numpy as np

client = chromadb.PersistentClient(path="data/chromadb")
collection = client.get_collection("songs")

batch = collection.get(limit=500, include=["embeddings"])
emb = np.array(batch["embeddings"], dtype=np.float32)

print("mean:", emb.mean())
print("std:", emb.std())
print("min:", emb.min(), "max:", emb.max())
print("per-dim std:", emb.std(axis=0).mean())