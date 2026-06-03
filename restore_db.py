"""
restore_chromadb.py — rebuild ChromaDB metadata from SQLite
Run this if ChromaDB metadata got overwritten and lost name/artist/features.
Embeddings are preserved — only metadata is restored.
"""
import json, pickle
import numpy as np
from pathlib import Path
from utils.db import get_conn

DATA_DIR   = Path("data")
CHROMA_DIR = str(DATA_DIR / "chromadb")

FEATURE_COLS = [
    "tempo", "energy_mean", "energy_std",
    "valence_proxy", "danceability_proxy", "acousticness_proxy",
    "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
    "spectral_contrast", "zcr", "chroma_mean", "chroma_std",
    *[f"mfcc_{i}" for i in range(1, 14)],
]

print("📂 Loading from SQLite...")
conn = get_conn()
rows = conn.execute("""
    SELECT s.id, s.name, s.artist, s.album, s.year,
           f.*,
           sm.cluster_id, sm.mood_scores
    FROM features f
    JOIN songs s ON s.id = f.song_id
    LEFT JOIN song_moods sm ON sm.song_id = s.id
    ORDER BY s.id
""").fetchall()
conn.close()
print(f"  ✅ {len(rows)} songs loaded")

print("\n🗄️  Rebuilding ChromaDB metadata...")
import chromadb
client     = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection("songs")

# Load cluster labels
clusters_path = DATA_DIR / "clusters.json"
cluster_labels = {}
if clusters_path.exists():
    with open(clusters_path) as f:
        cluster_labels = {int(k): v["label"] for k, v in json.load(f).items()}

batch_size = 500
total      = len(rows)

for start in range(0, total, batch_size):
    end        = min(start + batch_size, total)
    batch_rows = rows[start:end]

    ids      = [r["id"] for r in batch_rows]
    metadatas = []

    for r in batch_rows:
        r = dict(r)
        meta = {
            "name":          r["name"],
            "artist":        r["artist"],
            "album":         r.get("album", "") or "",
            "year":          str(r.get("year", "") or ""),
            "features_json": json.dumps({k: float(r[k]) for k in FEATURE_COLS}),
            **{k: float(r[k]) for k in FEATURE_COLS},
        }
        # add mood data if available
        if r.get("cluster_id") is not None:
            meta["cluster_id"]    = int(r["cluster_id"])
            meta["cluster_label"] = cluster_labels.get(int(r["cluster_id"]), "")
            meta["mood_scores"]   = r["mood_scores"] or "{}"
        metadatas.append(meta)

    collection.update(ids=ids, metadatas=metadatas)
    print(f"  📦 {end}/{total}")

print(f"\n✅ Done. ChromaDB metadata restored for {total} songs.")

# Sanity check
sample  = collection.get(limit=1, include=["metadatas"])
meta    = sample["metadatas"][0]
print(f"\nSample: {meta.get('name')} — {meta.get('artist')}")
print(f"  cluster_label: {meta.get('cluster_label')}")
print(f"  has features:  {'tempo' in meta}")