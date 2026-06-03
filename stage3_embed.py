"""
stage3_embed.py — Stage 3: UMAP Embeddings + ChromaDB
 
Pipeline:
  1. Load all 8870 feature vectors from SQLite
  2. Normalize with StandardScaler (zero mean, unit variance)
  3. Fit UMAP: 26-dim → 64-dim
  4. Save scaler + UMAP model to disk (needed at query time)
  5. Push all songs + 64-dim embeddings into ChromaDB
 
Usage:
    python stage3_embed.py
    python stage3_embed.py --chroma-only   # skip UMAP refit, just reload ChromaDB
"""
 
import argparse
import json
import os
import pickle
from pathlib import Path
 
import numpy as np
from utils.db import get_conn
 
DATA_DIR    = Path("data")
SCALER_PATH = DATA_DIR / "scaler.pkl"
UMAP_PATH   = DATA_DIR / "umap_model.pkl"
CHROMA_DIR  = str(DATA_DIR / "chromadb")
 
UMAP_COMPONENTS = 64
UMAP_NEIGHBORS  = 15    # higher = more global structure preserved
UMAP_MIN_DIST   = 0.1   # lower = tighter clusters
 
FEATURE_COLS = [
    "tempo", "energy_mean", "energy_std",
    "valence_proxy", "danceability_proxy", "acousticness_proxy",
    "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
    "spectral_contrast", "zcr", "chroma_mean", "chroma_std",
    *[f"mfcc_{i}" for i in range(1, 14)],
]
 
 
# ── Data loading ──────────────────────────────────────────────────────────────
 
def load_data() -> tuple[list[str], list[dict], np.ndarray]:
    """
    Returns:
        ids      — list of song_ids
        metadata — list of dicts (name, artist, + all raw features)
        X        — (N, 26) float32 numpy array
    """
    print("📂 Loading features from SQLite...")
    conn = get_conn()
    rows = conn.execute("""
        SELECT s.id, s.name, s.artist, s.album, s.year,
               f.tempo, f.energy_mean, f.energy_std,
               f.valence_proxy, f.danceability_proxy, f.acousticness_proxy,
               f.spectral_centroid, f.spectral_bandwidth, f.spectral_rolloff,
               f.spectral_contrast, f.zcr, f.chroma_mean, f.chroma_std,
               f.mfcc_1,  f.mfcc_2,  f.mfcc_3,  f.mfcc_4,  f.mfcc_5,
               f.mfcc_6,  f.mfcc_7,  f.mfcc_8,  f.mfcc_9,  f.mfcc_10,
               f.mfcc_11, f.mfcc_12, f.mfcc_13
        FROM features f
        JOIN songs s ON s.id = f.song_id
        ORDER BY f.song_id
    """).fetchall()
    conn.close()
 
    ids, metadata, vectors = [], [], []
    for row in rows:
        r = dict(row)
        ids.append(r["id"])
        metadata.append({
            "name":   r["name"],
            "artist": r["artist"],
            "album":  r.get("album", ""),
            "year":   str(r.get("year", "")),
            # raw features stored as metadata for later use
            **{k: float(r[k]) for k in FEATURE_COLS},
        })
        vectors.append([float(r[k]) for k in FEATURE_COLS])
 
    X = np.array(vectors, dtype=np.float32)
    print(f"  ✅ Loaded {len(ids)} songs, feature matrix: {X.shape}")
    return ids, metadata, X
 
 
# ── Normalization ─────────────────────────────────────────────────────────────
 
def fit_scaler(X: np.ndarray):
    from sklearn.preprocessing import StandardScaler
    print("\n⚖️  Fitting StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  ✅ Scaler saved → {SCALER_PATH}")
    return X_scaled
 
 
def load_scaler(X: np.ndarray) -> np.ndarray:
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return scaler.transform(X)
 
 
# ── UMAP ──────────────────────────────────────────────────────────────────────
 
def fit_umap(X_scaled: np.ndarray) -> np.ndarray:
    import umap
    print(f"\n🗺️  Fitting UMAP: {X_scaled.shape[1]}-dim → {UMAP_COMPONENTS}-dim")
    print(f"   n_neighbors={UMAP_NEIGHBORS}, min_dist={UMAP_MIN_DIST}")
    print(f"   This takes 2-5 minutes for 8k songs...")
 
    reducer = umap.UMAP(
        n_components=UMAP_COMPONENTS,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="euclidean",
        random_state=42,
        verbose=True,
    )
    embeddings = reducer.fit_transform(X_scaled)
 
    with open(UMAP_PATH, "wb") as f:
        pickle.dump(reducer, f)
    print(f"  ✅ UMAP model saved → {UMAP_PATH}")
    print(f"  ✅ Embeddings shape: {embeddings.shape}")
    return embeddings 
    
 
def load_umap_embeddings(X_scaled: np.ndarray) -> np.ndarray:
    with open(UMAP_PATH, "rb") as f:
        reducer = pickle.load(f)
    return reducer.transform(X_scaled).astype(np.float32)
 
 
# ── ChromaDB ──────────────────────────────────────────────────────────────────
 
def push_to_chromadb(
    ids: list[str],
    metadata: list[dict],
    embeddings: np.ndarray,
    batch_size: int = 500,
):
    import chromadb
    print(f"\n🗄️  Pushing {len(ids)} songs to ChromaDB...")
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
 
    # Drop and recreate so re-runs are idempotent
    try:
        client.delete_collection("songs")
        print("  ♻️  Dropped existing 'songs' collection")
    except Exception:
        pass
 
    collection = client.create_collection(
        name="songs",
        metadata={"hnsw:space": "l2"},   # L2 distance for queries
    )
 
    # ChromaDB metadata values must be str/int/float — no nested dicts
    # Serialize raw feature dict to JSON string for storage
    safe_meta = []
    for m in metadata:
        entry = {k: v for k, v in m.items() if isinstance(v, (str, int, float))}
        # Store full feature vector as JSON string for easy retrieval
        entry["features_json"] = json.dumps(
            {k: m[k] for k in FEATURE_COLS}
        )
        safe_meta.append(entry)
 
    # Batch insert
    total = len(ids)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=safe_meta[start:end],
        )
        print(f"  📦 Inserted {end}/{total}")
 
    count = collection.count()
    print(f"  ✅ ChromaDB ready — {count} songs in 'songs' collection")
    print(f"  📁 Persisted at: {CHROMA_DIR}")
 
 
# ── Sanity check ──────────────────────────────────────────────────────────────
 
def sanity_check():
    """Query ChromaDB for a random song and print its nearest neighbours."""
    import chromadb
    print("\n🔍 Sanity check — nearest neighbours for a random song...")
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection("songs")
 
    # Grab first song
    sample = collection.get(limit=1, include=["embeddings", "metadatas"])
    song_id   = sample["ids"][0]
    embedding = sample["embeddings"][0]
    name      = sample["metadatas"][0].get("name", "?")
    artist    = sample["metadatas"][0].get("artist", "?")
 
    results = collection.query(
        query_embeddings=[embedding],
        n_results=6,
        include=["metadatas", "distances"],
    )
 
    print(f"\n  Seed: {name} — {artist}")
    print(f"  {'Song':<40} {'Artist':<25} {'Distance':>8}")
    print("  " + "─" * 75)
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        if meta.get("name") == name:
            continue   # skip self
        print(f"  {meta.get('name','?'):<40} {meta.get('artist','?'):<25} {dist:>8.4f}")
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
 
def run(chroma_only: bool = False):
    DATA_DIR.mkdir(exist_ok=True)
 
    ids, metadata, X = load_data()
 
    if chroma_only and UMAP_PATH.exists() and SCALER_PATH.exists():
        print("\n⏭️  --chroma-only: reusing existing scaler + UMAP model")
        X_scaled   = load_scaler(X)
        embeddings = load_umap_embeddings(X_scaled)
    else:
        X_scaled   = fit_scaler(X)
        embeddings = fit_umap(X_scaled)
 
    push_to_chromadb(ids, metadata, embeddings)
    sanity_check()
 
    print("\n✅ Stage 3 complete.")
    print("   → Next: python stage4_mood.py")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chroma-only", action="store_true",
                        help="Skip UMAP refit, just repopulate ChromaDB from saved model")
    args = parser.parse_args()
    run(chroma_only=args.chroma_only)