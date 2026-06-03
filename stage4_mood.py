"""
stage4_mood.py — Stage 4: Mood Layer
 
Pipeline:
  1. Load 64-dim UMAP embeddings from ChromaDB
  2. Silhouette analysis → pick best K (range 8-25)
  3. K-Means clustering → cluster centroids
  4. Auto-name clusters via Claude API (JSON only)
  5. Compute softmax mood scores for every song
  6. Store cluster labels in SQLite + push mood_scores to ChromaDB
 
Usage:
    python stage4_mood.py
    python stage4_mood.py --skip-analysis   # use last saved K, skip silhouette sweep
    python stage4_mood.py --status          # show cluster summary and exit
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from groq import Groq

import numpy as np

DATA_DIR      = Path("data")
CHROMA_DIR    = str(DATA_DIR / "chromadb")
KMEANS_PATH   = DATA_DIR / "kmeans_model.pkl"
CLUSTERS_PATH = DATA_DIR / "clusters.json"

K_MIN = 8
K_MAX = 25

FEATURE_COLS = [
    "tempo", "energy_mean", "energy_std",
    "valence_proxy", "danceability_proxy", "acousticness_proxy",
    "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
    "spectral_contrast", "zcr", "chroma_mean", "chroma_std",
    *[f"mfcc_{i}" for i in range(1, 14)],
]


# ── Load embeddings from ChromaDB ─────────────────────────────────────────────

def load_embeddings() -> tuple[list[str], list[dict], np.ndarray]:
    import chromadb
    print("📂 Loading embeddings from ChromaDB...")
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection("songs")

    total = collection.count()
    print(f"   Found {total} songs in collection")

    # fetch in batches — ChromaDB has a default limit per get()
    batch_size = 500
    all_ids, all_meta, all_emb = [], [], []

    for offset in range(0, total, batch_size):
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["embeddings", "metadatas"],
        )
        all_ids.extend(batch["ids"])
        all_meta.extend(batch["metadatas"])
        all_emb.extend(batch["embeddings"])
        print(f"   Loaded {min(offset + batch_size, total)}/{total}")

    embeddings = np.array(all_emb, dtype=np.float32)
    print(f"   ✅ Embeddings shape: {embeddings.shape}")
    return all_ids, all_meta, embeddings


# ── Silhouette analysis → pick K ─────────────────────────────────────────────

def pick_best_k(embeddings: np.ndarray) -> int:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    print(f"\n📊 Silhouette analysis: K = {K_MIN} to {K_MAX}...")
    print(f"   {'K':>4}  {'Silhouette':>12}  {'Inertia':>14}")
    print("   " + "─" * 34)

    best_k     = K_MIN
    best_score = -1.0
    scores     = {}

    for k in range(K_MIN, K_MAX + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        score  = silhouette_score(embeddings, labels, sample_size=min(3000, len(embeddings)))
        scores[k] = score
        print(f"   {k:>4}  {score:>12.4f}  {km.inertia_:>14.1f}")

        if score > best_score:
            best_score = score
            best_k     = k

    print(f"\n   ✅ Best K = {best_k}  (silhouette = {best_score:.4f})")
    return best_k


# ── K-Means clustering ────────────────────────────────────────────────────────

def run_kmeans(embeddings: np.ndarray, k: int):
    from sklearn.cluster import KMeans
    print(f"\n🔵 Fitting K-Means (k={k})...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)

    with open(KMEANS_PATH, "wb") as f:
        pickle.dump(km, f)
    print(f"   ✅ K-Means model saved → {KMEANS_PATH}")
    return km, labels


# ── Cluster naming ─────────────────────────────────────────────────
def name_clusters(
    km,
    ids: list[str],
    metadata: list[dict],
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict[int, dict]:

    import json
    import os
    from groq import Groq
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found in .env")

    client = Groq(api_key=api_key)

    print(f"\n🤖 Naming {km.n_clusters} clusters via Groq...")

    clusters = {}
    cluster_data = {}

    for cluster_id in range(km.n_clusters):
        centroid = km.cluster_centers_[cluster_id]

        mask = np.where(labels == cluster_id)[0]

        distances = np.linalg.norm(
            embeddings[mask] - centroid,
            axis=1
        )

        top5_idx = mask[np.argsort(distances)[:5]]

        top5_songs = []

        for idx in top5_idx:
            m = metadata[idx]

            top5_songs.append({
                "name": m.get("name", "?"),
                "artist": m.get("artist", "?"),
                "tempo": round(m.get("tempo", 0), 1),
                "energy": round(m.get("energy_mean", 0), 3),
                "valence": round(m.get("valence_proxy", 0), 3),
                "danceability": round(
                    m.get("danceability_proxy", 0),
                    3
                ),
            })

        cluster_data[str(cluster_id)] = top5_songs

        clusters[cluster_id] = {
            "label": f"cluster_{cluster_id}",
            "description": "",
            "size": int(np.sum(labels == cluster_id)),
            "song_ids": [ids[i] for i in top5_idx],
        }

    prompt = f"""
You are naming music recommendation clusters.

For each cluster:

- Create a short mood label (1-3 words)
- Create a brief description

Examples:
Energetic
Late Night Drive
Happy Pop
Dance Floor
Chill Vibes
Melancholic
Workout Mix

Return ONLY valid JSON.

Cluster data:

{json.dumps(cluster_data, indent=2)}

Required format:

{{
  "0": {{
    "label": "...",
    "description": "..."
  }},
  "1": {{
    "label": "...",
    "description": "..."
  }}
}}
"""

    try:

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        parsed = json.loads(
            response.choices[0].message.content
        )

        for cluster_id in range(km.n_clusters):

            cluster_info = parsed.get(
                str(cluster_id),
                {}
            )

            clusters[cluster_id]["label"] = cluster_info.get(
                "label",
                f"cluster_{cluster_id}"
            )

            clusters[cluster_id]["description"] = cluster_info.get(
                "description",
                ""
            )

    except Exception as e:
        print(f"\n⚠️ Groq naming failed: {e}")
        print("Using fallback cluster names.")

    for cluster_id, info in clusters.items():
        print(
            f"   [{cluster_id:>2}] "
            f"{info['label']:<30} "
            f"({info['size']} songs)"
        )

    return clusters


# ── Softmax mood scores ───────────────────────────────────────────────────────

def compute_mood_scores(
    embeddings: np.ndarray,
    km,
    clusters: dict[int, dict],
) -> list[dict]:
    """
    For each song, compute softmax(−distances) over all cluster centroids.
    Returns list of dicts: {mood_label: score, ...}
    """
    print(f"\n⚙️  Computing mood scores for {len(embeddings)} songs...")
    centroids = km.cluster_centers_
    labels    = [clusters[i]["label"] for i in range(km.n_clusters)]

    # distances: (N, K)
    diff      = embeddings[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)

    # softmax over negative distances
    neg_dist = -distances
    neg_dist -= neg_dist.max(axis=1, keepdims=True)  # numerical stability
    exp_d    = np.exp(neg_dist)
    scores   = exp_d / exp_d.sum(axis=1, keepdims=True)

    mood_scores = []
    for row in scores:
        mood_scores.append({labels[i]: round(float(row[i]), 4) for i in range(len(labels))})

    print("   ✅ Done")
    return mood_scores


# ── Persist to SQLite ─────────────────────────────────────────────────────────

def save_to_sqlite(
    ids: list[str],
    labels: np.ndarray,
    mood_scores: list[dict],
    clusters: dict[int, dict],
):
    from utils.db import get_conn
    print("\n💾 Saving to SQLite...")
    conn = get_conn()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS clusters (
            id          INTEGER PRIMARY KEY,
            label       TEXT NOT NULL,
            description TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS song_moods (
            song_id     TEXT PRIMARY KEY,
            cluster_id  INTEGER,
            mood_scores TEXT   -- JSON string
        )
    """)

    conn.execute("DELETE FROM clusters")
    conn.execute("DELETE FROM song_moods")

    for cid, info in clusters.items():
        conn.execute(
            "INSERT INTO clusters (id, label, description) VALUES (?, ?, ?)",
            (cid, info["label"], info["description"]),
        )

    rows = [
        (ids[i], int(labels[i]), json.dumps(mood_scores[i]))
        for i in range(len(ids))
    ]
    conn.executemany(
        "INSERT INTO song_moods (song_id, cluster_id, mood_scores) VALUES (?, ?, ?)",
        rows,
    )

    conn.commit()
    conn.close()
    print(f"   ✅ clusters table: {len(clusters)} rows")
    print(f"   ✅ song_moods table: {len(rows)} rows")


# ── Push mood scores back to ChromaDB ────────────────────────────────────────

def update_chromadb(
    ids: list[str],
    mood_scores: list[dict],
    labels: np.ndarray,
    clusters: dict[int, dict],
    batch_size: int = 500,
):
    import chromadb
    print(f"\n🗄️  Updating ChromaDB metadata with mood scores...")
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection("songs")

    total = len(ids)
    for start in range(0, total, batch_size):
        end   = min(start + batch_size, total)
        chunk_ids = ids[start:end]

        # ChromaDB metadata values must be str/int/float — store as JSON string
        existing = collection.get(ids=chunk_ids, include=["metadatas"])
        meta_updates = []
        for i, existing_meta in enumerate(existing["metadatas"]):
            updated = {**existing_meta}
            updated["cluster_id"]    = int(labels[start + i])
            updated["cluster_label"] = clusters[int(labels[start + i])]["label"]
            updated["mood_scores"]   = json.dumps(mood_scores[start + i])
            meta_updates.append(updated)

        collection.update(ids=chunk_ids, metadatas=meta_updates)
        print(f"   📦 Updated {end}/{total}")

    print("   ✅ ChromaDB metadata updated")


# ── Cluster summary ───────────────────────────────────────────────────────────

def print_cluster_summary(clusters: dict[int, dict]):
    print(f"\n{'ID':>4}  {'Label':<30}  {'Size':>6}  Description")
    print("─" * 80)
    for cid, info in sorted(clusters.items()):
        desc = info.get("description", "")[:40]
        print(f"{cid:>4}  {info['label']:<30}  {info['size']:>6}  {desc}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(skip_analysis: bool = False):
    DATA_DIR.mkdir(exist_ok=True)

    ids, metadata, embeddings = load_embeddings()

    if skip_analysis and KMEANS_PATH.exists():
        print("\n⏭️  --skip-analysis: loading saved K-Means model")
        with open(KMEANS_PATH, "rb") as f:
            km = pickle.load(f)
        labels = km.predict(embeddings)
        print(f"   K = {km.n_clusters}")
    else:
        best_k = pick_best_k(embeddings)
        km, labels = run_kmeans(embeddings, best_k)

    clusters = name_clusters(km, ids, metadata, embeddings, labels)

    with open(CLUSTERS_PATH, "w") as f:
        json.dump(
            {str(k): v for k, v in clusters.items()},
            f, indent=2,
        )
    print(f"\n📄 Cluster map saved → {CLUSTERS_PATH}")

    mood_scores = compute_mood_scores(embeddings, km, clusters)
    save_to_sqlite(ids, labels, mood_scores, clusters)
    update_chromadb(ids, mood_scores, labels, clusters)

    print_cluster_summary(clusters)

    print("\n✅ Stage 4 complete.")
    print("   → Next: python stage5_graph.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip silhouette sweep, reuse saved K-Means model")
    parser.add_argument("--status", action="store_true",
                        help="Show cluster summary and exit")
    args = parser.parse_args()

    if args.status:
        if not CLUSTERS_PATH.exists():
            print("No clusters found. Run stage4_mood.py first.")
        else:
            with open(CLUSTERS_PATH) as f:
                clusters = {int(k): v for k, v in json.load(f).items()}
            print_cluster_summary(clusters)
    else:
        run(skip_analysis=args.skip_analysis)