"""
stage6_pairs.py — Stage 6: Transition Pair Dataset

Pipeline:
  1. Load all songs + embeddings + mood_scores from SQLite + ChromaDB
  2. Build positive pairs  — consecutive songs from same cluster (proxy for good transitions)
  3. Build negative pairs  — random songs from distant mood zones
  4. Balance 50/50, target 50k-100k pairs total
  5. Save to data/pairs.npz  (feature vectors + labels)
     and data/pairs_meta.json (song ids for inspection)

Feature vector per pair (131 + 2k dims):
  concat(embedding_A, embedding_B)  → 128-dim
  Δtempo, Δenergy, Δvalence         →   3-dim
  mood_vector_A                     →   k-dim
  mood_vector_B                     →   k-dim
  total: 131 + 2k

Usage:
    python stage6_pairs.py
    python stage6_pairs.py --pairs 20000   # smaller set for MVP
    python stage6_pairs.py --status        # show saved pairs info and exit
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

DATA_DIR       = Path("data")
CHROMA_DIR     = str(DATA_DIR / "chromadb")
CLUSTERS_PATH  = DATA_DIR / "clusters.json"
PAIRS_PATH     = DATA_DIR / "pairs.npz"
PAIRS_META     = DATA_DIR / "pairs_meta.json"

TARGET_PAIRS   = 50_000
MOOD_DIST_THRESHOLD = 0.3   # cosine distance between mood vectors for negatives


# ── Load all song data ────────────────────────────────────────────────────────

def load_all_songs() -> tuple[list[str], np.ndarray, list[dict], list[dict]]:
    """
    Returns:
        ids         — list of song_ids
        embeddings  — (N, 64) float32
        raw_feats   — list of dicts with tempo, energy_mean, valence_proxy
        mood_scores — list of dicts {label: score, ...}
    """
    import chromadb
    from utils.db import get_conn

    print("📂 Loading song data...")

    # embeddings + metadata from ChromaDB
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection("songs")
    total      = collection.count()

    batch_size = 500
    all_ids, all_emb, all_meta = [], [], []

    for offset in range(0, total, batch_size):
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["embeddings", "metadatas"],
        )
        all_ids.extend(batch["ids"])
        all_emb.extend(batch["embeddings"])
        all_meta.extend(batch["metadatas"])

    embeddings = np.array(all_emb, dtype=np.float32)
    print(f"   ✅ {len(all_ids)} songs, embeddings: {embeddings.shape}")

    # raw audio features from SQLite
    conn = get_conn()
    feat_rows = conn.execute("""
        SELECT f.song_id, f.tempo, f.energy_mean, f.valence_proxy
        FROM features f
    """).fetchall()
    conn.close()

    feat_map = {r["song_id"]: dict(r) for r in feat_rows}

    # mood scores from ChromaDB metadata
    raw_feats   = []
    mood_scores = []

    for sid, meta in zip(all_ids, all_meta):
        f = feat_map.get(sid, {})
        raw_feats.append({
            "tempo":        float(f.get("tempo", 0)),
            "energy_mean":  float(f.get("energy_mean", 0)),
            "valence_proxy": float(f.get("valence_proxy", 0)),
        })
        ms_raw = meta.get("mood_scores", "{}")
        mood_scores.append(json.loads(ms_raw) if isinstance(ms_raw, str) else ms_raw)

    return all_ids, embeddings, raw_feats, mood_scores


# ── Mood vector helpers ───────────────────────────────────────────────────────

def mood_vec(mood_score: dict, mood_labels: list[str]) -> np.ndarray:
    return np.array([mood_score.get(l, 0.0) for l in mood_labels], dtype=np.float32)


def mood_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(1.0 - np.dot(a, b) / denom)


# ── Feature vector for a pair ─────────────────────────────────────────────────

def make_pair_vector(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    feat_a: dict,
    feat_b: dict,
    mvec_a: np.ndarray,
    mvec_b: np.ndarray,
) -> np.ndarray:
    delta = np.array([
        feat_b["tempo"]        - feat_a["tempo"],
        feat_b["energy_mean"]  - feat_a["energy_mean"],
        feat_b["valence_proxy"] - feat_a["valence_proxy"],
    ], dtype=np.float32)

    return np.concatenate([emb_a, emb_b, delta, mvec_a, mvec_b])


# ── Build positive pairs ──────────────────────────────────────────────────────

def build_positive_pairs(
    ids: list[str],
    embeddings: np.ndarray,
    raw_feats: list[dict],
    mood_scores: list[dict],
    mood_labels: list[str],
    n: int,
) -> tuple[list[np.ndarray], list[tuple[str, str]]]:
    """
    Positive pairs: actual embedding nearest neighbours from ChromaDB.
    Much stronger signal than same-cluster random pairs.
    """
    import chromadb
    print(f"\n➕ Building {n} positive pairs (embedding proximity)...")

    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection("songs")
    id_to_idx  = {sid: i for i, sid in enumerate(ids)}

    vectors = []
    meta    = []

    # sample a random subset of songs as seeds
    sample_ids = random.choices(ids, k=n * 2)  # allow repeats so we can hit 25k

    for src_id in sample_ids:
        if len(vectors) >= n:
            break

        i = id_to_idx.get(src_id)
        if i is None:
            continue

        # get top-6 nearest neighbours from ChromaDB (includes self at pos 0)
        results = collection.query(
            query_embeddings=[embeddings[i].tolist()],
            n_results=6,
            include=["metadatas"],
        )
        nbr_ids = [nid for nid in results["ids"][0] if nid != src_id]
        if not nbr_ids:
            continue

        # pick randomly from top-3 to add variety
        nbr_id = random.choice(nbr_ids[:3])
        j = id_to_idx.get(nbr_id)
        if j is None:
            continue

        mvec_a = mood_vec(mood_scores[i], mood_labels)
        mvec_b = mood_vec(mood_scores[j], mood_labels)
        vec = make_pair_vector(
            embeddings[i], embeddings[j],
            raw_feats[i],  raw_feats[j],
            mvec_a, mvec_b,
        )
        vectors.append(vec)
        meta.append((src_id, nbr_id))

    print(f"   ✅ Built {len(vectors)} positive pairs")
    return vectors, meta


# ── Build negative pairs ──────────────────────────────────────────────────────

def build_negative_pairs(
    ids: list[str],
    embeddings: np.ndarray,
    raw_feats: list[dict],
    mood_scores: list[dict],
    mood_labels: list[str],
    n: int,
) -> tuple[list[np.ndarray], list[tuple[str, str]]]:
    """
    Negative pairs: random songs from different clusters.
    """
    print(f"\n➖ Building {n} negative pairs...")

    from utils.db import get_conn
    conn = get_conn()
    cluster_rows = conn.execute(
        "SELECT song_id, cluster_id FROM song_moods"
    ).fetchall()
    conn.close()

    id_to_cluster = {row["song_id"]: row["cluster_id"] for row in cluster_rows}
    id_to_idx     = {sid: i for i, sid in enumerate(ids)}

    n_songs  = len(ids)
    vectors  = []
    meta     = []
    attempts = 0
    max_attempts = n * 10

    while len(vectors) < n and attempts < max_attempts:
        attempts += 1
        i = random.randint(0, n_songs - 1)
        j = random.randint(0, n_songs - 1)
        if i == j:
            continue

        sid_a, sid_b = ids[i], ids[j]

        # must be from different clusters
        if id_to_cluster.get(sid_a) == id_to_cluster.get(sid_b):
            continue

        mvec_a = mood_vec(mood_scores[i], mood_labels)
        mvec_b = mood_vec(mood_scores[j], mood_labels)

        vec = make_pair_vector(
            embeddings[i], embeddings[j],
            raw_feats[i],  raw_feats[j],
            mvec_a, mvec_b,
        )
        vectors.append(vec)
        meta.append((sid_a, sid_b))

    print(f"   ✅ Built {len(vectors)} negative pairs")
    return vectors, meta

# ── Save ──────────────────────────────────────────────────────────────────────

def save_pairs(
    pos_vecs: list[np.ndarray],
    neg_vecs: list[np.ndarray],
    pos_meta: list[tuple],
    neg_meta: list[tuple],
):
    X = np.array(pos_vecs + neg_vecs, dtype=np.float32)
    y = np.array([1] * len(pos_vecs) + [0] * len(neg_vecs), dtype=np.float32)

    # shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    np.savez_compressed(PAIRS_PATH, X=X, y=y)
    print(f"\n💾 Saved pairs → {PAIRS_PATH}")
    print(f"   Shape : X={X.shape}, y={y.shape}")
    print(f"   Positives : {int(y.sum())}")
    print(f"   Negatives : {int((1 - y).sum())}")

    meta_all = pos_meta + neg_meta
    meta_all = [meta_all[i] for i in idx]
    with open(PAIRS_META, "w") as f:
        json.dump([{"a": a, "b": b} for a, b in meta_all], f)
    print(f"   Metadata  → {PAIRS_META}")

    return X.shape[1]


# ── Status ────────────────────────────────────────────────────────────────────

def print_status():
    if not PAIRS_PATH.exists():
        print("No pairs found. Run stage6_pairs.py first.")
        return
    data = np.load(PAIRS_PATH)
    X, y = data["X"], data["y"]
    print(f"Pairs file: {PAIRS_PATH}")
    print(f"  X shape    : {X.shape}")
    print(f"  y shape    : {y.shape}")
    print(f"  Positives  : {int(y.sum())}")
    print(f"  Negatives  : {int((1 - y).sum())}")
    print(f"  Feature dim: {X.shape[1]}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(target_pairs: int = TARGET_PAIRS):
    random.seed(42)
    np.random.seed(42)

    DATA_DIR.mkdir(exist_ok=True)

    if not CLUSTERS_PATH.exists():
        raise FileNotFoundError("clusters.json not found. Run stage4_mood.py first.")

    with open(CLUSTERS_PATH) as f:
        clusters = {int(k): v for k, v in json.load(f).items()}
    mood_labels = [clusters[i]["label"] for i in range(len(clusters))]
    print(f"   Mood labels: {mood_labels}")

    ids, embeddings, raw_feats, mood_scores = load_all_songs()

    n_each = target_pairs // 2

    pos_vecs, pos_meta = build_positive_pairs(
        ids, embeddings, raw_feats, mood_scores, mood_labels, n_each
    )
    neg_vecs, neg_meta = build_negative_pairs(
        ids, embeddings, raw_feats, mood_scores, mood_labels, n_each
    )

    feat_dim = save_pairs(pos_vecs, neg_vecs, pos_meta, neg_meta)

    print(f"\n✅ Stage 6 complete.")
    print(f"   Feature dim per pair : {feat_dim}  (= 128 + 3 + 2×{len(mood_labels)})")
    print(f"   → Next: python stage7_classifier.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=TARGET_PAIRS,
                        help=f"Total pairs to generate (default {TARGET_PAIRS})")
    parser.add_argument("--status", action="store_true",
                        help="Show saved pairs info and exit")
    args = parser.parse_args()

    if args.status:
        print_status()
    else:
        run(target_pairs=args.pairs)