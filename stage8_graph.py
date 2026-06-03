"""
stage8_graph.py — Stage 8: Song Graph + A* (Level 2)

Pipeline:
  1. Load all embeddings from ChromaDB
  2. For each song, find top-50 nearest neighbours (ChromaDB ANN)
  3. Score each edge with MLP classifier (or cosine fallback)
  4. Prune edges where score < 0.3
  5. Save sparse graph to data/song_graph.json
  6. A* pathfinding between any two songs via mood-constrained heuristic

Usage:
    python stage8_graph.py                    # build graph (takes ~10-20 min)
    python stage8_graph.py --cosine           # use cosine fallback instead of MLP
    python stage8_graph.py --test             # run A* test between two random songs
    python stage8_graph.py --status           # show graph stats
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

DATA_DIR    = Path("data")
CHROMA_DIR  = str(DATA_DIR / "chromadb")
MODEL_PATH  = DATA_DIR / "mlp_classifier.pt"
GRAPH_PATH  = DATA_DIR / "song_graph.json"
KMEANS_PATH = DATA_DIR / "kmeans_model.pkl"

TOP_K        = 50     # neighbours per song
PRUNE_BELOW  = 0.3    # drop edges with score < this
LAMBDA_MOOD  = 0.4    # A* heuristic mood weight

USE_ML_SCORER = True  # flip to False to use cosine fallback


# ── Load data ─────────────────────────────────────────────────────────────────

def load_songs() -> tuple[list[str], np.ndarray, list[dict]]:
    import chromadb
    print("📂 Loading songs from ChromaDB...")
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection("songs")
    total      = collection.count()

    all_ids, all_emb, all_meta = [], [], []
    for offset in range(0, total, 500):
        batch = collection.get(
            limit=500, offset=offset,
            include=["embeddings", "metadatas"],
        )
        all_ids.extend(batch["ids"])
        all_emb.extend(batch["embeddings"])
        all_meta.extend(batch["metadatas"])
        print(f"   Loaded {min(offset+500, total)}/{total}")

    embeddings = np.array(all_emb, dtype=np.float32)
    print(f"   ✅ {len(all_ids)} songs, embeddings: {embeddings.shape}")
    return all_ids, embeddings, all_meta


def load_mlp():
    import torch
    ckpt      = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    input_dim = ckpt["input_dim"]

    import torch.nn as nn
    layers, prev = [], input_dim
    for h in ckpt["hidden_dims"]:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(ckpt["dropout"])]
        prev = h
    layers += [nn.Linear(prev, 1), nn.Sigmoid()]
    model = nn.Sequential(*layers)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"   ✅ MLP loaded (input_dim={input_dim})")
    return model, input_dim


def load_mood_data(ids: list[str], meta: list[dict]) -> tuple[list[dict], list[str]]:
    """Extract mood score dicts and label list from metadata."""
    clusters_path = DATA_DIR / "clusters.json"
    with open(clusters_path) as f:
        clusters = {int(k): v for k, v in json.load(f).items()}
    mood_labels = [clusters[i]["label"] for i in range(len(clusters))]

    mood_scores = []
    for m in meta:
        ms_raw = m.get("mood_scores", "{}")
        ms     = json.loads(ms_raw) if isinstance(ms_raw, str) else ms_raw
        mood_scores.append(ms)

    return mood_scores, mood_labels


# ── Scoring ───────────────────────────────────────────────────────────────────

def cosine_score(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    a = emb_a / (np.linalg.norm(emb_a) + 1e-8)
    b = emb_b / (np.linalg.norm(emb_b) + 1e-8)
    return float(np.dot(a, b))


def mlp_score_batch(
    model,
    emb_a_batch: np.ndarray,
    emb_b_batch: np.ndarray,
    feat_a_batch: np.ndarray,
    feat_b_batch: np.ndarray,
    mvec_a_batch: np.ndarray,
    mvec_b_batch: np.ndarray,
) -> np.ndarray:
    import torch
    delta = feat_b_batch - feat_a_batch   # (B, 3)
    X = np.concatenate([
        emb_a_batch, emb_b_batch, delta, mvec_a_batch, mvec_b_batch
    ], axis=1)
    with torch.no_grad():
        scores = model(torch.tensor(X, dtype=torch.float32)).numpy().flatten()
    return scores


# ── Build sparse graph ────────────────────────────────────────────────────────

def build_graph(
    ids: list[str],
    embeddings: np.ndarray,
    meta: list[dict],
    use_ml: bool = True,
) -> dict:
    import chromadb

    print(f"\n🔗 Building sparse song graph (top-{TOP_K} neighbours per song)...")
    print(f"   Scorer: {'MLP' if use_ml else 'cosine fallback'}")
    print(f"   Prune threshold: {PRUNE_BELOW}")
    print(f"   Songs: {len(ids)}")

    model = None
    if use_ml:
        model, _ = load_mlp()

    # mood vectors
    mood_scores, mood_labels = load_mood_data(ids, meta)
    k_moods = len(mood_labels)

    id_to_idx = {sid: i for i, sid in enumerate(ids)}

    # raw features for MLP (tempo, energy, valence)
    raw_feats = np.array([
        [
            float(m.get("tempo", 0)),
            float(m.get("energy_mean", 0)),
            float(m.get("valence_proxy", 0)),
        ]
        for m in meta
    ], dtype=np.float32)

    # mood vectors matrix (N, k)
    mvecs = np.array([
        [ms.get(l, 0.0) for l in mood_labels]
        for ms in mood_scores
    ], dtype=np.float32)

    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection("songs")

    graph   = {}   # song_id → list of {target, score}
    total   = len(ids)
    pruned  = 0
    kept    = 0

    BATCH = 50   # query ChromaDB in batches for speed

    for batch_start in range(0, total, BATCH):
        batch_end  = min(batch_start + BATCH, total)
        batch_ids  = ids[batch_start:batch_end]
        batch_embs = embeddings[batch_start:batch_end]

        # Query ChromaDB for top-(K+1) neighbours (includes self)
        results = collection.query(
            query_embeddings=batch_embs.tolist(),
            n_results=TOP_K + 1,
            include=["metadatas"],
        )

        for local_i, (src_id, neighbour_ids) in enumerate(
            zip(batch_ids, results["ids"])
        ):
            global_i = batch_start + local_i
            edges    = []

            # collect neighbours (skip self)
            nbr_indices = []
            for nbr_id in neighbour_ids:
                if nbr_id == src_id:
                    continue
                if nbr_id in id_to_idx:
                    nbr_indices.append(id_to_idx[nbr_id])

            nbr_indices = nbr_indices[:TOP_K]
            if not nbr_indices:
                graph[src_id] = []
                continue

            if use_ml and model is not None:
                # batch MLP scoring
                n = len(nbr_indices)
                emb_a_b  = np.tile(embeddings[global_i], (n, 1))
                emb_b_b  = embeddings[nbr_indices]
                feat_a_b = np.tile(raw_feats[global_i], (n, 1))
                feat_b_b = raw_feats[nbr_indices]
                mvec_a_b = np.tile(mvecs[global_i], (n, 1))
                mvec_b_b = mvecs[nbr_indices]

                scores_arr = mlp_score_batch(
                    model,
                    emb_a_b, emb_b_b,
                    feat_a_b, feat_b_b,
                    mvec_a_b, mvec_b_b,
                )
            else:
                # cosine fallback
                scores_arr = np.array([
                    cosine_score(embeddings[global_i], embeddings[j])
                    for j in nbr_indices
                ])

            for nbr_idx, score in zip(nbr_indices, scores_arr):
                if score >= PRUNE_BELOW:
                    edges.append({
                        "target": ids[nbr_idx],
                        "score":  round(float(score), 4),
                    })
                    kept += 1
                else:
                    pruned += 1

            graph[src_id] = edges

        if (batch_start // BATCH) % 20 == 0:
            print(f"   [{batch_end}/{total}]  kept={kept}  pruned={pruned}")

    print(f"\n   ✅ Graph built")
    print(f"   Total edges : {kept + pruned}")
    print(f"   Kept (≥{PRUNE_BELOW})  : {kept}")
    print(f"   Pruned      : {pruned}")
    print(f"   Avg degree  : {kept / total:.1f}")

    return graph


# ── A* pathfinding ────────────────────────────────────────────────────────────

def astar(
    graph: dict,
    start_id: str,
    goal_mood_vec: np.ndarray,
    mood_scores: dict[str, dict],
    mood_labels: list[str],
    max_steps: int = 30,
) -> list[str]:
    """
    A* from start_id toward a target mood vector.
    Cost of each edge = (1 - transition_score) + λ * ||current_mood - target_mood||
    Returns list of song IDs (the path).
    """
    import heapq

    def mood_vec(song_id):
        ms = mood_scores.get(song_id, {})
        return np.array([ms.get(l, 0.0) for l in mood_labels], dtype=np.float32)

    def heuristic(song_id):
        mv = mood_vec(song_id)
        return float(np.linalg.norm(mv - goal_mood_vec))

    open_set  = [(0.0, start_id, [start_id])]
    visited   = set()

    while open_set:
        cost, current, path = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        if len(path) >= max_steps:
            return path

        for edge in graph.get(current, []):
            nbr = edge["target"]
            if nbr in visited:
                continue
            edge_cost = (1.0 - edge["score"]) + LAMBDA_MOOD * heuristic(nbr)
            new_cost  = cost + edge_cost
            heapq.heappush(open_set, (new_cost, nbr, path + [nbr]))

    return path


# ── Test A* ───────────────────────────────────────────────────────────────────

def test_astar(graph: dict, ids: list[str], meta: list[dict]):
    from utils.db import get_conn

    mood_scores, mood_labels = load_mood_data(ids, meta)
    id_to_ms = {sid: ms for sid, ms in zip(ids, mood_scores)}

    # pick two random songs
    start_id = ids[0]
    goal_id  = ids[len(ids) // 2]

    # use goal song's mood vector as target
    goal_ms  = id_to_ms.get(goal_id, {})
    goal_vec = np.array([goal_ms.get(l, 0.0) for l in mood_labels], dtype=np.float32)

    def song_name(sid):
        conn = get_conn()
        row = conn.execute("SELECT name, artist FROM songs WHERE id=?", (sid,)).fetchone()
        conn.close()
        return f"{row['name']} — {row['artist']}" if row else sid

    print(f"\n🔍 A* test:")
    print(f"   Start : {song_name(start_id)}")
    print(f"   Target mood: {goal_ms}")

    path = astar(graph, start_id, goal_vec, id_to_ms, mood_labels)

    print(f"\n   Path ({len(path)} steps):")
    for i, sid in enumerate(path[:10]):
        print(f"   {i+1:>3}. {song_name(sid)}")


# ── Save / load graph ─────────────────────────────────────────────────────────

def save_graph(graph: dict):
    with open(GRAPH_PATH, "w") as f:
        json.dump(graph, f)
    size_mb = GRAPH_PATH.stat().st_size / 1e6
    print(f"\n💾 Graph saved → {GRAPH_PATH}  ({size_mb:.1f} MB)")


def load_graph() -> dict:
    with open(GRAPH_PATH) as f:
        return json.load(f)


# ── Status ────────────────────────────────────────────────────────────────────

def print_status():
    if not GRAPH_PATH.exists():
        print("No graph found. Run stage8_graph.py first.")
        return
    graph = load_graph()
    total  = len(graph)
    edges  = sum(len(v) for v in graph.values())
    isolated = sum(1 for v in graph.values() if len(v) == 0)
    print(f"\n📊 Song graph stats:")
    print(f"   Nodes    : {total}")
    print(f"   Edges    : {edges}")
    print(f"   Avg deg  : {edges/total:.1f}")
    print(f"   Isolated : {isolated}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(use_ml: bool = True, run_test: bool = False):
    ids, embeddings, meta = load_songs()

    graph = build_graph(ids, embeddings, meta, use_ml=use_ml)
    save_graph(graph)
    print_status()

    if run_test:
        test_astar(graph, ids, meta)

    print("\n✅ Stage 8 complete.")
    print("   → Next: python stage9_sampler.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosine", action="store_true",
                        help="Use cosine fallback instead of MLP")
    parser.add_argument("--test",   action="store_true",
                        help="Run A* test after building graph")
    parser.add_argument("--status", action="store_true",
                        help="Show graph stats and exit")
    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.test and GRAPH_PATH.exists():
        ids, embeddings, meta = load_songs()
        graph = load_graph()
        test_astar(graph, ids, meta)
    else:
        run(use_ml=not args.cosine, run_test=args.test)