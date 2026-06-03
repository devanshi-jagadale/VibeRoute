"""
stage5_graph.py — Stage 5: Mood Graph (Level 1)

Pipeline:
  1. Load cluster centroids from saved K-Means model
  2. Load cluster labels from clusters.json
  3. Compute cosine similarity between every pair of centroids
  4. Build graph: nodes = mood clusters, edges = similarity above threshold
  5. Validate graph is fully connected (every node reachable)
  6. Save graph to data/mood_graph.json

Usage:
    python stage5_graph.py
    python stage5_graph.py --threshold 0.5   # custom edge threshold (default 0.7)
    python stage5_graph.py --status          # print graph summary and exit
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

DATA_DIR       = Path("data")
KMEANS_PATH    = DATA_DIR / "kmeans_model.pkl"
CLUSTERS_PATH  = DATA_DIR / "clusters.json"
GRAPH_PATH     = DATA_DIR / "mood_graph.json"

DEFAULT_THRESHOLD = 0.7


# ── Load cluster data ─────────────────────────────────────────────────────────

def load_clusters() -> tuple[dict, np.ndarray]:
    print("📂 Loading cluster data...")

    if not KMEANS_PATH.exists():
        raise FileNotFoundError("K-Means model not found. Run stage4_mood.py first.")
    if not CLUSTERS_PATH.exists():
        raise FileNotFoundError("clusters.json not found. Run stage4_mood.py first.")

    with open(KMEANS_PATH, "rb") as f:
        km = pickle.load(f)

    with open(CLUSTERS_PATH) as f:
        clusters = {int(k): v for k, v in json.load(f).items()}

    centroids = km.cluster_centers_
    print(f"   ✅ {len(clusters)} clusters, centroids shape: {centroids.shape}")
    return clusters, centroids


# ── Cosine similarity ─────────────────────────────────────────────────────────

def l2_similarity_matrix(centroids: np.ndarray) -> np.ndarray:
    # convert L2 distance to similarity: higher = more similar
    diff = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # normalise to [0, 1] where 1 = identical
    max_dist = dist.max() + 1e-8
    return 1.0 - (dist / max_dist)


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph(
    clusters: dict,
    centroids: np.ndarray,
    threshold: float,
) -> dict:
    print(f"\n🔗 Building mood graph (threshold = {threshold})...")

    sim_matrix = l2_similarity_matrix(centroids)
    k = len(clusters)

    nodes = {}
    for cid, info in clusters.items():
        nodes[str(cid)] = {
            "id":          cid,
            "label":       info["label"],
            "description": info.get("description", ""),
            "size":        info["size"],
        }

    edges = []
    for i in range(k):
        for j in range(i + 1, k):
            sim = float(sim_matrix[i, j])
            if sim >= threshold:
                edges.append({
                    "source":     i,
                    "target":     j,
                    "weight":     round(sim, 4),
                    "source_label": clusters[i]["label"],
                    "target_label": clusters[j]["label"],
                })

    print(f"   Nodes : {len(nodes)}")
    print(f"   Edges : {len(edges)}  (similarity ≥ {threshold})")

    # print similarity matrix
    print(f"\n   L2 similarity matrix:")
    labels = [clusters[i]["label"][:12] for i in range(k)]
    header = "   " + " " * 14 + "  ".join(f"{l:>12}" for l in labels)
    print(header)
    for i in range(k):
        row = "  ".join(
            f"{'—':>12}" if i == j else f"{sim_matrix[i, j]:>12.3f}"
            for j in range(k)
        )
        print(f"   {labels[i]:>12}  {row}")

    return {"nodes": nodes, "edges": edges, "threshold": threshold}


# ── Connectivity check ────────────────────────────────────────────────────────

def check_connectivity(graph: dict) -> bool:
    nodes = set(int(k) for k in graph["nodes"].keys())
    edges = graph["edges"]

    if not edges:
        print("\n   ⚠️  No edges — graph is disconnected")
        return False

    # build adjacency
    adj = {n: set() for n in nodes}
    for e in edges:
        adj[e["source"]].add(e["target"])
        adj[e["target"]].add(e["source"])

    # BFS from node 0
    visited = set()
    queue   = [min(nodes)]
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        queue.extend(adj[node] - visited)

    isolated = nodes - visited
    if isolated:
        isolated_labels = [graph["nodes"][str(i)]["label"] for i in isolated]
        print(f"\n   ⚠️  Isolated nodes (not reachable): {isolated_labels}")
        print(f"   Try lowering --threshold (current: {graph['threshold']})")
        return False

    print(f"\n   ✅ Graph is fully connected — all {len(nodes)} nodes reachable")
    return True


# ── Print graph summary ───────────────────────────────────────────────────────

def print_graph_summary(graph: dict):
    nodes = graph["nodes"]
    edges = graph["edges"]

    print(f"\n{'ID':>4}  {'Label':<25}  {'Size':>6}  {'Connections'}")
    print("─" * 70)

    adj_labels = {int(k): [] for k in nodes.keys()}
    for e in edges:
        adj_labels[e["source"]].append(f"{e['target_label']} ({e['weight']:.2f})")
        adj_labels[e["target"]].append(f"{e['source_label']} ({e['weight']:.2f})")

    for nid, info in sorted(nodes.items(), key=lambda x: int(x[0])):
        connections = ", ".join(adj_labels[int(nid)]) or "none"
        print(f"{int(nid):>4}  {info['label']:<25}  {info['size']:>6}  {connections}")

    print(f"\n   Total edges: {len(edges)}")
    print(f"   Threshold  : {graph['threshold']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(threshold: float = DEFAULT_THRESHOLD):
    DATA_DIR.mkdir(exist_ok=True)

    clusters, centroids = load_clusters()
    graph = build_graph(clusters, centroids, threshold)
    connected = check_connectivity(graph)

    if not connected:
        # auto-lower threshold until connected
        print("\n   🔄 Auto-lowering threshold to achieve connectivity...")
        for t in [0.6, 0.5, 0.4, 0.3, 0.2]:
            graph = build_graph(clusters, centroids, t)
            if check_connectivity(graph):
                print(f"   ✅ Connected at threshold = {t}")
                break

    with open(GRAPH_PATH, "w") as f:
        json.dump(graph, f, indent=2)
    print(f"\n📄 Mood graph saved → {GRAPH_PATH}")

    print_graph_summary(graph)

    print("\n✅ Stage 5 complete.")
    print("   → Next: python stage6_pairs.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Cosine similarity threshold for edges (default 0.7)")
    parser.add_argument("--status", action="store_true",
                        help="Print graph summary and exit")
    args = parser.parse_args()

    if args.status:
        if not GRAPH_PATH.exists():
            print("No graph found. Run stage5_graph.py first.")
        else:
            with open(GRAPH_PATH) as f:
                graph = json.load(f)
            print_graph_summary(graph)
    else:
        run(threshold=args.threshold)