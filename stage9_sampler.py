"""
stage9_sampler.py — Stage 9: Probabilistic Playlist Sampler

Generates a mood-arc playlist using:
  - Song graph (pre-scored MLP edges) as primary candidate source
  - ChromaDB ANN as secondary fallback
  - Global mood index as last-resort fallback (mood_index.json built on first run)
  - Softmax sampling with temperature
  - Mood blending that increases mood weight toward end of arc
  - Score dampening to prevent one song dominating
  - Bridge songs at mood boundaries
  - Played set to prevent repeats

Usage:
    python stage9_sampler.py
    python stage9_sampler.py --arc "Mellow Tunes,Upbeat Dance,High Energy"
    python stage9_sampler.py --arc "Mellow Tunes,High Energy" --seed "Tum Hi Ho" --temp 0.8 --length 20
    python stage9_sampler.py --temp 1.5    # adventurous
    python stage9_sampler.py --cosine      # cosine fallback
    python stage9_sampler.py --rebuild-index  # force rebuild mood_index.json
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

DATA_DIR        = Path("data")
CHROMA_DIR      = str(DATA_DIR / "chromadb")
MODEL_PATH      = DATA_DIR / "mlp_classifier.pt"
MOOD_INDEX_PATH = DATA_DIR / "mood_index.json"

USE_ML_SCORER    = True
DEFAULT_TEMP     = 0.8
DEFAULT_LENGTH   = 20
DEFAULT_ARC      = ["Mellow Tunes", "Upbeat Dance", "High Energy"]
BRIDGE_THRESHOLD = 0.4    # min mood score to qualify as bridge song
CANDIDATE_K      = 30     # initial candidate pool size
EXPAND_K         = 60     # expanded pool if not enough candidates
GLOBAL_K         = 60     # pool size for global mood index fallback


# ── Load everything ───────────────────────────────────────────────────────────

def load_resources(rebuild_index: bool = False):
    import chromadb

    print("📂 Loading resources...")

    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection("songs")

    with open(DATA_DIR / "clusters.json") as f:
        clusters = {int(k): v for k, v in json.load(f).items()}
    mood_labels = [clusters[i]["label"] for i in range(len(clusters))]

    with open(DATA_DIR / "song_graph.json") as f:
        graph = json.load(f)

    mood_index = load_or_build_mood_index(collection, mood_labels, rebuild=rebuild_index)

    model = None
    if USE_ML_SCORER and MODEL_PATH.exists():
        model = _load_mlp()

    print(f"   ✅ {collection.count()} songs | {len(mood_labels)} moods | graph loaded")
    return collection, mood_labels, graph, mood_index, model


def _load_mlp():
    import torch
    import torch.nn as nn
    ckpt   = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    layers, prev = [], ckpt["input_dim"]
    for h in ckpt["hidden_dims"]:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(ckpt["dropout"])]
        prev = h
    layers += [nn.Linear(prev, 1), nn.Sigmoid()]
    model = nn.Sequential(*layers)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ── Mood index ────────────────────────────────────────────────────────────────

def load_or_build_mood_index(collection, mood_labels: list[str], rebuild: bool = False) -> dict:
    """
    mood_index maps each mood label → list of song IDs where that mood
    is genuinely dominant. Built once, cached to disk.

    Indexing rules (applied per song):
      - Rank-0 mood (top scorer) with score >= 0.30  → indexed
      - Rank-1 mood (second scorer) with score >= 0.40
        AND within 15% of the top score              → indexed
    This prevents songs like (Upbeat Dance:0.64, High Energy:0.26) from
    polluting the High Energy index.
    """
    if MOOD_INDEX_PATH.exists() and not rebuild:
        with open(MOOD_INDEX_PATH) as f:
            index = json.load(f)
        total = sum(len(v) for v in index.values())
        print(f"   ✅ Mood index loaded ({total} entries across {len(index)} moods)")
        return index

    print("   🔨 Building mood index (one-time, ~30s)...")
    total_songs = collection.count()
    batch_size  = 500
    index       = {label: [] for label in mood_labels}

    for offset in range(0, total_songs, batch_size):
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"],
        )
        for sid, meta in zip(batch["ids"], batch["metadatas"]):
            ms_raw = meta.get("mood_scores", "{}")
            ms     = json.loads(ms_raw) if isinstance(ms_raw, str) else ms_raw
            if not ms:
                continue

            sorted_moods = sorted(ms.items(), key=lambda x: x[1], reverse=True)
            top_score    = sorted_moods[0][1]

            for rank, (label, score) in enumerate(sorted_moods):
                if label not in index:
                    continue
                if rank == 0 and score >= 0.30:
                    index[label].append(sid)
                elif rank == 1 and score >= 0.40 and score >= 0.85 * top_score:
                    index[label].append(sid)

    with open(MOOD_INDEX_PATH, "w") as f:
        json.dump(index, f)

    total = sum(len(v) for v in index.values())
    print(f"   ✅ Mood index built and saved ({total} entries) → {MOOD_INDEX_PATH}")
    for label in mood_labels:
        print(f"      {label:<22}: {len(index[label])} songs")
    return index


# ── Seed resolution ───────────────────────────────────────────────────────────

def resolve_seed(collection, query: str) -> tuple[str, dict]:
    """Find a song by name search in SQLite, fetch embedding from ChromaDB."""
    from utils.db import get_conn
    conn  = get_conn()
    rows  = conn.execute(
        "SELECT id, name, artist FROM songs WHERE name LIKE ? LIMIT 5",
        (f"%{query}%",)
    ).fetchall()
    conn.close()

    if not rows:
        raise ValueError(f"No song found matching '{query}'")

    row    = rows[0]
    result = collection.get(ids=[row["id"]], include=["embeddings", "metadatas"])
    if not result["ids"]:
        raise ValueError(f"Song '{query}' not in ChromaDB")

    meta = result["metadatas"][0]
    meta["_id"]        = result["ids"][0]
    meta["_embedding"] = result["embeddings"][0]
    print(f"   🎵 Seed: {meta.get('name')} — {meta.get('artist')}")
    return result["ids"][0], meta


# ── Candidate helpers ─────────────────────────────────────────────────────────

def _parse_candidate(rid: str, meta: dict, emb: list, mood_label: str,
                     graph_score=None, require_dominant: bool = False,
                     language: str | None = None) -> dict | None:
    """
    Parse a ChromaDB result into a candidate dict.

    require_dominant=True  (used by global tier): mood_label must be rank-0
      with score >= 0.30, or rank-1 with score >= 0.40 and within 15% of top.
      Anything weaker is rejected — keeps the index clean.

    require_dominant=False (graph / ANN tiers): looser floor of 0.15 to cut
      songs where the target mood is only a faint secondary signal.
    """
    ms_raw     = meta.get("mood_scores", "{}")
    ms         = json.loads(ms_raw) if isinstance(ms_raw, str) else ms_raw
    mood_score = ms.get(mood_label, 0.0)

    if require_dominant:
        sorted_moods = sorted(ms.items(), key=lambda x: x[1], reverse=True)
        top_score    = sorted_moods[0][1] if sorted_moods else 0.0
        rank         = next((i for i, (l, _) in enumerate(sorted_moods) if l == mood_label), 99)
        if rank == 0 and mood_score < 0.30:
            return None
        if rank == 1 and (mood_score < 0.40 or mood_score < 0.85 * top_score):
            return None
        if rank >= 2:
            return None
    else:
        if mood_score <= 0.15:
            return None

    # language filter
    if language and meta.get("language", "en") != language:
        return None

    return {
        "id":          rid,
        "name":        meta.get("name", "?"),
        "artist":      meta.get("artist", "?"),
        "embedding":   emb,
        "mood_score":  mood_score,
        "mood_scores": ms,
        "tempo":       float(meta.get("tempo", 0)),
        "energy":      float(meta.get("energy_mean", 0)),
        "valence":     float(meta.get("valence_proxy", 0)),
        "graph_score": graph_score,
        "language":    meta.get("language", "en"),
    }


# ── Candidate retrieval ───────────────────────────────────────────────────────

def get_candidates_from_graph(
    graph: dict,
    collection,
    current_id: str,
    mood_label: str,
    played: set,
    fallback_embedding: list,
    k: int = CANDIDATE_K,
    language: str | None = None,
) -> list[dict]:
    """
    Tier 1: walk pre-scored graph edges.
    Returns empty list (not a fallback here) if no unplayed neighbours exist,
    so the caller can decide which next tier to use.
    """
    edges      = graph.get(current_id, [])
    target_ids = [e["target"] for e in edges if e["target"] not in played]
    if not target_ids:
        return []

    result         = collection.get(ids=target_ids, include=["embeddings", "metadatas"])
    edge_score_map = {e["target"]: e["score"] for e in edges}

    candidates = []
    for rid, meta, emb in zip(result["ids"], result["metadatas"], result["embeddings"]):
        c = _parse_candidate(rid, meta, emb, mood_label,
                             graph_score=edge_score_map.get(rid, 0.0),
                             language=language)
        if c:
            c["source"] = "graph"
            candidates.append(c)

    candidates.sort(key=lambda x: x["mood_score"], reverse=True)
    return candidates[:k]


def get_candidates_ann(
    collection,
    seed_embedding: list,
    mood_label: str,
    played: set,
    k: int = CANDIDATE_K,
    language: str | None = None,
) -> list[dict]:
    """
    Tier 2: ChromaDB ANN query from current embedding, filtered by mood.
    Still acoustically local — will struggle when seed is far from target mood zone.
    """
    results = collection.query(
        query_embeddings=[seed_embedding],
        n_results=k * 3,
        include=["metadatas", "embeddings", "distances"],
    )

    candidates = []
    for rid, meta, emb, dist in zip(
        results["ids"][0],
        results["metadatas"][0],
        results["embeddings"][0],
        results["distances"][0],
    ):
        if rid in played:
            continue
        c = _parse_candidate(rid, meta, emb, mood_label, language=language)
        if c:
            c["distance"] = dist
            c["source"]   = "ann"
            candidates.append(c)

    candidates.sort(key=lambda x: x["mood_score"], reverse=True)
    return candidates[:k]


def get_candidates_global(
    collection,
    mood_index: dict,
    mood_label: str,
    played: set,
    k: int = GLOBAL_K,
    language: str | None = None,
) -> list[dict]:
    """
    Tier 3: global mood-first search using pre-built mood_index.
    Ignores acoustic proximity — guarantees we find songs in the target mood zone
    even when the current song is acoustically far from it.
    """
    pool = [sid for sid in mood_index.get(mood_label, []) if sid not in played]
    if not pool:
        return []

    # sample randomly from the mood pool so the playlist isn't always the same
    sample = random.sample(pool, min(k * 3, len(pool)))

    result = collection.get(ids=sample, include=["embeddings", "metadatas"])

    candidates = []
    for rid, meta, emb in zip(result["ids"], result["metadatas"], result["embeddings"]):
        c = _parse_candidate(rid, meta, emb, mood_label, require_dominant=True, language=language)
        if c:
            c["source"] = "global"
            candidates.append(c)

    candidates.sort(key=lambda x: x["mood_score"], reverse=True)
    return candidates[:k]


def get_bridge_candidates(
    graph: dict,
    collection,
    mood_index: dict,
    current_id: str,
    seed_embedding: list,
    zone_a: str,
    zone_b: str,
    played: set,
    k: int = CANDIDATE_K,
) -> list[dict]:
    """
    Songs scoring above BRIDGE_THRESHOLD in BOTH mood zones.
    Tries graph → ANN → global index in order.
    """
    # collect source candidates from graph first, then ANN
    edges      = graph.get(current_id, [])
    target_ids = [e["target"] for e in edges if e["target"] not in played]
    edge_score_map = {e["target"]: e["score"] for e in edges}

    if target_ids:
        result       = collection.get(ids=target_ids, include=["embeddings", "metadatas"])
        source_ids   = result["ids"]
        source_metas = result["metadatas"]
        source_embs  = result["embeddings"]
    else:
        ann          = collection.query(
            query_embeddings=[seed_embedding],
            n_results=k * 4,
            include=["metadatas", "embeddings"],
        )
        edge_score_map = {}
        source_ids   = ann["ids"][0]
        source_metas = ann["metadatas"][0]
        source_embs  = ann["embeddings"][0]

    candidates = []
    for rid, meta, emb in zip(source_ids, source_metas, source_embs):
        if rid in played:
            continue
        ms_raw  = meta.get("mood_scores", "{}")
        ms      = json.loads(ms_raw) if isinstance(ms_raw, str) else ms_raw
        score_a = ms.get(zone_a, 0.0)
        score_b = ms.get(zone_b, 0.0)
        if score_a > BRIDGE_THRESHOLD and score_b > BRIDGE_THRESHOLD:
            candidates.append({
                "id":          rid,
                "name":        meta.get("name", "?"),
                "artist":      meta.get("artist", "?"),
                "embedding":   emb,
                "mood_score":  (score_a + score_b) / 2,
                "mood_scores": ms,
                "tempo":       float(meta.get("tempo", 0)),
                "energy":      float(meta.get("energy_mean", 0)),
                "valence":     float(meta.get("valence_proxy", 0)),
                "graph_score": edge_score_map.get(rid, None),
                "is_bridge":   True,
            })

    # if graph/ANN found nothing, try global index intersection
    if not candidates:
        pool_a = set(mood_index.get(zone_a, []))
        pool_b = set(mood_index.get(zone_b, []))
        both   = list((pool_a & pool_b) - played)
        if both:
            sample = random.sample(both, min(k * 2, len(both)))
            result = collection.get(ids=sample, include=["embeddings", "metadatas"])
            for rid, meta, emb in zip(result["ids"], result["metadatas"], result["embeddings"]):
                ms_raw  = meta.get("mood_scores", "{}")
                ms      = json.loads(ms_raw) if isinstance(ms_raw, str) else ms_raw
                score_a = ms.get(zone_a, 0.0)
                score_b = ms.get(zone_b, 0.0)
                if score_a > BRIDGE_THRESHOLD and score_b > BRIDGE_THRESHOLD:
                    candidates.append({
                        "id":          rid,
                        "name":        meta.get("name", "?"),
                        "artist":      meta.get("artist", "?"),
                        "embedding":   emb,
                        "mood_score":  (score_a + score_b) / 2,
                        "mood_scores": ms,
                        "tempo":       float(meta.get("tempo", 0)),
                        "energy":      float(meta.get("energy_mean", 0)),
                        "valence":     float(meta.get("valence_proxy", 0)),
                        "graph_score": None,
                        "is_bridge":   True,
                    })

    candidates.sort(key=lambda x: x["mood_score"], reverse=True)
    return candidates[:k]


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_candidates(
    model,
    current: dict,
    candidates: list[dict],
    mood_labels: list[str],
) -> np.ndarray:
    """
    Score candidates against current song.
    Uses pre-computed graph_score where available.
    Falls back to MLP (or cosine) for candidates without a graph_score.
    Returns (N,) array in [0, 1].
    """
    if not candidates:
        return np.array([])

    # fast path: all candidates have graph scores
    if all(c.get("graph_score") is not None for c in candidates):
        return np.array([c["graph_score"] for c in candidates], dtype=np.float32)

    if model is not None and USE_ML_SCORER:
        import torch
        n      = len(candidates)
        emb_a  = np.array(current["embedding"], dtype=np.float32)
        emb_b  = np.array([c["embedding"] for c in candidates], dtype=np.float32)
        feat_a = np.tile([current["tempo"], current["energy"], current["valence"]], (n, 1)).astype(np.float32)
        feat_b = np.array([[c["tempo"], c["energy"], c["valence"]] for c in candidates], dtype=np.float32)
        delta  = feat_b - feat_a
        mvec_a = np.tile([current["mood_scores"].get(l, 0.0) for l in mood_labels], (n, 1)).astype(np.float32)
        mvec_b = np.array([[c["mood_scores"].get(l, 0.0) for l in mood_labels] for c in candidates], dtype=np.float32)
        X      = np.concatenate([np.tile(emb_a, (n, 1)), emb_b, delta, mvec_a, mvec_b], axis=1)

        with torch.no_grad():
            mlp_scores = model(torch.tensor(X, dtype=torch.float32)).numpy().flatten()

        # prefer graph_score where available, MLP elsewhere
        scores = mlp_scores.copy()
        for i, c in enumerate(candidates):
            if c.get("graph_score") is not None:
                scores[i] = c["graph_score"]
        return scores

    else:
        # cosine fallback
        a = np.array(current["embedding"], dtype=np.float32)
        a = a / (np.linalg.norm(a) + 1e-8)
        scores = []
        for c in candidates:
            if c.get("graph_score") is not None:
                scores.append(c["graph_score"])
            else:
                b = np.array(c["embedding"], dtype=np.float32)
                b = b / (np.linalg.norm(b) + 1e-8)
                scores.append(float(np.dot(a, b)))
        return np.array(scores, dtype=np.float32)


# ── Softmax sampling ──────────────────────────────────────────────────────────

def softmax_sample(scores: np.ndarray, temperature: float) -> int:
    """Sample index using softmax with temperature. Clips outlier top scores."""
    if len(scores) == 0:
        return 0

    if len(scores) > 1:
        med = float(np.median(scores))
        cap = 1.5 * med if med > 0 else float(scores.max())
        scores = np.clip(scores, None, cap)

    logits  = scores / (temperature + 1e-8)
    logits -= logits.max()
    exp     = np.exp(logits)
    probs   = exp / (exp.sum() + 1e-8)
    return int(np.random.choice(len(scores), p=probs))


# ── Main playlist generation ──────────────────────────────────────────────────

def generate_playlist(
    collection,
    mood_labels: list[str],
    graph: dict,
    mood_index: dict,
    model,
    mood_arc: list[str],
    seed_id: str,
    seed_meta: dict,
    playlist_length: int = DEFAULT_LENGTH,
    temperature: float = DEFAULT_TEMP,
    language: str | None = None,
) -> list[dict]:
    valid_moods = set(mood_labels)
    for m in mood_arc:
        if m not in valid_moods:
            raise ValueError(f"Unknown mood '{m}'. Valid: {sorted(valid_moods)}")

    playlist = []
    played   = set()

    current = {
        "id":          seed_id,
        "name":        seed_meta.get("name", "?"),
        "artist":      seed_meta.get("artist", "?"),
        "embedding":   seed_meta["_embedding"],
        "mood_scores": json.loads(seed_meta.get("mood_scores", "{}"))
                       if isinstance(seed_meta.get("mood_scores"), str)
                       else seed_meta.get("mood_scores", {}),
        "tempo":       float(seed_meta.get("tempo", 0)),
        "energy":      float(seed_meta.get("energy_mean", 0)),
        "valence":     float(seed_meta.get("valence_proxy", 0)),
        "graph_score": None,
    }
    playlist.append(current)
    played.add(seed_id)

    n_zones        = len(mood_arc)
    songs_per_zone = (playlist_length - 1) // n_zones
    remainder      = (playlist_length - 1) % n_zones
    zone_lengths   = [songs_per_zone + (1 if i < remainder else 0) for i in range(n_zones)]

    print(f"\n🎵 Generating playlist: {' → '.join(mood_arc)}")
    print(f"   Length: {playlist_length}  Temperature: {temperature}")
    print(f"   Zone lengths: {dict(zip(mood_arc, zone_lengths))}")
    print()

    for zone_idx, (mood, zone_len) in enumerate(zip(mood_arc, zone_lengths)):
        is_last_zone = zone_idx == n_zones - 1
        next_mood    = mood_arc[zone_idx + 1] if not is_last_zone else None

        zone_progress = zone_idx / max(n_zones - 1, 1)
        mood_alpha    = 0.4 + 0.3 * zone_progress

        songs_in_zone = 0
        got_bridge    = False

        while songs_in_zone < zone_len:
            need_bridge = (
                not is_last_zone
                and songs_in_zone == zone_len - 1
                and not got_bridge
            )

            if need_bridge:
                candidates = get_bridge_candidates(
                    graph, collection, mood_index,
                    current["id"], current["embedding"],
                    mood, next_mood, played,
                )
                got_bridge = bool(candidates)
                if not candidates:
                    candidates = _get_candidates_tiered(
                        graph, collection, mood_index,
                        current, mood, played,
                        language=language,              # ← fix
                    )
            else:
                candidates = _get_candidates_tiered(
                    graph, collection, mood_index,
                    current, mood, played,
                    language=language,                  # ← fix
                )

            if not candidates:
                print(f"   ⚠️  No candidates for mood '{mood}' — skipping zone")
                break

            scores       = score_candidates(model, current, candidates, mood_labels)
            mood_weights = np.array([c["mood_score"] for c in candidates], dtype=np.float32)
            combined     = (1 - mood_alpha) * scores + mood_alpha * mood_weights

            idx    = softmax_sample(combined, temperature)
            chosen = candidates[idx]

            playlist.append(chosen)
            played.add(chosen["id"])
            current = chosen
            songs_in_zone += 1

            src_tag  = " [global]" if chosen.get("source") == "global" else ""
            brd_tag  = " [bridge]" if need_bridge and got_bridge else ""
            print(f"   [{len(playlist):>3}] {chosen['name']:<40} {chosen['artist']:<25} "
                  f"[{mood}]{brd_tag}{src_tag}  score={combined[idx]:.3f}")

    return playlist


def _get_candidates_tiered(
    graph: dict,
    collection,
    mood_index: dict,
    current: dict,
    mood: str,
    played: set,
    language: str | None = None,
) -> list[dict]:
    """
    Three-tier candidate fetch with graceful fallback.
    Returns the first tier that yields >= 3 candidates.
    """
    # Tier 1: graph edges
    candidates = get_candidates_from_graph(
        graph, collection,
        current["id"], mood, played,
        current["embedding"],
        language=language,
    )
    if len(candidates) >= 3:
        return candidates

    # Tier 2: ANN from current embedding
    ann = get_candidates_ann(
        collection, current["embedding"], mood, played, k=EXPAND_K,
        language=language,
    )
    candidates = _merge_candidates(candidates, ann)
    if len(candidates) >= 3:
        return candidates

    # Tier 3: global mood index
    glob = get_candidates_global(collection, mood_index, mood, played, k=GLOBAL_K,
                                  language=language)
    candidates = _merge_candidates(candidates, glob)
    return candidates


def _merge_candidates(existing: list[dict], new: list[dict]) -> list[dict]:
    """Merge two candidate lists, deduplicating by id."""
    seen = {c["id"] for c in existing}
    merged = existing[:]
    for c in new:
        if c["id"] not in seen:
            merged.append(c)
            seen.add(c["id"])
    return merged


# ── Output ────────────────────────────────────────────────────────────────────

def print_playlist(playlist: list[dict], mood_arc: list[str]):
    print(f"\n{'─'*70}")
    print(f"  GENERATED PLAYLIST")
    print(f"  Arc: {' → '.join(mood_arc)}")
    print(f"{'─'*70}")
    for i, song in enumerate(playlist, 1):
        ms      = song.get("mood_scores", {})
        top2    = sorted(ms.items(), key=lambda x: x[1], reverse=True)[:2]
        top2_str = ", ".join(f"{k}:{v:.2f}" for k, v in top2)
        print(f"  {i:>3}. {song['name']:<40} {song['artist']:<25}")
        print(f"       moods: {top2_str}")
    print(f"{'─'*70}")
    print(f"  Total: {len(playlist)} songs")


# ── Main ──────────────────────────────────────────────────────────────────────
def run(
    arc: list[str],
    seed_query: str | None,
    length: int,
    temperature: float,
    use_cosine: bool = False,
    rebuild_index: bool = False,
    language: str | None = None,
):
    global USE_ML_SCORER
    if use_cosine:
        USE_ML_SCORER = False

    collection, mood_labels, graph, mood_index, model = load_resources(rebuild_index)

    print(f"\n   Available moods: {mood_labels}")

    for mood in arc:
        count = len(mood_index.get(mood, []))
        if count < 100:
            print(f"   ⚠️  '{mood}' has only {count} songs in index — playlist may be short")

    if seed_query:
        seed_id, seed_meta = resolve_seed(collection, seed_query)
    else:
        first_mood = arc[0]
        pool       = mood_index.get(first_mood, [])
        chosen_id  = None

        if pool:
            if language:
                # sample a subset and filter by language to avoid fetching entire pool
                sample_ids = random.sample(pool, min(200, len(pool)))
                result_sample = collection.get(ids=sample_ids, include=["metadatas"])
                filtered_ids = [
                    sid for sid, meta in zip(result_sample["ids"], result_sample["metadatas"])
                    if meta.get("language", "en") == language
                ]
                if filtered_ids:
                    chosen_id = random.choice(filtered_ids)
                else:
                    print(f"   ⚠️  No '{language}' seed found in '{first_mood}' — using any language")
                    chosen_id = random.choice(pool)
            else:
                chosen_id = random.choice(pool)

        if chosen_id:
            result = collection.get(ids=[chosen_id], include=["embeddings", "metadatas"])
        else:
            result = collection.get(limit=1, include=["embeddings", "metadatas"])

        seed_id              = result["ids"][0]
        seed_meta            = result["metadatas"][0]
        seed_meta["_id"]        = seed_id
        seed_meta["_embedding"] = result["embeddings"][0]
        print(f"   🎵 Seed (from '{first_mood}'): {seed_meta.get('name')} — {seed_meta.get('artist')}")

    if language:
        print(f"   🌐 Language filter: {language}")

    playlist = generate_playlist(
        collection, mood_labels, graph, mood_index, model,
        mood_arc=arc,
        seed_id=seed_id,
        seed_meta=seed_meta,
        playlist_length=length,
        temperature=temperature,
        language=language,
    )

    if len(playlist) < length:
        print(f"\n   ⚠️  Playlist shorter than requested ({len(playlist)}/{length} songs). "
              f"Try a different seed or broaden the arc.")

    print_playlist(playlist, arc)

    out_path = DATA_DIR / "last_playlist.json"
    with open(out_path, "w") as f:
        json.dump([{
            "id":          s["id"],
            "name":        s["name"],
            "artist":      s["artist"],
            "mood_scores": s.get("mood_scores", {}),
        } for s in playlist], f, indent=2)
    print(f"\n💾 Saved → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arc",           type=str, default=",".join(DEFAULT_ARC),
                        help="Comma-separated mood arc e.g. 'Mellow Tunes,High Energy'")
    parser.add_argument("--seed",          type=str, default=None,
                        help="Seed song name (partial match)")
    parser.add_argument("--length",        type=int, default=DEFAULT_LENGTH)
    parser.add_argument("--temp",          type=float, default=DEFAULT_TEMP)
    parser.add_argument("--cosine",        action="store_true",
                        help="Use cosine fallback instead of MLP")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="Force rebuild of mood_index.json")
    parser.add_argument("--lang",          type=str, default=None,
                        help="Language filter: en, hi, ja, ko, fr, etc.")
    args = parser.parse_args()

    arc = [m.strip() for m in args.arc.split(",")]
    run(
        arc=arc,
        seed_query=args.seed,
        length=args.length,
        temperature=args.temp,
        use_cosine=args.cosine,
        rebuild_index=args.rebuild_index,
        language=args.lang,
    )