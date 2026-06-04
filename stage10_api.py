"""
stage10_api.py — Stage 10: FastAPI Backend

Endpoints:
    POST /generate
        body: { seed_track_id?, seed_query?, mood_arc, temperature?, playlist_len? }
        returns: { playlist: [...], arc, temperature, seed }

    GET /search?q=<query>
        returns: [ { id, name, artist, mood_scores, top_mood } ]

    GET /moods
        returns: { moods: [...], graph: { nodes, edges } }

    GET /playlist/last
        returns: last generated playlist from disk

    GET /health
        returns: { status, songs, moods }

Usage:
    pip install fastapi uvicorn
    uvicorn stage10_api:app --reload --port 8000

    Then hit:
        GET  http://localhost:8000/moods
        GET  http://localhost:8000/search?q=blinding+lights
        POST http://localhost:8000/generate
             { "mood_arc": ["Mellow Tunes", "High Energy"], "playlist_len": 20 }
"""

import os
import json
import random
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dotenv import load_dotenv

_SPOTIFY_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID", "")
_SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
_SPOTIFY_REDIRECT_URI  = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8000/spotify/callback")
_SPOTIFY_SCOPES = ("user-read-private playlist-modify-public playlist-modify-private")

load_dotenv()

# ── Paths (mirror stage9) ──────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
DATA_DIR        = BASE_DIR / "data"
CHROMA_DIR      = str(DATA_DIR / "chromadb")
DB_PATH         = DATA_DIR / "songs.db"          # ← add this
MODEL_PATH      = DATA_DIR / "mlp_classifier.pt"
MOOD_INDEX_PATH = DATA_DIR / "mood_index.json"
CLUSTERS_PATH   = DATA_DIR / "clusters.json"
GRAPH_PATH      = DATA_DIR / "song_graph.json"
MOOD_GRAPH_PATH = DATA_DIR / "mood_graph.json"
LAST_PLAYLIST   = DATA_DIR / "last_playlist.json"

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Mood-Aware Playlist Sequencer",
    description="Generates mood-arc playlists using UMAP embeddings, K-Means clustering, and an MLP transition classifier.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state (loaded once at startup) ─────────────────────────────────────

_resources: dict = {}
_ready: bool = False


def get_resources() -> dict:
    """Return resources, or 503 if still loading."""
    if not _resources:
        raise HTTPException(503, "Server is still warming up — try again in a moment")
    return _resources


def _load_all():
    """Load ChromaDB, clusters, graph, mood index, MLP."""
    import chromadb

    print("🚀 Loading pipeline resources...")

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # ── Rebuild ChromaDB if collection missing ────────────────────────────────
    existing = [c.name for c in client.list_collections()]
    if "songs" not in existing:
        print("   ⚠️  ChromaDB 'songs' collection not found — rebuilding from songs.db...")
        _rebuild_chromadb(client)

    collection = client.get_collection("songs")

    with open(CLUSTERS_PATH) as f:
        raw = json.load(f)
    clusters    = {int(k): v for k, v in raw.items()}
    mood_labels = [clusters[i]["label"] for i in range(len(clusters))]

    with open(GRAPH_PATH) as f:
        song_graph = json.load(f)

    mood_graph = None
    if MOOD_GRAPH_PATH.exists():
        with open(MOOD_GRAPH_PATH) as f:
            mood_graph = json.load(f)

    print("CLIENT ID =", _SPOTIFY_CLIENT_ID)
    print("CLIENT SECRET EXISTS =", bool(_SPOTIFY_CLIENT_SECRET))

    if MOOD_INDEX_PATH.exists():
        with open(MOOD_INDEX_PATH) as f:
            mood_index = json.load(f)
        print(f"   ✅ Mood index loaded ({sum(len(v) for v in mood_index.values())} entries)")
    else:
        mood_index = _build_mood_index(collection, mood_labels)

    model = None
    if MODEL_PATH.exists():
        model = _load_mlp()
        print("   ✅ MLP classifier loaded")
    else:
        print("   ⚠️  MLP not found — using cosine fallback")

    _resources.update({
        "collection":  collection,
        "clusters":    clusters,
        "mood_labels": mood_labels,
        "song_graph":  song_graph,
        "mood_graph":  mood_graph,
        "mood_index":  mood_index,
        "model":       model,
    })
    print(f"   ✅ {collection.count()} songs | {len(mood_labels)} moods | ready\n")


def _rebuild_chromadb(client):
    """Rebuild the ChromaDB songs collection from songs.db + saved models."""
    import sqlite3, pickle, json
    import numpy as np
    import umap

    FEATURE_COLS = [
        "tempo", "energy_mean", "energy_std",
        "valence_proxy", "danceability_proxy", "acousticness_proxy",
        "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
        "spectral_contrast", "zcr", "chroma_mean", "chroma_std",
        *[f"mfcc_{i}" for i in range(1, 14)],
    ]

    DB_PATH = BASE_DIR / "data" / "songs.db"

    # 1. Load features from SQLite
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT s.id, s.name, s.artist, s.album, s.year, s.language,
               f.tempo, f.energy_mean, f.energy_std,
               f.valence_proxy, f.danceability_proxy, f.acousticness_proxy,
               f.spectral_centroid, f.spectral_bandwidth, f.spectral_rolloff,
               f.spectral_contrast, f.zcr, f.chroma_mean, f.chroma_std,
               f.mfcc_1,  f.mfcc_2,  f.mfcc_3,  f.mfcc_4,  f.mfcc_5,
               f.mfcc_6,  f.mfcc_7,  f.mfcc_8,  f.mfcc_9,  f.mfcc_10,
               f.mfcc_11, f.mfcc_12, f.mfcc_13,
               sm.mood_scores
        FROM features f
        JOIN songs s ON s.id = f.song_id
        LEFT JOIN song_moods sm ON sm.song_id = f.song_id
        ORDER BY f.song_id
    """).fetchall()
    conn.close()
    print(f"   📂 Loaded {len(rows)} songs from SQLite")

    ids, metadatas, vectors = [], [], []
    for row in rows:
        r = dict(row)
        ids.append(r["id"])
        meta = {
            "name":   r["name"],
            "artist": r["artist"],
            "album":  r.get("album", "") or "",
            "year":   str(r.get("year", "") or ""),
            "language": r.get("language", "en") or "en",
            "mood_scores": r.get("mood_scores", "{}") or "{}",
            **{k: float(r[k]) for k in FEATURE_COLS},
        }
        metadatas.append(meta)
        vectors.append([float(r[k]) for k in FEATURE_COLS])

    X = np.array(vectors, dtype=np.float32)

    # 2. Scale using saved scaler
    SCALER_PATH = BASE_DIR / "data" / "scaler.pkl"
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X)

    # 3. Embed using saved UMAP model
    UMAP_PATH = BASE_DIR / "data" / "umap_model.pkl"
    with open(UMAP_PATH, "rb") as f:
        reducer = pickle.load(f)
    embeddings = reducer.transform(X_scaled)
    print(f"   ✅ Embeddings shape: {embeddings.shape}")

    # 4. Push to ChromaDB
    collection = client.create_collection(
        "songs",
        metadata={"hnsw:space": "l2"},
    )
    BATCH = 500
    for start in range(0, len(ids), BATCH):
        end = start + BATCH
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=metadatas[start:end],
        )
        print(f"   ↳ Inserted {min(end, len(ids))}/{len(ids)}")

    print(f"   ✅ ChromaDB rebuilt — {collection.count()} songs")


def _load_mlp():
    import torch
    import torch.nn as nn

    ckpt   = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    layers, prev = [], ckpt["input_dim"]
    for h in ckpt["hidden_dims"]:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(ckpt["dropout"])]
        prev = h
    layers += [nn.Linear(prev, 1), nn.Sigmoid()]
    model  = nn.Sequential(*layers)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _build_mood_index(collection, mood_labels: list[str]) -> dict:
    print("   🔨 Building mood index (one-time, ~30s)...")
    total_songs = collection.count()
    index       = {label: [] for label in mood_labels}

    for offset in range(0, total_songs, 500):
        batch = collection.get(limit=500, offset=offset, include=["metadatas"])
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
    print(f"   ✅ Mood index built ({sum(len(v) for v in index.values())} entries)")
    return index


# ── Pydantic models ───────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    seed_track_id: Optional[str] = Field(None, description="Exact song ID in the DB")
    seed_query:    Optional[str] = Field(None, description="Partial song name search")
    mood_arc:      list[str]     = Field(...,  description="Ordered list of mood labels")
    temperature:   float         = Field(0.8,  ge=0.05, le=3.0)
    playlist_len:  int           = Field(20,   ge=5, le=50)
    language:      Optional[str] = Field(None, description="Language filter: en, hi, ja, ko, fr, etc.")


class SongOut(BaseModel):
    id:               str
    name:             str
    artist:           str
    mood_scores:      dict[str, float]
    top_mood:         str
    transition_score: Optional[float] = None


class GenerateResponse(BaseModel):
    playlist:    list[SongOut]
    arc:         list[str]
    temperature: float
    seed:        Optional[str]
    total:       int


# ── Sampler (self-contained, no import from stage9) ───────────────────────────

BRIDGE_THRESHOLD = 0.4
CANDIDATE_K      = 30
EXPAND_K         = 60
GLOBAL_K         = 60
PRUNE_BELOW      = 0.3


def _mood_vec(mood_score: dict, mood_labels: list[str]) -> np.ndarray:
    return np.array([mood_score.get(l, 0.0) for l in mood_labels], dtype=np.float32)


def _parse_candidate(rid, meta, emb, mood_label, require_dominant=False, language=None):
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

    if language and meta.get("language", "en") != language:
        return None

    return {
        "id":            rid,
        "name":          meta.get("name", "?"),
        "artist":        meta.get("artist", "?"),
        "embedding":     emb,
        "mood_score":    mood_score,
        "mood_scores":   ms,
        "tempo":         float(meta.get("tempo", 0)),
        "energy_mean":   float(meta.get("energy_mean", 0)),
        "valence_proxy": float(meta.get("valence_proxy", 0)),
    }


def _get_candidates_graph(graph, collection, src_id, mood_label, played, src_emb, language=None):
    edges = graph.get(src_id, [])
    if not edges:
        return []

    neighbour_ids = [e["target"] for e in edges
                     if e["target"] not in played and e["score"] >= PRUNE_BELOW]
    graph_scores  = {e["target"]: e["score"] for e in edges}

    if not neighbour_ids:
        return []

    result = collection.get(
        ids=neighbour_ids[:CANDIDATE_K],
        include=["embeddings", "metadatas"],
    )

    candidates = []
    for rid, meta, emb in zip(result["ids"], result["metadatas"], result["embeddings"]):
        c = _parse_candidate(rid, meta, emb, mood_label, language=language)
        if c:
            c["graph_score"] = graph_scores.get(rid, 0.0)
            candidates.append(c)
    return candidates


def _get_candidates_ann(collection, src_emb, mood_label, played, k=60, language=None):
    results = collection.query(
        query_embeddings=[src_emb],
        n_results=k + 10,
        include=["embeddings", "metadatas"],
    )
    candidates = []
    for rid, meta, emb in zip(
        results["ids"][0], results["metadatas"][0], results["embeddings"][0]
    ):
        if rid in played:
            continue
        c = _parse_candidate(rid, meta, emb, mood_label, language=language)
        if c:
            c["source"] = "ann"
            candidates.append(c)
        if len(candidates) >= k:
            break
    return candidates


def _get_candidates_global(collection, mood_index, mood_label, played, k=60, language=None):
    pool = [sid for sid in mood_index.get(mood_label, []) if sid not in played]
    if not pool:
        return []
    sample = random.sample(pool, min(k * 3, len(pool)))
    result = collection.get(ids=sample[:k], include=["embeddings", "metadatas"])
    candidates = []
    for rid, meta, emb in zip(result["ids"], result["metadatas"], result["embeddings"]):
        c = _parse_candidate(rid, meta, emb, mood_label, require_dominant=True, language=language)
        if c:
            c["source"] = "global"
            candidates.append(c)
    return candidates


def _merge_candidates(existing, new):
    seen   = {c["id"] for c in existing}
    merged = existing[:]
    for c in new:
        if c["id"] not in seen:
            merged.append(c)
            seen.add(c["id"])
    return merged


def _get_candidates_tiered(graph, collection, mood_index, current, mood, played, language=None):
    candidates = _get_candidates_graph(
        graph, collection, current["id"], mood, played, current["embedding"], language=language
    )
    if len(candidates) >= 3:
        return candidates

    ann = _get_candidates_ann(collection, current["embedding"], mood, played, k=EXPAND_K, language=language)
    candidates = _merge_candidates(candidates, ann)
    if len(candidates) >= 3:
        return candidates

    glob = _get_candidates_global(collection, mood_index, mood, played, k=GLOBAL_K, language=language)
    candidates = _merge_candidates(candidates, glob)
    return candidates


def _get_bridge_candidates(graph, collection, mood_index, src_id, src_emb,
                            mood_a, mood_b, played):
    all_ids  = list(
        {e["target"] for e in graph.get(src_id, [])
         if e["target"] not in played and e["score"] >= PRUNE_BELOW}
    )
    if not all_ids:
        return []

    result = collection.get(ids=all_ids[:CANDIDATE_K], include=["embeddings", "metadatas"])
    bridges = []
    for rid, meta, emb in zip(result["ids"], result["metadatas"], result["embeddings"]):
        ms_raw = meta.get("mood_scores", "{}")
        ms     = json.loads(ms_raw) if isinstance(ms_raw, str) else ms_raw
        if ms.get(mood_a, 0) >= BRIDGE_THRESHOLD and ms.get(mood_b, 0) >= BRIDGE_THRESHOLD:
            bridges.append({
                "id":          rid,
                "name":        meta.get("name", "?"),
                "artist":      meta.get("artist", "?"),
                "embedding":   emb,
                "mood_score":  (ms.get(mood_a, 0) + ms.get(mood_b, 0)) / 2,
                "mood_scores": ms,
                "tempo":       float(meta.get("tempo", 0)),
                "energy_mean": float(meta.get("energy_mean", 0)),
                "valence_proxy": float(meta.get("valence_proxy", 0)),
                "source":      "bridge",
            })
    return bridges


def _score_candidates(model, current, candidates, mood_labels):
    if model is None or not candidates:
        # cosine fallback
        cur_emb = np.array(current["embedding"], dtype=np.float32)
        scores  = []
        for c in candidates:
            cand_emb = np.array(c["embedding"], dtype=np.float32)
            cos = float(np.dot(cur_emb, cand_emb) /
                        (np.linalg.norm(cur_emb) * np.linalg.norm(cand_emb) + 1e-8))
            scores.append(cos)
        return np.array(scores, dtype=np.float32)

    import torch
    cur_emb  = np.array(current["embedding"], dtype=np.float32)
    cur_feat = np.array([current["tempo"], current["energy_mean"], current["valence_proxy"]], dtype=np.float32)
    cur_mv   = _mood_vec(current.get("mood_scores", {}), mood_labels)

    rows = []
    for c in candidates:
        c_emb  = np.array(c["embedding"], dtype=np.float32)
        c_feat = np.array([c["tempo"], c["energy_mean"], c["valence_proxy"]], dtype=np.float32)
        c_mv   = _mood_vec(c.get("mood_scores", {}), mood_labels)
        delta  = c_feat - cur_feat
        rows.append(np.concatenate([cur_emb, c_emb, delta, cur_mv, c_mv]))

    X = np.array(rows, dtype=np.float32)
    with torch.no_grad():
        scores = model(torch.tensor(X)).numpy().flatten()
    return scores


def _softmax_sample(scores: np.ndarray, temperature: float) -> int:
    if scores.max() > 2 * (np.sort(scores)[-2] if len(scores) > 1 else scores[0]):
        scores = scores ** 0.5

    logits = scores / max(temperature, 1e-6)
    logits -= logits.max()
    probs  = np.exp(logits)
    probs /= probs.sum()
    return int(np.random.choice(len(scores), p=probs))


def generate_playlist(
    collection,
    mood_labels,
    graph,
    mood_index,
    model,
    mood_arc,
    seed_id,
    seed_meta,
    playlist_length,
    temperature,
    language=None,
) -> list[dict]:
    """Core sampler — adapted from stage9, returns list of song dicts."""
    ms_raw = seed_meta.get("mood_scores", "{}")
    ms     = json.loads(ms_raw) if isinstance(ms_raw, str) else ms_raw

    current = {
        "id":            seed_id,
        "name":          seed_meta.get("name", "?"),
        "artist":        seed_meta.get("artist", "?"),
        "embedding":     seed_meta.get("_embedding", []),
        "mood_score":    0.5,
        "mood_scores":   ms,
        "tempo":         float(seed_meta.get("tempo", 0)),
        "energy_mean":   float(seed_meta.get("energy_mean", 0)),
        "valence_proxy": float(seed_meta.get("valence_proxy", 0)),
    }

    playlist = [current]
    played   = {seed_id}

    n_zones     = len(mood_arc)
    zone_base   = playlist_length // n_zones
    remainder   = playlist_length - zone_base * n_zones
    zone_lengths = [zone_base + (1 if i < remainder else 0) for i in range(n_zones)]

    for zone_idx, (mood, zone_len) in enumerate(zip(mood_arc, zone_lengths)):
        is_last_zone = (zone_idx == n_zones - 1)
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
                candidates = _get_bridge_candidates(
                    graph, collection, mood_index,
                    current["id"], current["embedding"],
                    mood, next_mood, played,
                )
                got_bridge = bool(candidates)
                if not candidates:
                    candidates = _get_candidates_tiered(
                        graph, collection, mood_index, current, mood, played, language=language
                    )
            else:
                candidates = _get_candidates_tiered(
                    graph, collection, mood_index, current, mood, played, language=language
                )

            if not candidates:
                break

            scores       = _score_candidates(model, current, candidates, mood_labels)
            mood_weights = np.array([c["mood_score"] for c in candidates], dtype=np.float32)
            combined     = (1 - mood_alpha) * scores + mood_alpha * mood_weights

            idx    = _softmax_sample(combined, temperature)
            chosen = candidates[idx]
            chosen["transition_score"] = float(combined[idx])

            playlist.append(chosen)
            played.add(chosen["id"])
            current = chosen
            songs_in_zone += 1

    return playlist


# ── Seed resolution ───────────────────────────────────────────────────────────

def _resolve_seed_by_id(collection, seed_id: str):
    result = collection.get(ids=[seed_id], include=["embeddings", "metadatas"])
    if not result["ids"]:
        raise HTTPException(404, f"Song ID '{seed_id}' not found")
    meta                  = result["metadatas"][0]
    meta["_id"]           = result["ids"][0]
    meta["_embedding"]    = result["embeddings"][0]
    return result["ids"][0], meta


def _resolve_seed_by_query(collection, query: str):
    from utils.db import get_conn
    conn = get_conn()

    # search name first, then artist, then combined
    rows = conn.execute(
        "SELECT id, name, artist FROM songs WHERE name LIKE ? LIMIT 5",
        (f"%{query}%",)
    ).fetchall()

    if not rows:
        rows = conn.execute(
            "SELECT id, name, artist FROM songs WHERE artist LIKE ? LIMIT 5",
            (f"%{query}%",)
        ).fetchall()

    if not rows:
        # try each word in the query independently
        for word in query.split():
            if len(word) < 2:
                continue
            rows = conn.execute(
                "SELECT id, name, artist FROM songs WHERE name LIKE ? OR artist LIKE ? LIMIT 5",
                (f"%{word}%", f"%{word}%")
            ).fetchall()
            if rows:
                break

    conn.close()

    if not rows:
        raise HTTPException(
            404,
            f"No song found matching '{query}'. "
            f"Try GET /search?q={query} to see what's available."
        )

    return _resolve_seed_by_id(collection, rows[0]["id"])


def _resolve_seed_random(collection, mood_index, first_mood, language=None):  # ← add language
    pool = mood_index.get(first_mood, [])
    chosen_id = None
    if pool:
        if language:
            sample_ids = random.sample(pool, min(200, len(pool)))
            result_sample = collection.get(ids=sample_ids, include=["metadatas"])
            filtered_ids = [
                sid for sid, meta in zip(result_sample["ids"], result_sample["metadatas"])
                if meta.get("language", "en") == language
            ]
            chosen_id = random.choice(filtered_ids) if filtered_ids else random.choice(pool)
        else:
            chosen_id = random.choice(pool)

    if chosen_id:
        result = collection.get(ids=[chosen_id], include=["embeddings", "metadatas"])
    else:
        result = collection.get(limit=1, include=["embeddings", "metadatas"])

    meta               = result["metadatas"][0]
    meta["_id"]        = result["ids"][0]
    meta["_embedding"] = result["embeddings"][0]
    return result["ids"][0], meta


# ── Helper: format song for response ─────────────────────────────────────────

def _song_out(song: dict, include_transition: bool = True) -> SongOut:
    ms    = song.get("mood_scores", {})
    top   = max(ms, key=ms.get) if ms else ""
    return SongOut(
        id=song["id"],
        name=song["name"],
        artist=song["artist"],
        mood_scores=ms,
        top_mood=top,
        transition_score=song.get("transition_score") if include_transition else None,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    if not _resources:
        return {"status": "loading", "songs": 0, "moods": []}
    r = get_resources()
    return {
        "status": "ok",
        "songs":  r["collection"].count(),
        "moods":  r["mood_labels"],
    }


@app.get("/moods")
def get_moods():
    r = get_resources()
    return {
        "moods":      r["mood_labels"],
        "mood_graph": r.get("mood_graph"),
    }


@app.get("/search")
def search(q: str = Query(..., min_length=1, description="Song name search")):
    """Search songs by name, returns up to 10 matches with mood scores."""
    r = get_resources()

    from utils.db import get_conn
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, name, artist FROM songs WHERE name LIKE ? OR artist LIKE ? LIMIT 10",
        (f"%{q}%", f"%{q}%")
    ).fetchall()
    conn.close()

    if not rows:
        return {"results": []}

    ids    = [row["id"] for row in rows]
    result = r["collection"].get(ids=ids, include=["metadatas"])

    meta_map = {rid: m for rid, m in zip(result["ids"], result["metadatas"])}

    out = []
    for row in rows:
        sid  = row["id"]
        meta = meta_map.get(sid, {})
        ms_raw = meta.get("mood_scores", "{}")
        ms   = json.loads(ms_raw) if isinstance(ms_raw, str) else ms_raw
        top  = max(ms, key=ms.get) if ms else ""
        out.append({
            "id":          sid,
            "name":        row["name"],
            "artist":      row["artist"],
            "mood_scores": ms,
            "top_mood":    top,
        })

    return {"results": out}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """Generate a mood-arc playlist."""
    r = get_resources()

    # validate arc
    valid_moods = set(r["mood_labels"])
    bad = [m for m in req.mood_arc if m not in valid_moods]
    if bad:
        raise HTTPException(
            400,
            f"Unknown mood(s): {bad}. Valid moods: {r['mood_labels']}"
        )
    if len(req.mood_arc) < 1:
        raise HTTPException(400, "mood_arc must have at least 1 mood")

    # resolve seed
    seed_name = None
    if req.seed_track_id:
        seed_id, seed_meta = _resolve_seed_by_id(r["collection"], req.seed_track_id)
        seed_name = seed_meta.get("name")
    elif req.seed_query:
        seed_id, seed_meta = _resolve_seed_by_query(r["collection"], req.seed_query)
        seed_name = seed_meta.get("name")
    else:
        seed_id, seed_meta = _resolve_seed_random(
            r["collection"], r["mood_index"], req.mood_arc[0], language=req.language  # ← pass language to seed resolver
        )
        seed_name = seed_meta.get("name")

    playlist = generate_playlist(
        collection     = r["collection"],
        mood_labels    = r["mood_labels"],
        graph          = r["song_graph"],
        mood_index     = r["mood_index"],
        model          = r["model"],
        mood_arc       = req.mood_arc,
        seed_id        = seed_id,
        seed_meta      = seed_meta,
        playlist_length= req.playlist_len,
        temperature    = req.temperature,
        language       = req.language,  # ← pass language to sampler
    )

    # save to disk (mirrors stage9 behaviour)
    with open(LAST_PLAYLIST, "w") as f:
        json.dump([{
            "id":               s["id"],
            "name":             s["name"],
            "artist":           s["artist"],
            "mood_scores":      s.get("mood_scores", {}),
            "transition_score": s.get("transition_score"),
        } for s in playlist], f, indent=2)

    return GenerateResponse(
        playlist    = [_song_out(s) for s in playlist],
        arc         = req.mood_arc,
        temperature = req.temperature,
        seed        = seed_name,
        total       = len(playlist),
    )


@app.get("/playlist/last")
def last_playlist():
    """Return the most recently generated playlist from disk."""
    if not LAST_PLAYLIST.exists():
        raise HTTPException(404, "No playlist generated yet. Call POST /generate first.")
    with open(LAST_PLAYLIST) as f:
        data = json.load(f)
    return {"playlist": data, "total": len(data)}


# ── Spotify OAuth + Save Playlist ────────────────────────────────────────────
#
# Flow:
#   1. Frontend opens /spotify/login  in a popup
#   2. User logs in on Spotify, redirected to /spotify/callback
#   3. Callback exchanges code → access_token, closes popup via postMessage
#   4. Frontend sends POST /spotify/save with { access_token, songs, arc, seed }
#   5. Backend creates a public playlist, adds tracks, returns playlist URL
#
# Add to .env:
#   SPOTIFY_CLIENT_ID=...
#   SPOTIFY_CLIENT_SECRET=...
#   SPOTIFY_REDIRECT_URI=http://localhost:8000/spotify/callback

import urllib.parse as _urlparse
import base64 as _base64
import secrets as _secrets
import httpx as _httpx
from fastapi.responses import HTMLResponse as _HTMLResponse, RedirectResponse as _RedirectResponse

# In-memory state store (good enough for local/single-user; swap for Redis in prod)
_oauth_states: dict[str, str] = {}

@app.get("/spotify/login")
def spotify_login():
    """
    Redirect the user to Spotify's authorization page.
    Open this in a popup from the frontend.
    """
    state = _secrets.token_urlsafe(16)
    _oauth_states[state] = state  # store for CSRF check

    params = {
        "client_id":     _SPOTIFY_CLIENT_ID,
        "response_type": "code",
        "redirect_uri":  _SPOTIFY_REDIRECT_URI,
        "scope":         _SPOTIFY_SCOPES,
        "state":         state,
        "show_dialog":   "true",   # ← add this
    }
    url = "https://accounts.spotify.com/authorize?" + _urlparse.urlencode(params)
    return _RedirectResponse(url)


@app.get("/spotify/callback", response_class=_HTMLResponse)
async def spotify_callback(code: str = None, state: str = None, error: str = None):
    """
    Spotify redirects here after login.
    Exchanges code for access_token, then closes the popup via postMessage.
    """
    if error or not code:
        return _HTMLResponse("""
        <script>
          window.opener && window.opener.postMessage(
            { type: 'SPOTIFY_AUTH_ERROR', error: '""" + (error or 'no_code') + """' },
            '*'
          );
          window.close();
        </script>
        <p>Auth failed. You can close this tab.</p>
        """)

    # exchange code for token
    creds = _base64.b64encode(
        f"{_SPOTIFY_CLIENT_ID}:{_SPOTIFY_CLIENT_SECRET}".encode()
    ).decode()

    async with _httpx.AsyncClient() as client:
        resp = await client.post(
            "https://accounts.spotify.com/api/token",
            headers={
                "Authorization": f"Basic {creds}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "grant_type":   "authorization_code",
                "code":         code,
                "redirect_uri": _SPOTIFY_REDIRECT_URI,
            },
        )

    if resp.status_code != 200:
        return _HTMLResponse("""
        <script>
          window.opener && window.opener.postMessage(
            { type: 'SPOTIFY_AUTH_ERROR', error: 'token_exchange_failed' }, '*'
          );
          window.close();
        </script>
        """)

    token_data = resp.json()

    print("\nTOKEN DATA:")
    print(token_data)
    print("TOKEN SCOPES:", token_data.get("scope"))
    print()

    access_token = token_data.get("access_token", "")

    # ADD THIS BLOCK HERE
    async with _httpx.AsyncClient() as client:
        me = await client.get(
            "https://api.spotify.com/v1/me",
            headers={"Authorization": f"Bearer {access_token}"}
        )

    print("ME STATUS:", me.status_code)
    print("ME BODY:", me.text)
    print("ME SCOPES:", me.headers.get("scope"))
    print()

    # Send token back to the opener window, then close the popup
    return _HTMLResponse(f"""
    <script>
    window.opener && window.opener.postMessage(
        {{ type: 'SPOTIFY_AUTH_SUCCESS', access_token: '{access_token}' }},
        '*'
    );
    window.close();
    </script>
    <p>Logged in! Closing…</p>
    """)


class SpotifySaveRequest(BaseModel):
    access_token: str
    songs:        list[dict]   # each must have 'id' and 'name'
    arc:          list[str]
    seed:         Optional[str] = None
    temperature:  float = 0.8


@app.post("/spotify/save")
async def spotify_save(req: SpotifySaveRequest):
    """
    Creates a public Spotify playlist and adds the generated songs to it.
    Returns { playlist_url, playlist_id, added, skipped }.
    """
    import re
    SPOTIFY_ID_RE = re.compile(r'^[A-Za-z0-9]+$')

    headers = {
        "Authorization": f"Bearer {req.access_token}",
        "Content-Type":  "application/json",
    }

    print("SAVE TOKEN PREFIX:", req.access_token[:50])

    async with _httpx.AsyncClient() as client:

        # 1. Get current user ID
        me = await client.get("https://api.spotify.com/v1/me", headers=headers)
        if me.status_code == 401:
            raise HTTPException(401, "Spotify token expired. Please log in again.")
        me.raise_for_status()
        user_id = me.json()["id"]

        # 2. Create playlist
        arc_str = " → ".join(req.arc)
        pl_name = f"Smart Shuffle: {arc_str}"
        pl_desc = (
            f"Generated by Smart Shuffle · arc: {arc_str} · "
            f"temp: {req.temperature}"
            + (f" · seed: {req.seed}" if req.seed else "")
        )
        create = await client.post(
            "https://api.spotify.com/v1/me/playlists",
            headers=headers,
            json={
                "name": pl_name,
                "description": pl_desc,
                "public": True,
            },
        )

        print("CREATE PLAYLIST STATUS:", create.status_code)
        print("CREATE PLAYLIST RESPONSE:")
        print(create.text)
        print()
        if create.status_code != 201:
            print("CREATE PLAYLIST FAILED")
            print("Status:", create.status_code)
            print("Response:", create.text)
            raise HTTPException(status_code=create.status_code, detail=create.text)

        playlist_id  = create.json()["id"]
        playlist_url = create.json()["external_urls"]["spotify"]

        # 3. Build URIs — validate format, not length
        uris = []
        skipped = []

        print("SAMPLE IDs:", [(s.get("id", ""), len(str(s.get("id", "")))) for s in req.songs[:5]])

        for s in req.songs:
            sid = s.get("id", "")
            if sid and SPOTIFY_ID_RE.match(str(sid)):
                uris.append(f"spotify:track:{sid}")
            else:
                skipped.append(s.get("name", sid))

        print("URIS GENERATED:", len(uris))
        print("SKIPPED:", skipped)

        if uris:
            track_check = await client.get(
                f"https://api.spotify.com/v1/tracks/{uris[0].split(':')[2]}",
                headers=headers,
            )
            print("TRACK VALID:", track_check.status_code, track_check.text[:200])

        # 4. Add tracks in batches of 100
        added = 0
        for i in range(0, len(uris), 100):
            batch = uris[i:i + 100]  # ← use the actual slice
            
            r = await client.post(
                f"https://api.spotify.com/v1/playlists/{playlist_id}/items",
                headers=headers,  # ← also use the headers dict you already built
                json={"uris": batch},
            )
            
            if r.status_code not in (200, 201):
                print("ADD TRACKS FAILED:", r.status_code, r.text)
                raise HTTPException(status_code=r.status_code, detail=r.text)
            added += len(batch)
        return {
            "playlist_url": playlist_url,
            "playlist_id":  playlist_id,
            "added":        added,
            "skipped":      skipped,
        }
    

# ── Startup ───────────────────────────────────────────────────────────────────

# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Load resources in a background thread so the port binds immediately."""
    def _bg():
        global _ready
        try:
            _load_all()
            _ready = True
        except Exception as e:
            import traceback
            print("❌ STARTUP FAILED:", e)
            traceback.print_exc()
    threading.Thread(target=_bg, daemon=True).start()