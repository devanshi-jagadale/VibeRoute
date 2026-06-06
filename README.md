# 🎵 VibeRoute

> Route your music through emotion.

VibeRoute is a graph-based mood-aware playlist generation system that creates playlists which intentionally travel through user-selected emotional states.

Unlike traditional recommendation systems that focus on finding similar songs, VibeRoute generates playlists that evolve through a sequence of moods while maintaining smooth transitions and musical coherence.

Example:

```text
Mellow Reflections
      ↓
Dynamic Beats
      ↓
Energetic Anthems
```

---

## ✨ Features

### 🎭 Mood Arc Playlists

Create playlists that follow a chosen emotional journey.

Examples:

```text
Mellow Reflections → Dynamic Beats → Energetic Anthems

Indie Delights → Upbeat Dance → High Energy Party

Folk Inspired → Mellow Reflections → Dynamic Beats
```

---

### 🎵 Seed Song Support

Start playlist generation from a specific song:

```text
Kesariya
Blinding Lights
Shape of You
```

Or let the system automatically choose a seed.

---

### 🧠 Continuous Mood Representation

Songs are not assigned to a single mood.

Each song receives a continuous mood profile:

```json
{
  "Mellow Reflections": 0.72,
  "Energetic Anthems": 0.13,
  "Folk Inspired": 0.11
}
```

This allows smooth transitions between mood regions.

---

### 🔀 Different Every Run

Playlist generation uses:

- Softmax sampling
- Temperature control
- Graph traversal
- Transition scoring

This ensures playlists remain coherent while still being unique every time.

---

### 🎚 Temperature Control

Users can control exploration:

```text
Safe ←────→ Adventurous
```

Lower temperatures:
- More predictable
- Stronger transitions

Higher temperatures:
- More variety
- More surprising selections

---

### 🌐 Language Filtering

Filter playlists to songs in a specific language.

```bash
python stage9_sampler.py --arc "Mellow Reflections,Folk Inspired" --lang hi   # Hindi only
python stage9_sampler.py --arc "Indie Delights,Energetic Anthems" --lang en   # English only
python stage9_sampler.py --arc "Upbeat Dance,High Energy Party"   --lang ko   # Korean only
```

Supported language codes follow ISO 639-1 (`en`, `hi`, `ja`, `ko`, `fr`, `pt`, etc.).

Language filtering applies to both seed selection and all candidate retrieval tiers. If no seed is available in the requested language within the first mood zone, the system falls back to any language automatically.

---

### 🎵 Spotify Integration

Generated playlists can be exported directly to Spotify through OAuth authentication.

---

# 🏗 System Architecture

---

## Stage 1 — Dataset Collection

Music metadata is loaded from a Spotify tracks dataset and stored in SQLite.

Stored fields include:

- Track ID
- Song Name
- Artist
- Album
- Duration
- Spotify URI

The loader:

- Removes invalid records
- Removes duplicates
- Creates diverse samples
- Supports MVP (1k songs)
- Supports Full Dataset (~11k songs)

---

## Stage 2 — Audio Feature Extraction

For every song:

1. Download a short audio sample
2. Extract audio features using Librosa
3. Store features in SQLite

Extracted features include:

- Tempo
- Energy
- Valence Proxy
- Danceability Proxy
- Acousticness Proxy
- Spectral Features
- Chroma Features
- MFCC Features

Result:

```text
26-dimensional feature vector per song
```

---

## Stage 3 — Embedding Generation

Features are normalized using:

```text
StandardScaler
```

Then reduced using UMAP:

```text
26D → 64D
```

The resulting embeddings are stored in ChromaDB for efficient similarity search.

---

## Stage 4 — Mood Discovery

The system automatically discovers moods from the dataset.

Pipeline:

1. Load UMAP embeddings
2. Run silhouette analysis (K = 8 to 25)
3. Determine optimal cluster count
4. Perform K-Means clustering
5. Auto-name clusters using an LLM
6. Compute mood scores for every song

**Results (9942 songs):**

| Mood | Songs |
|------|-------|
| Energetic Anthems | 1857 |
| High Energy Party | 1745 |
| Upbeat Dance | 1484 |
| Indie Delights | 1527 |
| Mellow Reflections | 1470 |
| Folk Inspired | 1148 |
| Dynamic Beats | 688 |
| Experimental Vibes | 23 |

```text
Optimal K  : 8
Silhouette : 0.2964
```

Songs exist in continuous mood space rather than fixed categories.

---

## Stage 5 — Mood Graph

Mood clusters become graph nodes.

Edges are automatically generated using similarity between cluster centroids.

**Mood connectivity (cosine similarity):**

| | Upbeat Dance | Energetic Anthems | Mellow Reflections | High Energy Party | Dynamic Beats | Folk Inspired | Indie Delights |
|---|---|---|---|---|---|---|---|
| **Upbeat Dance** | — | 0.855 | 0.735 | 0.825 | 0.777 | 0.787 | 0.887 |
| **Energetic Anthems** | 0.855 | — | 0.870 | 0.707 | 0.749 | 0.846 | 0.837 |
| **Mellow Reflections** | 0.735 | 0.870 | — | 0.620 | 0.722 | 0.859 | 0.762 |
| **High Energy Party** | 0.825 | 0.707 | 0.620 | — | 0.827 | 0.732 | 0.853 |
| **Dynamic Beats** | 0.777 | 0.749 | 0.722 | 0.827 | — | 0.856 | 0.882 |
| **Folk Inspired** | 0.787 | 0.846 | 0.859 | 0.732 | 0.856 | — | 0.867 |
| **Indie Delights** | 0.887 | 0.837 | 0.762 | 0.853 | 0.882 | 0.867 | — |

```text
Total edges : 21
```

> **Note:** Experimental Vibes is an isolated cluster with no edges to other moods — it represents a genuinely distinct outlier region in the embedding space and is excluded from mood arc traversal.

The mood graph enables:

- Mood arc validation
- Navigation between mood regions
- Transition zone discovery

---

## Stage 6 — Transition Dataset

Training pairs are generated automatically.

Positive examples:

- Embedding-neighbor songs

Negative examples:

- Songs from distant mood regions

Each pair contains:

```text
Embedding A      (64D)
Embedding B      (64D)
ΔTempo
ΔEnergy
ΔValence
Mood Vector A    (8D)
Mood Vector B    (8D)
```

```text
Total pairs : 50,000  (25k positive / 25k negative)
Feature dim : 147
```

---

## Stage 7 — Transition Classifier

A PyTorch neural network learns transition quality between songs.

Architecture:

```text
Input (147)
 ↓
256 + ReLU + Dropout(0.3)
 ↓
128 + ReLU + Dropout(0.3)
 ↓
64  + ReLU + Dropout(0.3)
 ↓
Sigmoid
```

Output:

```text
Transition Score ∈ [0,1]
```

**Test set results:**

| Metric | Score |
|--------|-------|
| AUC | **0.9869** |
| Accuracy | **95.24%** |
| Precision | 91.27% |
| Recall | 99.88% |
| F1 | 95.38% |

Higher scores indicate smoother musical transitions.

---

## Stage 8 — Song Graph

Each song becomes a node in a sparse graph.

For every song, top-50 nearest neighbours are retrieved and scored by the transition classifier. Low-quality edges (score < 0.3) are pruned.

**Graph stats:**

```text
Nodes         : 9,942
Total edges   : 492,962
Avg degree    : 49.6
Isolated nodes: 0
Pruned edges  : 4,138
```

This creates a scalable graph suitable for real-time playlist generation.

---

## Stage 9 — Probabilistic Playlist Generation

Playlist generation combines:

- Song Graph Traversal
- ChromaDB ANN Search
- Mood Index Lookup
- Transition Scoring
- Softmax Sampling
- Bridge Song Selection
- Temperature Control

Generation process:

1. Start from seed song
2. Find mood-compatible candidates
3. Score transitions
4. Sample probabilistically
5. Move toward the next mood zone

Result:

```text
Coherent + Mood-Aware + Non-Repetitive + Different Every Run
```

---

## Stage 10 — FastAPI Backend

Available endpoints:

```http
GET  /health
GET  /moods
GET  /search
GET  /playlist/last
POST /generate
POST /spotify/save
```

Capabilities:

- Song search
- Mood graph access
- Playlist generation
- Spotify export
- OAuth authentication

---

# 🖥 Frontend

Built with:

- React
- Vite

Features:

- Song search
- Mood arc builder
- Temperature slider
- Playlist visualization
- Playlist regeneration
- Spotify export

---

# 🛠 Tech Stack

## Frontend

- React
- Vite
- JavaScript

## Backend

- FastAPI
- Python

## Machine Learning

- UMAP
- K-Means
- PyTorch
- NumPy
- Scikit-Learn

## Data Storage

- SQLite
- ChromaDB

## Music APIs

- Spotify Web API
- Spotify OAuth

---

# 🚀 Running Locally

## Backend

```bash
pip install -r requirements.txt

uvicorn stage10_api:app --reload --port 8000
```

Backend URL:

```text
http://localhost:8000
```

## Frontend

```bash
npm install

npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

---

# 📊 Example

Input:

```text
Seed Song  : I Surrender All — David Nevue
Mood Arc   : Mellow Reflections → Dynamic Beats → Energetic Anthems
Temperature: 0.8
```

Output:

```text
 1. I Surrender All               — David Nevue              [Mellow Reflections]
 2. Flute Quartet in D major      — Mozart / Galliano        [Mellow Reflections]
 3. From Gold                     — Novo Amor                [Mellow Reflections]
 4. Hamari Adhuri Kahani (Lofi)   — Arijit Singh             [Mellow Reflections]
 5. Follow                        — Martin Garrix & Zedd     [Mellow Reflections]
         ↓
 9. Sentinal                      — VNV Nation               [Dynamic Beats]
10. Troubles in Paradise          — Hozho                    [Dynamic Beats]
11. Ich & Du                      — AKA AKA & Umami          [Dynamic Beats]
         ↓
15. Stephen King                  — N.I.N.A                  [Energetic Anthems]
17. Mann Bharryaa 2.0             — B Praak                  [Energetic Anthems]
20. Subrosa (Come Closer)         — Shenseea                 [Energetic Anthems]
```

Smooth transitions across all 3 mood zones, 20 songs total.

---

# 🔮 Future Improvements
- Genre filtering
- User profiles
- Playlist sharing
- Personalized recommendations
- Real-time Spotify playback integration

---

# 🎯 Project Goal

Most music recommendation systems answer:

> "What songs are similar?"

VibeRoute answers:

> "How should a playlist evolve over time?"

By combining audio analysis, machine learning, graph traversal, and probabilistic sampling, VibeRoute generates playlists that intentionally travel through emotional space rather than simply grouping similar songs together.