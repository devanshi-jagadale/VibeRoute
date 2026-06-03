# 🎵 VibeRoute

> Route your music through emotion.

VibeRoute is a graph-based mood-aware playlist generation system that creates playlists which intentionally travel through user-selected emotional states.

Unlike traditional recommendation systems that focus on finding similar songs, VibeRoute generates playlists that evolve through a sequence of moods while maintaining smooth transitions and musical coherence.

Example:

```text
Mellow Tunes
      ↓
Electronic Pulse
      ↓
Energetic Beats
```

---

## ✨ Features

### 🎭 Mood Arc Playlists

Create playlists that follow a chosen emotional journey.

Examples:

```text
Mellow Tunes → Electronic Pulse → Energetic Beats

Indie Delights → Upbeat Dance → High Energy

Folk Pop → Mellow Tunes → Electronic Pulse
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
  "Mellow Tunes": 0.52,
  "Energetic Beats": 0.34,
  "Folk Pop": 0.08
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
26-dimensional feature vector
```

for every song.

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
2. Run silhouette analysis
3. Determine optimal cluster count
4. Perform K-Means clustering
5. Auto-name clusters using an LLM
6. Compute mood scores for every song

Example discovered moods:

```text
Mellow Tunes
Electronic Pulse
Energetic Beats
Indie Delights
Folk Pop
Upbeat Dance
```

Songs exist in continuous mood space rather than fixed categories.

---

## Stage 5 — Mood Graph

Mood clusters become graph nodes.

Edges are automatically generated using similarity between cluster centroids.

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
Embedding A
Embedding B
ΔTempo
ΔEnergy
ΔValence
Mood Vector A
Mood Vector B
```

These pairs are used to train the transition scoring model.

---

## Stage 7 — Transition Classifier

A PyTorch neural network learns transition quality between songs.

Architecture:

```text
Input
 ↓
256
 ↓
128
 ↓
64
 ↓
Sigmoid
```

Output:

```text
Transition Score ∈ [0,1]
```

Higher scores indicate smoother musical transitions.

---

## Stage 8 — Song Graph

Each song becomes a node in a sparse graph.

For every song:

```text
Top 50 nearest neighbours
```

are retrieved and scored.

Low-quality edges are removed.

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
Coherent
+
Mood-Aware
+
Non-Repetitive
+
Different Every Run
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

---

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
Seed Song:
Kesariya

Mood Arc:
Mellow Tunes
→ Electronic Pulse
→ Energetic Beats

Temperature:
0.8
```

Output:

```text
Ambient / Chill Tracks
          ↓
Electronic Tracks
          ↓
High Energy Tracks
```

with smooth transitions between mood zones.

---

# 🔮 Future Improvements

- Language filtering
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