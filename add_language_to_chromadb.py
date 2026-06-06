"""
add_language_to_chromadb.py — patch genre + language into ChromaDB metadata
No reextraction needed. Reads CSV, matches by track_id, updates ChromaDB.

Run once:
    python add_language_to_chromadb.py
"""

import json
import pandas as pd
import chromadb
from pathlib import Path

DATA_DIR   = Path("data")
CHROMA_DIR = str(DATA_DIR / "chromadb")
DEFAULT_CSV = "data/dataset.csv"

GENRE_TO_LANG = {
    "indian":    "hi",
    "cantopop":  "zh", "mandopop": "zh",
    "j-pop":     "ja", "j-rock": "ja", "j-dance": "ja", "j-idol": "ja", "anime": "ja",
    "k-pop":     "ko",
    "french":    "fr",
    "german":    "de",
    "swedish":   "sv",
    "spanish":   "es", "latin": "es", "latino": "es", "tango": "es",
    "salsa":     "es", "reggaeton": "es",
    "brazil":    "pt", "forro": "pt", "samba": "pt",
    "pagode":    "pt", "sertanejo": "pt", "mpb": "pt",
    "turkish":   "tr",
    "iranian":   "fa",
    "malay":     "ms",
}

def infer_language(genre: str) -> str:
    return GENRE_TO_LANG.get(genre.lower().strip(), "en")

print("📂 Loading CSV...")
df = pd.read_csv(DEFAULT_CSV, usecols=["track_id", "track_genre"])
df = df.dropna(subset=["track_id", "track_genre"])
df["track_id"]  = df["track_id"].str.strip()
df["language"]  = df["track_genre"].apply(infer_language)
genre_map = dict(zip(df["track_id"], df["track_genre"]))
lang_map  = dict(zip(df["track_id"], df["language"]))
print(f"   {len(df)} rows, language dist: {df['language'].value_counts().to_dict()}")

print("\n🗄️  Updating ChromaDB metadata...")
client     = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection("songs")
total      = collection.count()

batch_size = 500
updated    = 0

for offset in range(0, total, batch_size):
    batch = collection.get(
        limit=batch_size,
        offset=offset,
        include=["metadatas"],
    )
    ids       = batch["ids"]
    metadatas = batch["metadatas"]

    new_metas = []
    for sid, meta in zip(ids, metadatas):
        updated_meta = {**meta}
        updated_meta["genre"]    = genre_map.get(sid, "unknown")
        updated_meta["language"] = lang_map.get(sid, "en")
        new_metas.append(updated_meta)

    collection.update(ids=ids, metadatas=new_metas)
    updated += len(ids)
    print(f"   📦 {min(offset + batch_size, total)}/{total}")

print(f"\n✅ Updated {updated} songs in ChromaDB with genre + language")

# verify
sample = collection.get(limit=5, include=["metadatas"])
print("\nSample:")
for sid, meta in zip(sample["ids"], sample["metadatas"]):
    print(f"  {meta.get('name','?'):<35} lang={meta.get('language','?')}  genre={meta.get('genre','?')}")
