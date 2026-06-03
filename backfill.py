"""
backfill_language.py — add language column to songs table and populate from CSV

Run once. Safe to rerun — uses UPDATE, not INSERT.

Usage:
    python backfill_language.py
    python backfill_language.py --csv data/dataset.csv
"""

import argparse
import pandas as pd
from pathlib import Path
from utils.db import get_conn

DEFAULT_CSV = "data/dataset.csv"

GENRE_TO_LANG = {
    # Indian
    "indian":       "hi",
    # East Asian
    "cantopop":     "zh",
    "mandopop":     "zh",
    "j-pop":        "ja",
    "j-rock":       "ja",
    "j-dance":      "ja",
    "j-idol":       "ja",
    "anime":        "ja",
    "k-pop":        "ko",
    # European / Latin
    "french":       "fr",
    "german":       "de",
    "swedish":      "sv",
    "spanish":      "es",
    "latin":        "es",
    "latino":       "es",
    "brazil":       "pt",
    "forro":        "pt",
    "samba":        "pt",
    "pagode":       "pt",
    "sertanejo":    "pt",
    "mpb":          "pt",
    "turkish":      "tr",
    "iranian":      "fa",
    "malay":        "ms",
    "tango":        "es",
    "salsa":        "es",
    "reggaeton":    "es",
}


def infer_language(genre: str) -> str:
    return GENRE_TO_LANG.get(genre.lower().strip(), "en")


def run(csv_path: str):
    conn = get_conn()

    # add language column if it doesn't exist
    try:
        conn.execute("ALTER TABLE songs ADD COLUMN language TEXT DEFAULT 'en'")
        conn.commit()
        print("✅ Added language column to songs table")
    except Exception:
        print("ℹ️  language column already exists")

    # load CSV — only need track_id and track_genre
    print(f"\n📂 Loading {csv_path}...")
    df = pd.read_csv(csv_path, usecols=["track_id", "track_genre"])
    df = df.dropna(subset=["track_id", "track_genre"])
    df["track_id"] = df["track_id"].str.strip()
    df["language"] = df["track_genre"].apply(infer_language)
    print(f"   {len(df)} rows loaded")

    # language distribution in CSV
    print("\n   Language distribution in CSV:")
    dist = df["language"].value_counts()
    for lang, count in dist.items():
        print(f"     {lang:>4} : {count}")

    # update songs table
    print("\n💾 Updating songs table...")
    updated = 0
    for _, row in df.iterrows():
        result = conn.execute(
            "UPDATE songs SET language = ? WHERE id = ?",
            (row["language"], row["track_id"])
        )
        updated += result.rowcount

    conn.commit()
    conn.close()
    print(f"   ✅ Updated {updated} songs")

    # verify
    conn = get_conn()
    dist_db = conn.execute(
        "SELECT language, COUNT(*) as n FROM songs GROUP BY language ORDER BY n DESC"
    ).fetchall()
    conn.close()

    print("\n📊 Language distribution in DB:")
    for row in dist_db:
        print(f"     {row['language']:>4} : {row['n']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    args = parser.parse_args()
    run(args.csv)