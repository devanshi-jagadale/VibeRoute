"""
stage1_collect.py — Stage 1: Load Kaggle dataset into SQLite
Replaces Spotify API collection entirely.
Spotify API is too restricted for new apps (Extended Quota Mode required).

Usage:
    python stage1_collect.py                        # load all, take 11k diverse sample
    python stage1_collect.py --mvp                  # take 1k sample
    python stage1_collect.py --csv path/to/file.csv # custom CSV path
    python stage1_collect.py --status               # show DB status
"""

import argparse
import pandas as pd
from utils.db import init_db, insert_songs, status

DEFAULT_CSV = "data/dataset.csv"
MVP_SIZE    = 1_000
FULL_SIZE   = 11_000

def load_kaggle_dataset(csv_path: str, sample_size: int) -> list[dict]:
    print(f"\n📂 Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"   Raw rows: {len(df)}")

    # drop rows missing essential fields
    df = df.dropna(subset=["track_id", "track_name", "artists"])
    df = df[df["track_id"].str.strip() != ""]
    print(f"   After cleaning: {len(df)}")

    # deduplicate by track_id
    df = df.drop_duplicates(subset="track_id")
    print(f"   After dedup: {len(df)}")

    # sample diverse subset — stratify by genre so all moods covered
    if "track_genre" in df.columns and len(df) > sample_size:
        try:
            # sample proportionally from each genre
            sampled = (
                df.groupby("track_genre", group_keys=False)
                .apply(lambda g: g.sample(
                    min(len(g), max(1, sample_size // df["track_genre"].nunique())),
                    random_state=42
                ))
            )
            # top up to sample_size if under
            if len(sampled) < sample_size:
                remaining = df[~df["track_id"].isin(sampled["track_id"])]
                topup = remaining.sample(
                    min(len(remaining), sample_size - len(sampled)),
                    random_state=42
                )
                sampled = pd.concat([sampled, topup])
            df = sampled.head(sample_size)
        except Exception:
            df = df.sample(min(len(df), sample_size), random_state=42)
    elif len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    print(f"   Sampled: {len(df)} tracks")

    # map to songs table schema
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "id":          str(row["track_id"]).strip(),
            "name":        str(row.get("track_name", "")).strip(),
            "artist":      str(row.get("artists", "")).strip(),
            "album":       str(row.get("album_name", "")).strip(),
            "year":        0,
            "duration_ms": int(row["duration_ms"]) if "duration_ms" in df.columns and pd.notna(row.get("duration_ms")) else 0,
            "spotify_uri": f"spotify:track:{str(row['track_id']).strip()}",
        })
    return rows


def run(csv_path: str, sample_size: int):
    init_db()
    rows = load_kaggle_dataset(csv_path, sample_size)

    inserted = insert_songs(rows)
    print(f"💾 Inserted {inserted} new songs into DB (duplicates skipped)")

    s = status()
    print(f"\n📈 DB Status: {s['total']} total | {s['done']} done | {s['pending']} pending | {s['failed']} failed")
    print("\n✅ Stage 1 complete. Run stage2_extract.py next.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvp",    action="store_true",          help="Load 1k sample")
    parser.add_argument("--csv",    default=DEFAULT_CSV,           help="Path to dataset CSV")
    parser.add_argument("--status", action="store_true",           help="Show DB status and exit")
    args = parser.parse_args()

    if args.status:
        print(status())
    else:
        sample_size = MVP_SIZE if args.mvp else FULL_SIZE
        run(args.csv, sample_size)