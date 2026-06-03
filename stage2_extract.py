"""
stage2_extract.py — Stage 2: Feature Extraction
Downloads 30s audio clips via yt-dlp, extracts 26-dim librosa features,
writes to SQLite. Resume-safe (skips already processed + failed).

Usage:
    python stage2_extract.py              # process all pending songs
    python stage2_extract.py --workers 4  # explicit worker count
    python stage2_extract.py --limit 50   # process only first 50 (test run)
    python stage2_extract.py --status     # show progress and exit
"""

import argparse
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from utils.db import get_pending, mark_done, mark_failed, status
from utils.audio import download_audio_clip, extract_features

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_WORKERS    = 4
SLEEP_BETWEEN      = 2.5   # seconds between yt-dlp requests (per worker)
CHECKPOINT_EVERY   = 100   # print progress every N songs

# ── Worker ────────────────────────────────────────────────────────────────────

_print_lock = Lock()

def process_song(row: dict) -> tuple[str, bool, str]:
    """
    Download + extract one song.
    Returns (song_id, success, message).
    Audio file is deleted immediately after extraction.
    """
    song_id = row["id"]
    name    = row["name"]
    artist  = row["artist"]
    audio_path = None

    try:
        audio_path = download_audio_clip(name, artist, duration_sec=30)
        features   = extract_features(audio_path)
        mark_done(song_id, features)
        return song_id, True, f"✅ {name} — {artist}"

    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:200]}"
        mark_failed(song_id, err)
        return song_id, False, f"❌ {name} — {artist} | {err}"

    finally:
        # Always delete audio file — don't accumulate 11k × 30s
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except OSError:
                pass
        time.sleep(SLEEP_BETWEEN)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(workers: int = DEFAULT_WORKERS, limit: int = 0):
    pending = get_pending(limit=limit)

    if not pending:
        print("🎉 No pending songs. All done (or run stage1_collect.py first).")
        s = status()
        print(f"DB status: {s}")
        return

    total   = len(pending)
    done    = 0
    failed  = 0

    print(f"\n🚀 Stage 2: extracting features for {total} songs")
    print(f"   Workers : {workers}")
    print(f"   Sleep   : {SLEEP_BETWEEN}s between requests per worker")
    eta_min = total * SLEEP_BETWEEN / workers / 60
    print(f"   Est. time: ~{eta_min:.0f} min ({eta_min/60:.1f}h)")
    print()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_song, dict(row)): dict(row) for row in pending}

        for i, future in enumerate(as_completed(futures), 1):
            song_id, success, msg = future.result()

            if success:
                done += 1
            else:
                failed += 1

            with _print_lock:
                print(f"  [{i}/{total}] {msg}")

            # Checkpoint: print summary every N songs
            if i % CHECKPOINT_EVERY == 0:
                s = status()
                with _print_lock:
                    print(f"\n  ── checkpoint {i}/{total} ──")
                    print(f"     done={s['done']}  failed={s['failed']}  pending={s['pending']}")
                    print()

    s = status()
    print(f"\n✅ Stage 2 complete.")
    print(f"   Total  : {total}")
    print(f"   Done   : {done}")
    print(f"   Failed : {failed}")
    print(f"\n📈 DB Status: total={s['total']} | done={s['done']} | failed={s['failed']} | pending={s['pending']}")
    print("\n→ Next: run stage3_embed.py (UMAP embeddings + ChromaDB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Number of parallel workers (default 4)")
    parser.add_argument("--limit",   type=int, default=0,
                        help="Only process first N pending songs (0 = all)")
    parser.add_argument("--status",  action="store_true",
                        help="Show DB status and exit")
    args = parser.parse_args()

    if args.status:
        s = status()
        print(f"DB Status: total={s['total']} | done={s['done']} | failed={s['failed']} | pending={s['pending']}")
    else:
        run(workers=args.workers, limit=args.limit)