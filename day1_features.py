import os
import tempfile
import numpy as np
import requests
import librosa
import yt_dlp
from spotify_auth import get_access_token

# ── Spotify Search ────────────────────────────────────────────────────────────

def search_track(token: str, query: str) -> dict:
    resp = requests.get(
        "https://api.spotify.com/v1/search",
        headers={"Authorization": f"Bearer {token}"},
        params={"q": f"track:{query}", "type": "track", "limit": 1},
    )
    resp.raise_for_status()
    items = resp.json()["tracks"]["items"]
    if not items:
        raise ValueError(f"No tracks found for: '{query}'")
    return items[0]

# ── YouTube Audio Download ────────────────────────────────────────────────────

def download_youtube_preview(song_name: str, artist: str, duration_sec: int = 30) -> str:
    """Search YouTube, download first 30s as MP3, return file path."""
    query = f"{song_name} {artist} official audio"
    tmp_path = tempfile.mktemp()

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": tmp_path + ".%(ext)s",
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
        "download_ranges": yt_dlp.utils.download_range_func(None, [(0, duration_sec)]),
        "force_keyframes_at_cuts": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(f"ytsearch1:{query}", download=True)

    return tmp_path + ".mp3"

# ── Librosa Feature Extraction ────────────────────────────────────────────────

def extract_features(audio_path: str) -> dict:
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    # Energy (computed first so tempo correction can use it)
    rms         = librosa.feature.rms(y=y)[0]
    energy_mean = float(np.mean(rms))
    energy_std  = float(np.std(rms))

    # Tempo & rhythm
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.asarray(tempo).flatten()[0])

    # Correct double-time / half-time detection errors
    if tempo > 150 and energy_mean <= 0.25:
        tempo = tempo / 2
    elif tempo < 60:
        tempo = tempo * 2

    # Spectral
    spectral_centroid  = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    spectral_rolloff   = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    spectral_contrast  = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
    zcr                = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))

    # MFCCs
    mfccs      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = {f"mfcc_{i+1}": float(np.mean(mfccs[i])) for i in range(13)}

    # Chroma
    chroma      = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = float(np.mean(chroma))
    chroma_std  = float(np.std(chroma))

    # Valence proxy (major vs minor chroma bins)
    major_score   = float(np.mean(chroma[[0, 4, 7]]))
    minor_score   = float(np.mean(chroma[[0, 3, 7]]))
    valence_proxy = float(np.clip((major_score - minor_score + 1) / 2, 0, 1))

    # Danceability proxy (beat interval consistency)
    if len(beat_frames) > 1:
        beat_times     = librosa.frames_to_time(beat_frames, sr=sr)
        beat_intervals = np.diff(beat_times)
        dance_proxy    = float(np.clip(
            1 - np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-6), 0, 1
        ))
    else:
        dance_proxy = 0.0

    # Acousticness proxy (low vs high freq energy ratio)
    stft        = np.abs(librosa.stft(y))
    freqs       = librosa.fft_frequencies(sr=sr)
    low_energy  = float(np.mean(stft[freqs < 500]))
    high_energy = float(np.mean(stft[freqs >= 500]))
    acoustic_proxy = float(low_energy / (low_energy + high_energy + 1e-6))

    return {
        "tempo":              tempo,
        "energy_mean":        energy_mean,
        "energy_std":         energy_std,
        "valence_proxy":      valence_proxy,
        "danceability_proxy": dance_proxy,
        "acousticness_proxy": acoustic_proxy,
        "spectral_centroid":  spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_rolloff":   spectral_rolloff,
        "spectral_contrast":  spectral_contrast,
        "zcr":                zcr,
        "chroma_mean":        chroma_mean,
        "chroma_std":         chroma_std,
        **mfcc_means,
    }

# ── Pretty Print ──────────────────────────────────────────────────────────────

def print_results(track: dict, features: dict):
    name       = track["name"]
    artists    = ", ".join(a["name"] for a in track["artists"])
    album      = track["album"]["name"]
    year       = track["album"]["release_date"][:4]
    duration_s = track["duration_ms"] // 1000

    print("\n" + "=" * 54)
    print(f"  🎵  {name}")
    print(f"  👤  {artists}")
    print(f"  💿  {album} ({year})")
    print(f"  🆔  {track['id']}")
    print(f"  ⏱️   {duration_s // 60}m {duration_s % 60}s")
    print("=" * 54)

    sections = {
        "Core Mood Features": [
            "tempo", "energy_mean", "energy_std",
            "valence_proxy", "danceability_proxy", "acousticness_proxy"
        ],
        "Spectral Texture": [
            "spectral_centroid", "spectral_bandwidth",
            "spectral_rolloff", "spectral_contrast", "zcr",
            "chroma_mean", "chroma_std"
        ],
        "MFCCs (Timbre)": [f"mfcc_{i}" for i in range(1, 14)],
    }

    for section, keys in sections.items():
        print(f"\n  ── {section} ──")
        print(f"  {'Feature':<25} {'Value':>10}")
        print("  " + "-" * 37)
        for k in keys:
            print(f"  {k:<25} {features[k]:>10.4f}")

    print("\n" + "=" * 54)

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    token = get_access_token()

    query = input("Enter a song name (tip: include artist e.g. 'blinding lights weeknd'): ").strip()
    print(f"\n🔍 Searching Spotify for: {query}")
    track = search_track(token, query)

    artists_str = ", ".join(a["name"] for a in track["artists"])
    print(f"✅ Found: {track['name']} — {artists_str}")

    print(f"⬇️  Fetching 30s audio from YouTube...")
    audio_path = download_youtube_preview(track["name"], artists_str)

    print(f"🔬 Extracting features with librosa...")
    features = extract_features(audio_path)

    print_results(track, features)

    os.unlink(audio_path)

    print(f"\n✅ Day 1 complete!")
    print(f"   Seed track ID : {track['id']}")
    print(f"   Feature vector: {len(features)} dimensions")
    print(f"\n→ Day 2: build candidate pool, extract features for all, store as JSON/CSV")