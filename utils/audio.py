"""
utils/audio.py — yt-dlp download + librosa feature extraction
Extracted from day1_features.py, reused by stage2_extract.py
"""
import os
import tempfile
import numpy as np
import librosa
import yt_dlp


def download_audio_clip(song_name: str, artist: str, duration_sec: int = 30) -> str:
    """
    Search YouTube, download first `duration_sec` seconds as MP3.
    Returns local file path. Caller is responsible for deleting it.
    """
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


def extract_features(audio_path: str) -> dict:
    """
    Extract 26-dim feature vector from an audio file using librosa.
    Returns a flat dict ready to be written to the features table.
    """
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    # ── Energy ────────────────────────────────────────────────────────────────
    rms         = librosa.feature.rms(y=y)[0]
    energy_mean = float(np.mean(rms))
    energy_std  = float(np.std(rms))

    # ── Tempo ─────────────────────────────────────────────────────────────────
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.asarray(tempo).flatten()[0])

    # Correct double-time / half-time detection errors (from pipeline spec)
    if tempo > 150 and energy_mean <= 0.25:
        tempo /= 2
    elif tempo < 90 and energy_mean > 0.20:
        tempo *= 2

    # ── Spectral ──────────────────────────────────────────────────────────────
    spectral_centroid  = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    spectral_rolloff   = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    spectral_contrast  = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
    zcr                = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))

    # ── MFCCs ─────────────────────────────────────────────────────────────────
    mfccs      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = {f"mfcc_{i+1}": float(np.mean(mfccs[i])) for i in range(13)}

    # ── Chroma ────────────────────────────────────────────────────────────────
    chroma      = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = float(np.mean(chroma))
    chroma_std  = float(np.std(chroma))

    # ── Proxies ───────────────────────────────────────────────────────────────
    major_score   = float(np.mean(chroma[[0, 4, 7]]))
    minor_score   = float(np.mean(chroma[[0, 3, 7]]))
    valence_proxy = float(np.clip((major_score - minor_score + 1) / 2, 0, 1))

    if len(beat_frames) > 1:
        beat_times     = librosa.frames_to_time(beat_frames, sr=sr)
        beat_intervals = np.diff(beat_times)
        dance_proxy    = float(np.clip(
            1 - np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-6), 0, 1
        ))
    else:
        dance_proxy = 0.0

    stft           = np.abs(librosa.stft(y))
    freqs          = librosa.fft_frequencies(sr=sr)
    low_energy     = float(np.mean(stft[freqs < 500]))
    high_energy    = float(np.mean(stft[freqs >= 500]))
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