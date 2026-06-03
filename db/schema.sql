-- Songs table: raw metadata from Spotify
CREATE TABLE IF NOT EXISTS songs (
    id               TEXT PRIMARY KEY,   -- Spotify track ID
    name             TEXT NOT NULL,
    artist           TEXT NOT NULL,
    album            TEXT,
    year             INTEGER,
    duration_ms      INTEGER,
    spotify_uri      TEXT,
    processed_at     TEXT DEFAULT NULL,  -- ISO timestamp when features extracted
    extraction_error TEXT DEFAULT NULL   -- error message if failed
);

-- Features table: 26-dim librosa vector
CREATE TABLE IF NOT EXISTS features (
    song_id              TEXT PRIMARY KEY REFERENCES songs(id),
    tempo                REAL,
    energy_mean          REAL,
    energy_std           REAL,
    valence_proxy        REAL,
    danceability_proxy   REAL,
    acousticness_proxy   REAL,
    spectral_centroid    REAL,
    spectral_bandwidth   REAL,
    spectral_rolloff     REAL,
    spectral_contrast    REAL,
    zcr                  REAL,
    chroma_mean          REAL,
    chroma_std           REAL,
    mfcc_1               REAL,
    mfcc_2               REAL,
    mfcc_3               REAL,
    mfcc_4               REAL,
    mfcc_5               REAL,
    mfcc_6               REAL,
    mfcc_7               REAL,
    mfcc_8               REAL,
    mfcc_9               REAL,
    mfcc_10              REAL,
    mfcc_11              REAL,
    mfcc_12              REAL,
    mfcc_13              REAL
);

-- Progress tracking view
CREATE VIEW IF NOT EXISTS extraction_status AS
SELECT
    COUNT(*)                                          AS total,
    SUM(processed_at IS NOT NULL)                     AS done,
    SUM(extraction_error IS NOT NULL)                 AS failed,
    SUM(processed_at IS NULL AND extraction_error IS NULL) AS pending
FROM songs;