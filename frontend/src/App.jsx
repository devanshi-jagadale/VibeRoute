import { useState, useEffect, useRef, useCallback } from "react";

const API = "http://localhost:8000";

// ── Palette ───────────────────────────────────────────────────────────────────
const MOOD_COLORS = {
  "Mellow Tunes":      { bg: "#0d1117", accent: "#7c6af7", light: "#2a2347" },
  "High Energy":       { bg: "#0d1117", accent: "#f7604a", light: "#3a1a13" },
  "Upbeat Dance":      { bg: "#0d1117", accent: "#f7c94a", light: "#3a3113" },
  "Energetic Beats":   { bg: "#0d1117", accent: "#4af7a0", light: "#133a2a" },
  "Electronic Pulse":  { bg: "#0d1117", accent: "#4ab8f7", light: "#132d3a" },
  "Experimental Vibes":{ bg: "#0d1117", accent: "#f74ab8", light: "#3a133a" },
  "Folk Pop":          { bg: "#0d1117", accent: "#f7964a", light: "#3a2013" },
  "Indie Delights":    { bg: "#0d1117", accent: "#a0f74a", light: "#253a13" },
};
const DEFAULT_COLOR = { accent: "#7c6af7", light: "#2a2347" };

function moodColor(mood) {
  return MOOD_COLORS[mood] || DEFAULT_COLOR;
}

// ── API helpers ───────────────────────────────────────────────────────────────
async function apiFetch(path, opts = {}) {
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ── Components ────────────────────────────────────────────────────────────────

function SearchBar({ onSelect }) {
  const [q, setQ] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const debounce = useRef(null);
  const wrapRef = useRef(null);

  const search = useCallback((query) => {
    if (!query.trim()) { setResults([]); setOpen(false); return; }
    clearTimeout(debounce.current);
    debounce.current = setTimeout(async () => {
      setLoading(true);
      try {
        const data = await apiFetch(`/search?q=${encodeURIComponent(query)}`);
        setResults(data.results || []);
        setOpen(true);
      } catch { setResults([]); }
      setLoading(false);
    }, 350);
  }, []);

  useEffect(() => { search(q); }, [q, search]);

  useEffect(() => {
    function handler(e) {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false);
    }
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  function pick(song) {
    setQ(song.name + " — " + song.artist);
    setOpen(false);
    onSelect(song);
  }

  return (
    <div ref={wrapRef} style={{ position: "relative" }}>
      <div style={styles.searchWrap}>
        <span style={styles.searchIcon}>♪</span>
        <input
          value={q}
          onChange={e => setQ(e.target.value)}
          placeholder="Search for a seed song…"
          style={styles.searchInput}
          onFocus={() => results.length && setOpen(true)}
        />
        {loading && <span style={styles.searchSpinner}>○</span>}
        {q && (
          <button onClick={() => { setQ(""); setResults([]); setOpen(false); onSelect(null); }}
            style={styles.clearBtn}>✕</button>
        )}
      </div>
      {open && results.length > 0 && (
        <div style={styles.dropdown}>
          {results.map(s => (
            <div key={s.id} style={styles.dropdownItem} onClick={() => pick(s)}>
              <div style={styles.dropdownName}>{s.name}</div>
              <div style={styles.dropdownMeta}>
                <span style={styles.dropdownArtist}>{s.artist}</span>
                <span style={{ ...styles.moodPill, background: moodColor(s.top_mood).accent + "22", color: moodColor(s.top_mood).accent }}>
                  {s.top_mood}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
      {open && q && results.length === 0 && !loading && (
        <div style={styles.dropdown}>
          <div style={{ padding: "12px 16px", color: "#666", fontSize: 13 }}>No songs found — try a different title or artist</div>
        </div>
      )}
    </div>
  );
}

function MoodArcPicker({ moods, arc, setArc }) {
  function toggle(mood) {
    if (arc.includes(mood)) {
      setArc(arc.filter(m => m !== mood));
    } else if (arc.length < 3) {
      setArc([...arc, mood]);
    }
  }
  function moveUp(i) { if (i === 0) return; const a = [...arc]; [a[i-1],a[i]]=[a[i],a[i-1]]; setArc(a); }
  function moveDown(i) { if (i === arc.length-1) return; const a=[...arc]; [a[i],a[i+1]]=[a[i+1],a[i]]; setArc(a); }
  function remove(i) { setArc(arc.filter((_,j)=>j!==i)); }

  return (
    <div>
      <div style={styles.sectionLabel}>Choose mood arc <span style={styles.hint}>(pick 1–3, in order)</span></div>

      <div style={styles.moodGrid}>
        {moods.map(m => {
          const sel = arc.includes(m);
          const c = moodColor(m);
          return (
            <button key={m} onClick={() => toggle(m)}
              style={{
                ...styles.moodChip,
                background: sel ? c.accent + "22" : "rgba(255,255,255,0.04)",
                border: `1px solid ${sel ? c.accent : "rgba(255,255,255,0.1)"}`,
                color: sel ? c.accent : "#aaa",
              }}>
              <span style={{ width: 8, height: 8, borderRadius: "50%", background: sel ? c.accent : "transparent", border: `1.5px solid ${sel ? c.accent : "#555"}`, display: "inline-block", marginRight: 8, flexShrink: 0 }} />
              {m}
            </button>
          );
        })}
      </div>

      {arc.length > 0 && (
        <div style={styles.arcRow}>
          {arc.map((m, i) => {
            const c = moodColor(m);
            return (
              <div key={m} style={styles.arcItem}>
                {i > 0 && <span style={{ color: "#444", fontSize: 18, margin: "0 4px" }}>→</span>}
                <div style={{ ...styles.arcChip, background: c.accent + "22", border: `1px solid ${c.accent}`, color: c.accent }}>
                  <button onClick={() => moveUp(i)} style={styles.arcArrow} disabled={i===0}>↑</button>
                  <span style={{ fontSize: 13, fontWeight: 500 }}>{m}</span>
                  <button onClick={() => moveDown(i)} style={styles.arcArrow} disabled={i===arc.length-1}>↓</button>
                  <button onClick={() => remove(i)} style={{ ...styles.arcArrow, color: "#f7604a", marginLeft: 4 }}>✕</button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function TemperatureSlider({ value, onChange }) {
  const labels = ["Safe", "Balanced", "Adventurous"];
  const pct = ((value - 0.1) / (1.5 - 0.1)) * 100;
  return (
    <div>
      <div style={{ ...styles.sectionLabel, display: "flex", justifyContent: "space-between" }}>
        <span>Vibe temperature</span>
        <span style={{ color: "#7c6af7", fontVariantNumeric: "tabular-nums" }}>{value.toFixed(1)}</span>
      </div>
      <input type="range" min="0.1" max="1.5" step="0.1" value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={styles.slider} />
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
        {labels.map(l => <span key={l} style={{ fontSize: 11, color: "#555" }}>{l}</span>)}
      </div>
    </div>
  );
}

function LengthPicker({ value, onChange }) {
  return (
    <div>
      <div style={{ ...styles.sectionLabel, display: "flex", justifyContent: "space-between" }}>
        <span>Playlist length</span>
        <span style={{ color: "#7c6af7" }}>{value} songs</span>
      </div>
      <input type="range" min="5" max="50" step="1" value={value}
        onChange={e => onChange(parseInt(e.target.value))}
        style={styles.slider} />
    </div>
  );
}

function MoodBar({ scores }) {
  const sorted = Object.entries(scores).sort((a,b) => b[1]-a[1]).slice(0,3);
  return (
    <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginTop: 4 }}>
      {sorted.map(([mood, score]) => (
        <span key={mood} style={{
          fontSize: 11, padding: "2px 7px", borderRadius: 100,
          background: moodColor(mood).accent + "18",
          color: moodColor(mood).accent,
          border: `1px solid ${moodColor(mood).accent}33`,
        }}>
          {mood.split(" ")[0]} {Math.round(score * 100)}%
        </span>
      ))}
    </div>
  );
}

function TransitionBar({ score }) {
  if (score == null) return null;
  const pct = Math.round(score * 100);
  const color = pct >= 70 ? "#4af7a0" : pct >= 50 ? "#f7c94a" : "#f7604a";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 6 }}>
      <div style={{ flex: 1, height: 2, background: "rgba(255,255,255,0.06)", borderRadius: 1 }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 1, transition: "width 0.4s" }} />
      </div>
      <span style={{ fontSize: 11, color, minWidth: 28 }}>{pct}%</span>
    </div>
  );
}

function PlaylistCard({ song, index, isFirst }) {
  const top = song.top_mood;
  const c = moodColor(top);
  return (
    <div style={{
      ...styles.songCard,
      borderLeft: `2px solid ${c.accent}44`,
      animationDelay: `${index * 40}ms`,
    }}>
      <div style={styles.songIndex}>{index + 1}</div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={styles.songName}>{song.name}</div>
        <div style={styles.songArtist}>{song.artist}</div>
        <MoodBar scores={song.mood_scores} />
        {!isFirst && <TransitionBar score={song.transition_score} />}
      </div>
      {!isFirst && (
        <div style={{ ...styles.transScore, color: song.transition_score >= 0.7 ? "#4af7a0" : song.transition_score >= 0.5 ? "#f7c94a" : "#f7604a" }}>
          {song.transition_score != null ? Math.round(song.transition_score * 100) : "—"}
        </div>
      )}
    </div>
  );
}

function ArcViz({ arc, playlist }) {
  if (!arc.length || !playlist.length) return null;
  const zones = arc.map((mood, i) => {
    const count = playlist.filter(s => s.top_mood === mood).length;
    return { mood, count };
  });
  return (
    <div style={{ display: "flex", gap: 2, marginBottom: 16, borderRadius: 8, overflow: "hidden", height: 6 }}>
      {zones.map(({ mood, count }, i) => (
        <div key={mood} title={`${mood}: ${count} songs`}
          style={{ flex: count || 1, background: moodColor(mood).accent, opacity: 0.7 }} />
      ))}
    </div>
  );
}

// ── Spotify helpers ───────────────────────────────────────────────────────────

function SpotifyButton({ playlist, spotifyToken, onToken }) {
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(null);
  const [saveError, setSaveError] = useState(null);

  function connectSpotify() {
    const popup = window.open(
      `${API}/spotify/login`,
      "spotify_auth",
      "width=480,height=640,left=200,top=100"
    );
    function onMessage(e) {
      if (e.data?.type === "SPOTIFY_AUTH_SUCCESS") {
        onToken(e.data.access_token);
        window.removeEventListener("message", onMessage);
        popup?.close();
      } else if (e.data?.type === "SPOTIFY_AUTH_ERROR") {
        setSaveError("Spotify login failed: " + e.data.error);
        window.removeEventListener("message", onMessage);
      }
    }
    window.addEventListener("message", onMessage);
  }

  async function savePlaylist() {
    if (!spotifyToken || !playlist) return;
    setSaving(true);
    setSaveError(null);
    try {
      const res = await apiFetch("/spotify/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          access_token: spotifyToken,
          songs:        playlist.playlist,
          arc:          playlist.arc,
          seed:         playlist.seed,
          temperature:  playlist.temperature,
        }),
      });
      setSaved(res);
    } catch(e) {
      setSaveError(e.message);
      if (e.message.includes("401")) onToken(null); // token expired
    }
    setSaving(false);
  }

  if (saved) return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <a href={saved.playlist_url} target="_blank" rel="noreferrer"
        style={{ ...styles.spotifyBtn, background: "#1DB954", color: "#fff", textDecoration: "none", border: "none" }}>
        ▶ Open in Spotify ({saved.added} songs)
      </a>
      <button onClick={() => setSaved(null)} style={{ ...styles.regenBtn, fontSize: 12 }}>Save another</button>
    </div>
  );

  return (
    <div>
      {!spotifyToken ? (
        <button onClick={connectSpotify} style={styles.spotifyBtn}>
          <span style={{ color: "#1DB954", marginRight: 6 }}>♫</span>
          Connect Spotify
        </button>
      ) : (
        <button onClick={savePlaylist} disabled={saving} style={{ ...styles.spotifyBtn, borderColor: "#1DB954", color: "#1DB954" }}>
          {saving ? "Saving…" : "＋ Add to Spotify"}
        </button>
      )}
      {saveError && <div style={{ fontSize: 12, color: "#f7604a", marginTop: 6 }}>{saveError}</div>}
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [moods, setMoods] = useState([]);
  const [seed, setSeed] = useState(null);
  const [arc, setArc] = useState([]);
  const [temperature, setTemperature] = useState(0.8);
  const [length, setLength] = useState(20);
  const [playlist, setPlaylist] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [spotifyToken, setSpotifyToken] = useState(null);
  const playlistRef = useRef(null);

  useEffect(() => {
    apiFetch("/moods").then(d => setMoods(d.moods || [])).catch(() => {});
  }, []);

  async function generate() {
    if (!arc.length) { setError("Pick at least one mood."); return; }
    setLoading(true);
    setError(null);
    try {
      const body = { mood_arc: arc, temperature, playlist_len: length };
      if (seed) body.seed_track_id = seed.id;
      const data = await apiFetch("/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      setPlaylist(data);
      setTimeout(() => playlistRef.current?.scrollIntoView({ behavior: "smooth" }), 100);
    } catch(e) {
      setError(e.message);
    }
    setLoading(false);
  }

  const ready = arc.length > 0;

  return (
    <div style={styles.root}>
      {/* Header */}
      <div style={styles.header}>
        <div style={styles.logo}>◈</div>
        <div>
          <h1 style={styles.title}>Smart Shuffle</h1>
          <p style={styles.subtitle}>Mood-aware playlist sequencer</p>
        </div>
      </div>

      {/* Controls */}
      <div style={styles.panel}>
        <div style={styles.panelSection}>
          <div style={styles.sectionLabel}>Seed song <span style={styles.hint}>(optional — or we pick one)</span></div>
          <SearchBar onSelect={setSeed} />
        </div>

        <div style={styles.divider} />

        <div style={styles.panelSection}>
          <MoodArcPicker moods={moods} arc={arc} setArc={setArc} />
        </div>

        <div style={styles.divider} />

        <div style={styles.panelSection}>
          <TemperatureSlider value={temperature} onChange={setTemperature} />
        </div>

        <div style={styles.panelSection}>
          <LengthPicker value={length} onChange={setLength} />
        </div>

        {error && <div style={styles.error}>{error}</div>}

        <button onClick={generate} disabled={!ready || loading} style={{
          ...styles.generateBtn,
          opacity: ready && !loading ? 1 : 0.4,
          cursor: ready && !loading ? "pointer" : "not-allowed",
        }}>
          {loading ? "Generating…" : "Generate playlist →"}
        </button>
      </div>

      {/* Playlist */}
      {playlist && (
        <div ref={playlistRef} style={styles.playlistSection}>
          <div style={styles.playlistHeader}>
            <div>
              <div style={styles.playlistTitle}>{playlist.total} songs</div>
              <div style={styles.playlistMeta}>
                {playlist.arc.join(" → ")} · temp {playlist.temperature.toFixed(1)}
                {playlist.seed && ` · seed: ${playlist.seed}`}
              </div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 8 }}>
            <button onClick={generate} style={styles.regenBtn} disabled={loading}>
              {loading ? "…" : "↻ Regenerate"}
            </button>
            <SpotifyButton
              playlist={playlist}
              spotifyToken={spotifyToken}
              onToken={setSpotifyToken}
            />
          </div>
          </div>

          <ArcViz arc={playlist.arc} playlist={playlist.playlist} />

          <div style={styles.columnHeaders}>
            <span style={{ width: 28 }}>#</span>
            <span style={{ flex: 1 }}>Song</span>
            <span style={{ fontSize: 11, color: "#555" }}>flow</span>
          </div>

          <div style={styles.songList}>
            {playlist.playlist.map((song, i) => (
              <PlaylistCard key={song.id} song={song} index={i} isFirst={i === 0} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = {
  root: {
    minHeight: "100vh",
    background: "#0d1117",
    color: "#e0e0e0",
    fontFamily: "'DM Sans', 'Helvetica Neue', sans-serif",
    padding: "0 0 80px",
  },
  header: {
    display: "flex",
    alignItems: "center",
    gap: 16,
    padding: "32px 24px 24px",
    borderBottom: "1px solid rgba(255,255,255,0.06)",
  },
  logo: {
    fontSize: 36,
    color: "#7c6af7",
    lineHeight: 1,
  },
  title: {
    margin: 0,
    fontSize: 22,
    fontWeight: 600,
    letterSpacing: "-0.5px",
    color: "#f0f0f0",
  },
  subtitle: {
    margin: "2px 0 0",
    fontSize: 13,
    color: "#555",
  },
  panel: {
    margin: "24px 24px 0",
    background: "rgba(255,255,255,0.03)",
    border: "1px solid rgba(255,255,255,0.07)",
    borderRadius: 16,
    overflow: "hidden",
  },
  panelSection: {
    padding: "20px 20px",
  },
  divider: {
    height: 1,
    background: "rgba(255,255,255,0.05)",
  },
  sectionLabel: {
    fontSize: 13,
    fontWeight: 500,
    color: "#888",
    marginBottom: 12,
    textTransform: "uppercase",
    letterSpacing: "0.05em",
  },
  hint: {
    fontWeight: 400,
    textTransform: "none",
    letterSpacing: 0,
    color: "#444",
    fontSize: 11,
  },
  searchWrap: {
    display: "flex",
    alignItems: "center",
    background: "rgba(255,255,255,0.05)",
    border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: 10,
    padding: "0 12px",
    gap: 8,
  },
  searchIcon: {
    color: "#555",
    fontSize: 16,
    flexShrink: 0,
  },
  searchInput: {
    flex: 1,
    background: "transparent",
    border: "none",
    outline: "none",
    color: "#e0e0e0",
    fontSize: 14,
    padding: "11px 0",
    fontFamily: "inherit",
  },
  searchSpinner: {
    color: "#555",
    fontSize: 14,
    animation: "spin 1s linear infinite",
  },
  clearBtn: {
    background: "none",
    border: "none",
    color: "#555",
    cursor: "pointer",
    fontSize: 13,
    padding: 0,
  },
  dropdown: {
    position: "absolute",
    top: "calc(100% + 4px)",
    left: 0,
    right: 0,
    background: "#161b22",
    border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: 10,
    zIndex: 100,
    boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
    overflow: "hidden",
  },
  dropdownItem: {
    padding: "10px 16px",
    cursor: "pointer",
    borderBottom: "1px solid rgba(255,255,255,0.05)",
    transition: "background 0.1s",
  },
  dropdownName: {
    fontSize: 14,
    color: "#e0e0e0",
    fontWeight: 500,
  },
  dropdownMeta: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    marginTop: 3,
  },
  dropdownArtist: {
    fontSize: 12,
    color: "#666",
    flex: 1,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  moodPill: {
    fontSize: 11,
    padding: "2px 8px",
    borderRadius: 100,
    fontWeight: 500,
    flexShrink: 0,
  },
  moodGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))",
    gap: 8,
    marginBottom: 16,
  },
  moodChip: {
    display: "flex",
    alignItems: "center",
    padding: "8px 12px",
    borderRadius: 8,
    fontSize: 13,
    cursor: "pointer",
    transition: "all 0.15s",
    fontFamily: "inherit",
    textAlign: "left",
  },
  arcRow: {
    display: "flex",
    alignItems: "center",
    flexWrap: "wrap",
    gap: 4,
    marginTop: 4,
  },
  arcItem: {
    display: "flex",
    alignItems: "center",
  },
  arcChip: {
    display: "flex",
    alignItems: "center",
    gap: 4,
    padding: "4px 8px",
    borderRadius: 100,
    fontSize: 12,
  },
  arcArrow: {
    background: "none",
    border: "none",
    cursor: "pointer",
    color: "inherit",
    fontSize: 11,
    padding: "0 2px",
    opacity: 0.7,
    fontFamily: "inherit",
  },
  slider: {
    width: "100%",
    accentColor: "#7c6af7",
    cursor: "pointer",
  },
  error: {
    margin: "0 20px 16px",
    padding: "10px 14px",
    background: "rgba(247,96,74,0.1)",
    border: "1px solid rgba(247,96,74,0.3)",
    borderRadius: 8,
    color: "#f7604a",
    fontSize: 13,
  },
  generateBtn: {
    display: "block",
    width: "100%",
    padding: "14px 20px",
    background: "#7c6af7",
    border: "none",
    color: "#fff",
    fontSize: 15,
    fontWeight: 600,
    fontFamily: "inherit",
    letterSpacing: "-0.2px",
    transition: "opacity 0.15s",
  },
  playlistSection: {
    margin: "24px 24px 0",
  },
  playlistHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: 16,
  },
  playlistTitle: {
    fontSize: 20,
    fontWeight: 600,
    color: "#f0f0f0",
  },
  playlistMeta: {
    fontSize: 12,
    color: "#555",
    marginTop: 2,
  },
  spotifyBtn: {
  display: "flex",
  alignItems: "center",
  background: "none",
  border: "1px solid rgba(255,255,255,0.15)",
  color: "#ccc",
  borderRadius: 8,
  padding: "6px 14px",
  fontSize: 13,
  cursor: "pointer",
  fontFamily: "inherit",
  transition: "all 0.15s",
  whiteSpace: "nowrap",
},

regenBtn: {
  background: "none",
  border: "1px solid rgba(255,255,255,0.1)",
  color: "#888",
  borderRadius: 8,
  padding: "6px 14px",
  fontSize: 13,
  cursor: "pointer",
  fontFamily: "inherit",
  transition: "all 0.15s",
},
  columnHeaders: {
    display: "flex",
    alignItems: "center",
    gap: 12,
    padding: "0 12px 8px",
    fontSize: 11,
    color: "#444",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
  },
  songList: {
    display: "flex",
    flexDirection: "column",
    gap: 2,
  },
  songCard: {
    display: "flex",
    alignItems: "flex-start",
    gap: 12,
    padding: "12px",
    borderRadius: 10,
    background: "rgba(255,255,255,0.02)",
    border: "1px solid rgba(255,255,255,0.05)",
    animation: "fadeUp 0.3s both",
    transition: "background 0.15s",
  },
  songIndex: {
    fontSize: 12,
    color: "#444",
    width: 20,
    textAlign: "right",
    paddingTop: 2,
    flexShrink: 0,
    fontVariantNumeric: "tabular-nums",
  },
  songName: {
    fontSize: 14,
    fontWeight: 500,
    color: "#e0e0e0",
    whiteSpace: "nowrap",
    overflow: "hidden",
    textOverflow: "ellipsis",
  },
  songArtist: {
    fontSize: 12,
    color: "#666",
    marginTop: 1,
    whiteSpace: "nowrap",
    overflow: "hidden",
    textOverflow: "ellipsis",
  },
  transScore: {
    fontSize: 12,
    fontWeight: 600,
    fontVariantNumeric: "tabular-nums",
    flexShrink: 0,
    paddingTop: 2,
    minWidth: 24,
    textAlign: "right",
  },
};