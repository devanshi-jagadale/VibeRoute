import { useState, useEffect, useRef, useCallback } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const FALLBACK_MOODS = [
  "Upbeat Dance",
  "Energetic Anthems",
  "Mellow Reflections",
  "High Energy Party",
  "Dynamic Beats",
  "Experimental Vibes",
  "Folk Inspired",
  "Indie Delights",
];

const MOOD_COLORS = {
  "Upbeat Dance":       { accent: "#e6c84a", glow: "#e6c84a33" },
  "Energetic Anthems":  { accent: "#e8614a", glow: "#e8614a33" },
  "Mellow Reflections": { accent: "#c8a96e", glow: "#c8a96e33" },
  "High Energy Party":  { accent: "#e8614a", glow: "#e8614a33" },
  "Dynamic Beats":      { accent: "#5ab4e8", glow: "#5ab4e833" },
  "Experimental Vibes": { accent: "#c86eb4", glow: "#c86eb433" },
  "Folk Inspired":      { accent: "#d4845a", glow: "#d4845a33" },
  "Indie Delights":     { accent: "#8fd45a", glow: "#8fd45a33" },
};

const DEFAULT_COLOR = { accent: "#c8a96e", glow: "#c8a96e33" };
function moodColor(mood) { return MOOD_COLORS[mood] || DEFAULT_COLOR; }

async function apiFetch(path, opts = {}) {
  const res = await fetch(API + path, opts);
  if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.detail || `HTTP ${res.status}`); }
  return res.json();
}

// ── Search ────────────────────────────────────────────────────────────────────
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
      try { const data = await apiFetch(`/search?q=${encodeURIComponent(query)}`); setResults(data.results || []); setOpen(true); }
      catch { setResults([]); }
      setLoading(false);
    }, 350);
  }, []);

  useEffect(() => { search(q); }, [q, search]);
  useEffect(() => {
    function handler(e) { if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false); }
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  function pick(song) { setQ(song.name + " — " + song.artist); setOpen(false); onSelect(song); }

  return (
    <div ref={wrapRef} style={{ position: "relative" }}>
      <div style={S.searchWrap}>
        <svg width="13" height="13" viewBox="0 0 14 14" fill="none" style={{ flexShrink: 0, opacity: 0.35 }}>
          <circle cx="5.5" cy="5.5" r="4.5" stroke="#c8a96e" strokeWidth="1.5"/>
          <path d="M9 9L12.5 12.5" stroke="#c8a96e" strokeWidth="1.5" strokeLinecap="round"/>
        </svg>
        <input value={q} onChange={e => setQ(e.target.value)} placeholder="Search seed song…"
          style={S.searchInput} onFocus={() => results.length && setOpen(true)} />
        {loading && <div style={S.spinner} />}
        {q && <button onClick={() => { setQ(""); setResults([]); setOpen(false); onSelect(null); }} style={S.clearBtn}>✕</button>}
      </div>
      {open && results.length > 0 && (
        <div style={S.dropdown}>
          {results.map((s, idx) => (
            <div key={s.id}
              style={S.dropdownItem}
              onMouseEnter={e => e.currentTarget.style.background = "rgba(200,169,110,0.07)"}
              onMouseLeave={e => e.currentTarget.style.background = "transparent"}
              onClick={() => pick(s)}>
              <span style={S.dropdownNum}>{String(idx+1).padStart(2,"0")}</span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={S.dropdownName}>{s.name}</div>
                <div style={S.dropdownArtist}>{s.artist}</div>
              </div>
              <span style={{ ...S.moodTag, background: moodColor(s.top_mood).glow, color: moodColor(s.top_mood).accent }}>
                {s.top_mood?.split(" ")[0]}
              </span>
            </div>
          ))}
        </div>
      )}
      {open && q && !results.length && !loading && (
        <div style={S.dropdown}>
          <div style={{ padding: "14px 16px", color: "#5a5040", fontSize: 12, fontStyle: "italic" }}>Nothing found</div>
        </div>
      )}
    </div>
  );
}

// ── Mood List ─────────────────────────────────────────────────────────────────
function MoodList({ moods, arc, setArc }) {
  function toggle(mood) {
    if (arc.includes(mood)) setArc(arc.filter(m => m !== mood));
    else if (arc.length < 3) setArc([...arc, mood]);
  }
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {moods.map(m => {
        const sel = arc.includes(m);
        const c = moodColor(m);
        return (
          <button key={m} onClick={() => toggle(m)} style={{
            display: "flex", alignItems: "center", gap: 10,
            padding: "9px 12px",
            borderRadius: 7,
            border: `1px solid ${sel ? c.accent + "55" : "rgba(255,255,255,0.06)"}`,
            background: sel ? c.glow : "rgba(255,255,255,0.02)",
            color: sel ? c.accent : "#9a8878",
            fontSize: 13, fontWeight: sel ? 600 : 400,
            fontFamily: "'Syne', sans-serif",
            cursor: "pointer", textAlign: "left",
            transition: "all 0.18s",
            boxShadow: sel ? `0 0 10px ${c.glow}` : "none",
          }}>
            <span style={{
              width: 7, height: 7, borderRadius: "50%", flexShrink: 0,
              background: sel ? c.accent : "transparent",
              border: `1.5px solid ${sel ? c.accent : "#3a3020"}`,
              transition: "all 0.18s",
            }} />
            {m}
          </button>
        );
      })}
    </div>
  );
}

// ── Arc Display ───────────────────────────────────────────────────────────────
function ArcDisplay({ arc, setArc }) {
  function remove(i) { setArc(arc.filter((_,j) => j !== i)); }
  if (!arc.length) return (
    <div style={{ fontSize: 12, color: "#3a3020", fontStyle: "italic", padding: "4px 0" }}>
      Pick moods above →
    </div>
  );
  return (
    <div style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 4 }}>
      {arc.map((m, i) => {
        const c = moodColor(m);
        return (
          <div key={m} style={{ display: "flex", alignItems: "center", gap: 3 }}>
            {i > 0 && <span style={{ color: "#3a3020", fontSize: 12, margin: "0 1px" }}>→</span>}
            <span style={{
              display: "flex", alignItems: "center", gap: 4,
              padding: "3px 8px 3px 7px",
              borderRadius: 5,
              background: c.glow,
              border: `1px solid ${c.accent}44`,
              color: c.accent,
              fontSize: 11, fontWeight: 600, letterSpacing: "0.02em",
            }}>
              {m.split(" ")[0]}
              <button onClick={() => remove(i)} style={{ background: "none", border: "none", color: c.accent, cursor: "pointer", fontSize: 10, padding: 0, opacity: 0.6, lineHeight: 1, fontFamily: "inherit" }}>×</button>
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ── Sliders ───────────────────────────────────────────────────────────────────
function TrackSlider({ label, value, display, min, max, step, onChange, pct }) {
  return (
    <div style={{ flex: 1 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
        <span style={S.sliderLabel}>{label}</span>
        <span style={S.sliderVal}>{display}</span>
      </div>
      <div style={{ position: "relative", height: 3, background: "rgba(255,255,255,0.05)", borderRadius: 2 }}>
        <div style={{ position: "absolute", left: 0, top: 0, height: "100%", width: `${pct}%`, background: "linear-gradient(90deg, #c8a96e66, #c8a96e)", borderRadius: 2 }} />
        <div style={{ position: "absolute", top: "50%", left: `${pct}%`, transform: "translate(-50%,-50%)", width: 11, height: 11, borderRadius: "50%", background: "#c8a96e", boxShadow: "0 0 6px #c8a96e88" }} />
        <input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(parseFloat(e.target.value))}
          style={{ position: "absolute", inset: "-10px 0", height: 24, opacity: 0, cursor: "pointer", width: "100%" }} />
      </div>
    </div>
  );
}

// ── Language ──────────────────────────────────────────────────────────────────
const LANGUAGES = [
  { code: null, label: "All languages" },
  { code: "en", label: "English" },
  { code: "hi", label: "Hindi" },
  { code: "ja", label: "Japanese" },
  { code: "ko", label: "Korean" },
  { code: "fr", label: "French" },
  { code: "pt", label: "Portuguese" },
  { code: "es", label: "Spanish" },
];

// ── Playlist helpers ──────────────────────────────────────────────────────────
function MoodBar({ scores }) {
  const sorted = Object.entries(scores).sort((a,b) => b[1]-a[1]).slice(0,3);
  return (
    <div style={{ display: "flex", gap: 3, flexWrap: "wrap", marginTop: 4 }}>
      {sorted.map(([mood, score]) => {
        const c = moodColor(mood);
        return (
          <span key={mood} style={{ fontSize: 9, padding: "1px 6px", borderRadius: 3, background: c.glow, color: c.accent, letterSpacing: "0.05em" }}>
            {mood.split(" ")[0].toUpperCase()} {Math.round(score*100)}%
          </span>
        );
      })}
    </div>
  );
}

function TransitionBar({ score }) {
  if (score == null) return null;
  const pct = Math.round(score * 100);
  const color = pct >= 70 ? "#5ecfa0" : pct >= 50 ? "#e6c84a" : "#e8614a";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 5 }}>
      <div style={{ flex: 1, height: 1.5, background: "rgba(255,255,255,0.04)", borderRadius: 1 }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 1, transition: "width 0.5s ease" }} />
      </div>
      <span style={{ fontSize: 9, color, minWidth: 24, textAlign: "right", fontFamily: "'DM Mono', monospace", fontVariantNumeric: "tabular-nums" }}>{pct}%</span>
    </div>
  );
}

function PlaylistCard({ song, index, isFirst, delay = 0 }) {
  const [hov, setHov] = useState(false);
  const c = moodColor(song.top_mood);
  return (
    <div onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)} style={{
      display: "flex", alignItems: "flex-start", gap: 12,
      padding: "10px 14px",
      borderLeft: `3px solid ${hov ? c.accent : c.accent + "22"}`,
      borderBottom: "1px solid rgba(200,169,110,0.04)",
      background: hov ? "rgba(200,169,110,0.03)" : "transparent",
      transition: "all 0.18s",
      animation: "fadeUp 0.3s both",
      animationDelay: `${delay}ms`,
    }}>
      <div style={{ fontSize: 10, color: "#2a2010", width: 20, textAlign: "right", paddingTop: 2, flexShrink: 0, fontFamily: "'DM Mono', monospace" }}>
        {String(index+1).padStart(2,"0")}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: "#e0d4c0", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", letterSpacing: "-0.1px" }}>{song.name}</div>
        <div style={{ fontSize: 11, color: "#5a5040", marginTop: 1, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", fontFamily: "'DM Mono', monospace" }}>{song.artist}</div>
        <MoodBar scores={song.mood_scores} />
        {!isFirst && <TransitionBar score={song.transition_score} />}
      </div>
      {!isFirst && song.transition_score != null && (
        <div style={{ fontSize: 11, fontWeight: 700, fontFamily: "'DM Mono', monospace", flexShrink: 0, paddingTop: 2, minWidth: 22, textAlign: "right", color: song.transition_score >= 0.7 ? "#5ecfa0" : song.transition_score >= 0.5 ? "#e6c84a" : "#e8614a" }}>
          {Math.round(song.transition_score*100)}
        </div>
      )}
    </div>
  );
}

function ArcViz({ arc, playlist }) {
  if (!arc.length || !playlist.length) return null;
  const zones = arc.map(mood => ({ mood, count: playlist.filter(s => s.top_mood === mood).length }));
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: "flex", gap: 2, height: 3, borderRadius: 2, overflow: "hidden" }}>
        {zones.map(({ mood, count }) => <div key={mood} style={{ flex: count || 1, background: moodColor(mood).accent, opacity: 0.55 }} />)}
      </div>
      <div style={{ display: "flex", gap: 3, marginTop: 5 }}>
        {zones.map(({ mood, count }) => (
          <div key={mood} style={{ flex: count || 1, display: "flex", alignItems: "center", gap: 3 }}>
            <div style={{ width: 5, height: 5, borderRadius: "50%", background: moodColor(mood).accent, flexShrink: 0 }} />
            <span style={{ fontSize: 9, color: moodColor(mood).accent, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", letterSpacing: "0.05em" }}>
              {mood.split(" ")[0].toUpperCase()}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Spotify ───────────────────────────────────────────────────────────────────
function SpotifyButton({ playlist, spotifyToken, onToken }) {
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(null);
  const [err, setErr] = useState(null);

  function connectSpotify() {
    const popup = window.open(`${API}/spotify/login`, "spotify_auth", "width=480,height=640,left=200,top=100");
    function onMsg(e) {
      if (e.data?.type === "SPOTIFY_AUTH_SUCCESS") { onToken(e.data.access_token); window.removeEventListener("message", onMsg); popup?.close(); }
      else if (e.data?.type === "SPOTIFY_AUTH_ERROR") { setErr("Login failed"); window.removeEventListener("message", onMsg); }
    }
    window.addEventListener("message", onMsg);
  }

  async function savePlaylist() {
    if (!spotifyToken || !playlist) return;
    setSaving(true); setErr(null);
    try {
      const res = await apiFetch("/spotify/save", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ access_token: spotifyToken, songs: playlist.playlist, arc: playlist.arc, seed: playlist.seed, temperature: playlist.temperature }) });
      setSaved(res);
    } catch(e) { setErr(e.message); if (e.message.includes("401")) onToken(null); }
    setSaving(false);
  }

  if (saved) return (
    <a href={saved.playlist_url} target="_blank" rel="noreferrer"
      style={{ display: "flex", alignItems: "center", gap: 6, padding: "6px 14px", borderRadius: 6, background: "rgba(29,185,84,0.12)", border: "1px solid rgba(29,185,84,0.3)", color: "#1DB954", fontSize: 12, fontWeight: 600, fontFamily: "'Syne', sans-serif", textDecoration: "none", whiteSpace: "nowrap" }}>
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#1DB954" }} />
      Open in Spotify
    </a>
  );

  return (
    <div>
      {!spotifyToken
        ? <button onClick={connectSpotify} style={{ display: "flex", alignItems: "center", gap: 6, padding: "6px 14px", borderRadius: 6, background: "none", border: "1px solid rgba(255,255,255,0.1)", color: "#9a8878", fontSize: 12, cursor: "pointer", fontFamily: "'Syne', sans-serif" }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#1DB954" }} /> Connect Spotify
          </button>
        : <button onClick={savePlaylist} disabled={saving} style={{ display: "flex", alignItems: "center", gap: 6, padding: "6px 14px", borderRadius: 6, background: "none", border: "1px solid rgba(29,185,84,0.35)", color: "#1DB954", fontSize: 12, cursor: "pointer", fontFamily: "'Syne', sans-serif", fontWeight: 600 }}>
            {saving ? "Saving…" : "+ Save to Spotify"}
          </button>}
      {err && <div style={{ fontSize: 10, color: "#e8614a", marginTop: 4 }}>{err}</div>}
    </div>
  );
}

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const [moods, setMoods] = useState(FALLBACK_MOODS);
  const [seed, setSeed] = useState(null);
  const [arc, setArc] = useState([]);
  const [temperature, setTemperature] = useState(0.8);
  const [length, setLength] = useState(23);
  const [language, setLanguage] = useState(null);
  const [playlist, setPlaylist] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [spotifyToken, setSpotifyToken] = useState(null);
  const rightRef = useRef(null);

  useEffect(() => { apiFetch("/moods").then(d => setMoods(d.moods || [])).catch(() => {}); }, []);

  async function generate() {
    if (!arc.length) { setError("Pick at least one mood."); return; }
    setLoading(true); setError(null);
    try {
      const body = { mood_arc: arc, temperature, playlist_len: length };
      if (seed) body.seed_track_id = seed.id;
      if (language) body.language = language;
      const data = await apiFetch("/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      setPlaylist(data);
      setTimeout(() => rightRef.current?.scrollTo({ top: 0, behavior: "smooth" }), 100);
    } catch(e) { setError(e.message); }
    setLoading(false);
  }

  const tempPct = ((temperature - 0.1) / (1.5 - 0.1)) * 100;
  const lenPct  = ((length - 5) / (50 - 5)) * 100;

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;1,300&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        html, body, #root { height: 100%; background: #0e0b07; }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeUp { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
        ::-webkit-scrollbar { width: 3px; }
        ::-webkit-scrollbar-thumb { background: #2a2010; border-radius: 2px; }
        button:focus { outline: none; }
      `}</style>

      <div style={S.root}>

        {/* ── LEFT SIDEBAR ── */}
        <div style={S.sidebar}>

          {/* Logo */}
          <div style={S.logoRow}>
            <div style={S.disc}>
              <div style={S.discRing} />
              <div style={S.discCore} />
            </div>
            <div>
              <div style={S.logoName}>VibeRoute</div>
              <div style={S.logoSub}>mood sequencer</div>
            </div>
          </div>

          {/* Search */}
          <div style={{ marginBottom: 20 }}>
            <SearchBar onSelect={setSeed} />
          </div>

          {/* Moods */}
          <div style={{ ...S.sideSection, flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
            <div style={S.sideLabel}>Moods</div>
            <div style={{ overflowY: "auto", flex: 1 }}>
              <MoodList moods={moods} arc={arc} setArc={setArc} />
            </div>
          </div>

          {/* Arc */}
          <div style={{ ...S.sideSection, borderTop: "1px solid rgba(200,169,110,0.06)", paddingTop: 16, marginTop: 8 }}>
            <div style={S.sideLabel}>Your Arc</div>
            <ArcDisplay arc={arc} setArc={setArc} />
          </div>

          {/* Generate */}
          <div style={{ marginTop: "auto", paddingTop: 20 }}>
            {error && <div style={S.error}>{error}</div>}
            <button onClick={generate} disabled={!arc.length || loading} style={{
              ...S.generateBtn,
              opacity: arc.length && !loading ? 1 : 0.35,
              cursor: arc.length && !loading ? "pointer" : "not-allowed",
            }}>
              {loading
                ? <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
                    <span style={{ width: 11, height: 11, border: "1.5px solid #0e0b0755", borderTopColor: "#0e0b07", borderRadius: "50%", display: "inline-block", animation: "spin 0.7s linear infinite" }} />
                    Generating
                  </span>
                : <span>▷ Generate</span>}
            </button>
          </div>
        </div>

        {/* ── RIGHT PANEL ── */}
        <div ref={rightRef} style={S.right}>

          {/* Top controls row */}
          <div style={S.topControls}>
            <div style={{ display: "flex", gap: 24, flex: 1 }}>
              <TrackSlider label="Temperature" value={temperature} display={temperature.toFixed(1)}
                min={0.1} max={1.5} step={0.1} onChange={setTemperature} pct={tempPct} />
              <TrackSlider label="Length" value={length} display={String(length)}
                min={5} max={50} step={1} onChange={setLength} pct={lenPct} />
            </div>
          </div>

          {/* Language pills */}
          <div style={S.langRow}>
            {LANGUAGES.map(l => {
              const active = language === l.code;
              return (
                <button key={l.code ?? "all"} onClick={() => setLanguage(l.code)} style={{
                  padding: "6px 14px", borderRadius: 6, fontSize: 12,
                  fontFamily: "'Syne', sans-serif", fontWeight: active ? 600 : 400,
                  cursor: "pointer",
                  background: active ? "rgba(200,169,110,0.14)" : "rgba(255,255,255,0.03)",
                  border: `1px solid ${active ? "#c8a96e55" : "rgba(255,255,255,0.07)"}`,
                  color: active ? "#c8a96e" : "#5a5040",
                  transition: "all 0.15s",
                  letterSpacing: "0.01em",
                }}>
                  {l.label}
                </button>
              );
            })}
          </div>

          {/* Divider */}
          <div style={{ height: 1, background: "rgba(200,169,110,0.06)", margin: "0 0 0 0" }} />

          {/* Playlist or empty state */}
          {!playlist ? (
            <div style={S.emptyState}>
              <svg width="36" height="36" viewBox="0 0 24 24" fill="none" style={{ opacity: 0.15, marginBottom: 12 }}>
                <path d="M9 18V5l12-2v13" stroke="#c8a96e" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                <circle cx="6" cy="18" r="3" stroke="#c8a96e" strokeWidth="1.5"/>
                <circle cx="18" cy="16" r="3" stroke="#c8a96e" strokeWidth="1.5"/>
              </svg>
              <div style={{ fontSize: 13, color: "#3a3020" }}>Pick moods and hit generate</div>
            </div>
          ) : (
            <div style={{ padding: "20px 24px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
                <div>
                  <div style={{ fontSize: 20, fontWeight: 800, color: "#e0d4c0", letterSpacing: "-0.5px" }}>{playlist.total} Tracks</div>
                  <div style={{ fontSize: 10, color: "#3a3020", marginTop: 3, fontFamily: "'DM Mono', monospace" }}>
                    {playlist.arc.join(" → ")} · {playlist.temperature.toFixed(1)} temp
                    {playlist.seed && ` · ${playlist.seed}`}
                  </div>
                </div>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <SpotifyButton playlist={playlist} spotifyToken={spotifyToken} onToken={setSpotifyToken} />
                  <button onClick={generate} disabled={loading} style={{ padding: "6px 12px", borderRadius: 6, background: "none", border: "1px solid rgba(200,169,110,0.15)", color: "#5a5040", fontSize: 12, cursor: "pointer", fontFamily: "'Syne', sans-serif" }}>
                    ↻ Regen
                  </button>
                </div>
              </div>

              <ArcViz arc={playlist.arc} playlist={playlist.playlist} />

              <div style={{ display: "flex", gap: 12, padding: "0 14px 8px", fontSize: 9, color: "#2a2010", textTransform: "uppercase", letterSpacing: "0.12em", borderBottom: "1px solid rgba(200,169,110,0.06)", marginBottom: 4 }}>
                <span style={{ width: 20 }}>#</span>
                <span style={{ flex: 1 }}>Track</span>
                <span>Flow</span>
              </div>

              {playlist.playlist.map((song, i) => (
                <PlaylistCard key={song.id} song={song} index={i} isFirst={i === 0} delay={i * 30} />
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────
const S = {
  root: {
    display: "flex",
    height: "100vh",
    background: "#0e0b07",
    color: "#e0d4c0",
    fontFamily: "'Syne', sans-serif",
    overflow: "hidden",
  },

  // Sidebar
  sidebar: {
    width: 260,
    flexShrink: 0,
    background: "#080602",
    borderRight: "1px solid rgba(200,169,110,0.07)",
    display: "flex",
    flexDirection: "column",
    padding: "22px 16px",
    height: "100vh",
    overflow: "hidden",
  },
  logoRow: {
    display: "flex", alignItems: "center", gap: 10, marginBottom: 20,
  },
  disc: {
    width: 30, height: 30, borderRadius: "50%",
    background: "rgba(200,169,110,0.08)",
    border: "1.5px solid rgba(200,169,110,0.25)",
    display: "flex", alignItems: "center", justifyContent: "center",
    position: "relative", flexShrink: 0,
  },
  discRing: {
    position: "absolute", inset: 4, borderRadius: "50%",
    background: "rgba(200,169,110,0.1)", border: "1px solid rgba(200,169,110,0.15)",
  },
  discCore: {
    width: 5, height: 5, borderRadius: "50%", background: "#c8a96e", position: "relative", zIndex: 1,
  },
  logoName: { fontSize: 20, fontWeight: 800, color: "#e0d4c0", letterSpacing: "-0.3px" },
  logoSub: { fontSize: 14, color: "#493d2a", letterSpacing: "0.1em", textTransform: "uppercase", marginTop: 1 },

  sideSection: { marginBottom: 16 },
  sideLabel: {
    fontSize: 13, fontWeight: 700, color: "#4d412e",
    textTransform: "uppercase", letterSpacing: "0.12em", marginBottom: 8,
  },

  error: {
    padding: "8px 12px", borderRadius: 6, marginBottom: 10,
    background: "rgba(232,97,74,0.08)", border: "1px solid rgba(232,97,74,0.2)",
    color: "#e8614a", fontSize: 11, fontFamily: "'DM Mono', monospace",
  },
  generateBtn: {
    width: "100%", padding: "11px 16px",
    background: "#c8a96e", border: "none", borderRadius: 8,
    color: "#0e0b07", fontSize: 13, fontWeight: 700,
    fontFamily: "'Syne', sans-serif", letterSpacing: "0.02em",
    transition: "opacity 0.15s",
  },

  // Right
  right: {
    flex: 1, display: "flex", flexDirection: "column",
    overflowY: "auto", height: "100vh",
  },
  topControls: {
    padding: "18px 24px 16px",
    borderBottom: "1px solid rgba(200,169,110,0.06)",
    display: "flex", alignItems: "center", gap: 24,
  },
  sliderLabel: {
    fontSize: 12, fontWeight: 700, color: "#524531",
    textTransform: "uppercase", letterSpacing: "0.12em",
  },
  sliderVal: {
    fontSize: 13, color: "#c8a96e", fontVariantNumeric: "tabular-nums",
    fontWeight: 700, fontFamily: "'DM Mono', monospace",
  },
  langRow: {
    display: "flex", flexWrap: "wrap", gap: 6,
    padding: "14px 24px",
    borderBottom: "1px solid rgba(200,169,110,0.06)",
  },
  emptyState: {
    flex: 1, display: "flex", flexDirection: "column",
    alignItems: "center", justifyContent: "center",
    padding: 40, minHeight: 300,
  },

  // Search
  searchWrap: {
    display: "flex", alignItems: "center",
    background: "rgba(0,0,0,0.35)", border: "1px solid rgba(200,169,110,0.1)",
    borderRadius: 8, padding: "0 11px", gap: 7,
  },
  searchInput: {
    flex: 1, background: "transparent", border: "none", outline: "none",
    color: "#e0d4c0", fontSize: 12, padding: "9px 0",
    fontFamily: "'DM Mono', monospace",
  },
  spinner: {
    width: 11, height: 11, border: "1.5px solid #c8a96e33",
    borderTopColor: "#c8a96e", borderRadius: "50%",
    animation: "spin 0.7s linear infinite", flexShrink: 0,
  },
  clearBtn: {
    background: "none", border: "none", color: "#3a3020",
    cursor: "pointer", fontSize: 11, padding: 0, fontFamily: "inherit",
  },
  dropdown: {
    position: "absolute", top: "calc(100% + 4px)", left: 0, right: 0,
    background: "#100d08", border: "1px solid rgba(200,169,110,0.12)",
    borderRadius: 9, zIndex: 200, boxShadow: "0 16px 40px rgba(0,0,0,0.7)", overflow: "hidden",
  },
  dropdownItem: {
    padding: "9px 12px", cursor: "pointer",
    display: "flex", alignItems: "center", gap: 8,
    transition: "background 0.1s",
  },
  dropdownNum: {
    fontSize: 9, color: "#3a3020", fontFamily: "'DM Mono', monospace", width: 18, textAlign: "right", flexShrink: 0,
  },
  dropdownName: { fontSize: 12, color: "#e0d4c0", fontWeight: 600 },
  dropdownArtist: { fontSize: 10, color: "#5a5040", marginTop: 1, fontFamily: "'DM Mono', monospace" },
  moodTag: {
    fontSize: 9, padding: "2px 5px", borderRadius: 3, fontWeight: 700,
    letterSpacing: "0.06em", flexShrink: 0,
  },
};