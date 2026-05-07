"""
utils/ui_utils.py
Accessibility-first CSS theming and reusable HTML renderers.
"""
import streamlit as st


def apply_accessibility_css():
    st.markdown("""
    <style>
    /* ── Base ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body, .stApp {
        background-color: #0A0E1A !important;
        color: #F0F2F8 !important;
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0D1117 !important;
        border-right: 1px solid #1E2533;
    }
    section[data-testid="stSidebar"] * { color: #C9D1D9 !important; }

    /* ── Hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #1A237E 0%, #006064 100%);
        border-radius: 14px;
        padding: 1.4rem 2rem;
        margin-bottom: 1.2rem;
        border: 1px solid #283593;
        box-shadow: 0 8px 32px rgba(26,35,126,0.4);
    }
    .hero-banner h1 {
        margin: 0; font-size: 1.9rem; font-weight: 800;
        color: #FFFFFF; letter-spacing: -0.5px;
    }
    .hero-banner p { margin: 0.3rem 0 0 0; color: #90CAF9; font-size: 0.95rem; }

    /* ── Live camera panel ── */
    .cam-panel {
        border: 2px solid #1E2533;
        border-radius: 12px;
        overflow: hidden;
        background: #0D1117;
        min-height: 320px;
    }

    /* ── Gesture indicator ── */
    .gesture-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.03em;
        margin: 4px 4px 4px 0;
    }
    .badge-listen  { background:#1B5E20; color:#A5D6A7; border:1px solid #2E7D32; }
    .badge-train   { background:#E65100; color:#FFE0B2; border:1px solid #BF360C; }
    .badge-fist    { background:#4A148C; color:#E1BEE7; border:1px solid #6A1B9A; }
    .badge-ready   { background:#1A237E; color:#BBDEFB; border:1px solid #283593; }
    .badge-error   { background:#B71C1C; color:#FFCDD2; border:1px solid #C62828; }

    /* ── Word bubbles ── */
    .word-strip { display:flex; flex-wrap:wrap; gap:8px; margin:8px 0; }
    .word-bubble {
        background: linear-gradient(135deg, #1565C0, #0277BD);
        color: #FFFFFF;
        padding: 10px 20px;
        border-radius: 28px;
        font-size: 1.3rem;
        font-weight: 700;
        box-shadow: 0 3px 10px rgba(21,101,192,0.4);
        border: 1px solid #1976D2;
        letter-spacing: 0.03em;
    }

    /* ── Sentence suggestion box ── */
    .suggestion-box {
        background: #0D1117;
        border-left: 5px solid #00E676;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0;
        font-size: 1.1rem;
        font-weight: 500;
        color: #E8F5E9;
        line-height: 1.5;
        box-shadow: 0 2px 8px rgba(0,230,118,0.1);
    }

    /* ── Status panel ── */
    .status-ok   { color:#69F0AE; font-size:0.92rem; font-weight:600; }
    .status-warn { color:#FFD740; font-size:0.92rem; font-weight:600; }
    .status-err  { color:#FF5252; font-size:0.92rem; font-weight:600; }

    /* ── Memory bank chip ── */
    .mem-chip {
        display:inline-block;
        background:#1C2A3A;
        color:#4FC3F7;
        border:1px solid #0288D1;
        border-radius:14px;
        padding:3px 12px;
        font-size:0.82rem;
        margin:3px;
        font-weight:600;
    }

    /* ── Confidence meter ── */
    .conf-track { background:#1E2533; border-radius:6px; height:8px; margin-top:4px; }
    .conf-fill  { height:8px; border-radius:6px; transition: width .3s ease; }

    /* ── Pulse animation for active listening ── */
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.45} }
    .pulse { animation: pulse 1.1s ease-in-out infinite; }

    /* ── Section header ── */
    .sec-header {
        font-size:0.75rem;
        font-weight:700;
        letter-spacing:0.12em;
        color:#546E7A;
        text-transform:uppercase;
        margin:1rem 0 0.4rem 0;
    }

    /* Streamlit overrides */
    div[data-testid="stMetricValue"] { color:#4FC3F7 !important; font-weight:800; }
    .stButton button {
        border-radius:10px !important;
        font-weight:700 !important;
        border:1px solid #283593 !important;
    }
    div[data-testid="stTextInput"] input { background:#0D1117 !important; color:#F0F2F8 !important; }
    </style>
    """, unsafe_allow_html=True)


# ── Render helpers ────────────────────────────────────────────────────────────

def render_hero():
    st.markdown("""
    <div class="hero-banner">
        <h1>🤫 Silent Speech AI &nbsp;<span style="font-size:1rem;font-weight:400;opacity:.7">HACKHIVE-2k26</span></h1>
        <p>State-triggered KNN lip recognition · Gesture-to-LLM sentence generation · Multi-language accessibility</p>
    </div>
    """, unsafe_allow_html=True)


def render_gesture_badge(state: str) -> str:
    """Return coloured badge HTML for current gesture state."""
    badges = {
        "READY":    ('<span class="gesture-badge badge-ready">⬤ READY</span>', ""),
        "LISTEN":   ('<span class="gesture-badge badge-listen pulse">☝️ LISTENING</span>', "Raise index finger · mouth your word · lower finger to confirm"),
        "TRAINING": ('<span class="gesture-badge badge-train pulse">📝 TRAINING</span>', "Keep finger raised · mouth the word · lower to save"),
        "TRANSLATE":('<span class="gesture-badge badge-fist">✊ TRANSLATING…</span>', "Sending words to Gemini"),
        "ERROR":    ('<span class="gesture-badge badge-error">⚠ CAM ERROR</span>', "Check camera permissions / index"),
    }
    badge, subtitle = badges.get(state, badges["READY"])
    html = badge
    if subtitle:
        html += f"&nbsp;<span style='color:#78909C;font-size:0.8rem;'>{subtitle}</span>"
    return html


def render_word_strip(words: list) -> str:
    if not words:
        return "<p style='color:#37474F;font-style:italic;'>No words captured yet…</p>"
    chips = "".join(f"<span class='word-bubble'>{w}</span>" for w in words)
    return f"<div class='word-strip'>{chips}</div>"


def render_memory_chips(memory: dict) -> str:
    if not memory:
        return "<p style='color:#37474F;font-size:0.85rem;'>Empty — use sidebar to teach words</p>"
    return "".join(f"<span class='mem-chip'>{w}</span>" for w in sorted(memory.keys()))


def render_suggestions(sentences: list) -> str:
    if not sentences:
        return ""
    return "".join(f"<div class='suggestion-box'>💬 {s}</div>" for s in sentences)


def render_confidence(label: str, value: float, color: str = "#00E676"):
    pct = int(max(0.0, min(1.0, value)) * 100)
    return f"""
    <div style='margin:6px 0;'>
        <div style='display:flex;justify-content:space-between;'>
            <span style='font-size:0.78rem;color:#546E7A;'>{label}</span>
            <span style='font-size:0.78rem;color:{color};font-weight:700;'>{pct}%</span>
        </div>
        <div class='conf-track'>
            <div class='conf-fill' style='width:{pct}%;background:{color};'></div>
        </div>
    </div>"""
