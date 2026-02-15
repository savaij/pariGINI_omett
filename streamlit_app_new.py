"""
Streamlit App: pariGINI
Calcola e visualizza il Gini Index per la disuguaglianza di accessibilità in metro a Parigi
"""

import json
import math
import random
import requests
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

import os
import base64
from pathlib import Path

from streamlit_searchbox import st_searchbox  # pip install streamlit-searchbox

from gini_paris_distances_calculations import (
    build_graph_from_edgelist,
    build_node_index,
    accessibility_inequality_to_target,
    accessibility_inequality_to_targets,  # <-- AGGIUNGI
)




# ============================================================
# HELPERS
# ============================================================

def get_viewport_width() -> int:
    """
    Legge la larghezza del viewport via JS e la salva in session_state.
    Ritorna un int (fallback 1200).
    """
    if "viewport_width" not in st.session_state:
        st.session_state.viewport_width = 1200

    components.html(
        """
<script>
  const w = window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth;
  // Streamlit components -> invia valore al parent usando postMessage
  window.parent.postMessage({ type: "STREAMLIT_VIEWPORT_WIDTH", width: w }, "*");
</script>
""",
        height=0,
    )

    # ascolto il messaggio tramite hack CSS/JS? -> Streamlit non espone un listener diretto.
    # Quindi usiamo un approccio più pratico: st.session_state rimane fallback e aggiorna quando rerender.
    return int(st.session_state.get("viewport_width", 1200))


def round_minutes(x) -> int:
    """Arrotondamento classico (13.7 -> 14). Assume x >= 0."""
    try:
        v = float(x)
    except Exception:
        return 0
    if not np.isfinite(v) or v <= 0:
        return 0
    return int(math.floor(v + 0.5))


def fmt_min(x) -> str:
    return str(round_minutes(x))

# ============================================================
# FRIENDS IMAGES (./imgs/<nome>.png) + placeholder
# ============================================================
IMGS_DIR = Path("./imgs")
PLACEHOLDER_IMG = IMGS_DIR / "placeholder.png"

@st.cache_data(show_spinner=False)
def _img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

@st.cache_data(show_spinner=False)
def get_friend_img_b64(name: str) -> str:
    """
    Cerca ./imgs/<nome>.png, altrimenti ./imgs/placeholder.png
    Ritorna base64 (senza prefisso data:)
    """
    p = IMGS_DIR / f"{name}.png"
    if p.exists():
        return _img_to_b64(str(p))

    if PLACEHOLDER_IMG.exists():
        return _img_to_b64(str(PLACEHOLDER_IMG))

    raise FileNotFoundError("Manca anche ./imgs/placeholder.png")

def css_escape_attr(s: str) -> str:
    # Escape minimo per usarlo in selector [aria-label="..."]
    return s.replace("\\", "\\\\").replace('"', '\\"')

# ============================================================
# FRIENDS ADDRESSES (JSON)
# ============================================================
@st.cache_data(show_spinner=False)
def load_friends_addresses(path: str = "friends_address.json") -> dict:
    """
    Atteso formato: {"nome": [lat, lon]}
    Ritorna dict: {"nome": (lon, lat)}  <-- coerente col resto dell'app (x=lon, y=lat)
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    friends = {}
    if not isinstance(raw, dict):
        raise ValueError("friends_address.json deve contenere un oggetto JSON (dict) del tipo {'nome':[lat,lon]}")

    for name, coords in raw.items():
        if not isinstance(name, str) or not name.strip():
            continue
        if (
            not isinstance(coords, (list, tuple))
            or len(coords) != 2
            or coords[0] is None
            or coords[1] is None
        ):
            continue

        lat = float(coords[0])
        lon = float(coords[1])
        friends[name.strip()] = (lon, lat)

    if not friends:
        raise ValueError("friends_address.json non contiene indirizzi validi (atteso {'nome':[lat,lon]}).")

    return friends


# ============================================================
# BARS ADDRESSES (JSON)
# ============================================================
@st.cache_data(show_spinner=False)
def load_bars_addresses(path: str = "bar_address.json") -> dict:
    """
    Atteso formato: {"nome_bar": [lat, lon]}
    Ritorna dict: {"nome_bar": (lon, lat)}  <-- coerente col resto dell'app (x=lon, y=lat)
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    bars = {}
    if not isinstance(raw, dict):
        raise ValueError("bar_address.json deve contenere un oggetto JSON (dict) del tipo {'nome_bar':[lat,lon]}")

    for name, coords in raw.items():
        if not isinstance(name, str) or not name.strip():
            continue
        if (
            not isinstance(coords, (list, tuple))
            or len(coords) != 2
            or coords[0] is None
            or coords[1] is None
        ):
            continue

        lat = float(coords[0])
        lon = float(coords[1])
        bars[name.strip()] = (lon, lat)

    if not bars:
        raise ValueError("bar_address.json non contiene bar validi (atteso {'nome_bar':[lat,lon]}).")

    return bars


# ============================================================
# PAGE CONFIG (UNA SOLA VOLTA, SUBITO)
# ============================================================
st.set_page_config(
    page_title="pariGINI omett",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# COLORS
# ============================================================
LINE_COLORS = {
    "1": "#FFCD00",
    "2": "#003CA6",
    "3": "#7A8B2E",
    "3bis": "#8E9AE6",
    "4": "#7C2E83",
    "5": "#FF7E2E",
    "6": "#6EC4B1",
    "7": "#FA9ABA",
    "7bis": "#6EC4B1",
    "8": "#CEADD2",
    "9": "#B7D84B",
    "10": "#C9910D",
    "11": "#704B1C",
    "12": "#007852",
    "13": "#8E9AE6",
    "14": "#62259D",
}
WALK_COLOR = "#9CA3AF"

# ============================================================
# WHITE BACKGROUND + DARK TEXT (GLOBAL CSS) + FIX SEARCHBOX + FIX PRIMARY BUTTON
# ============================================================
st.markdown(
    """
<style>
/* ===========================
   FORZA TEMA CHIARO (browser-level)
   =========================== */
:root, html {
  color-scheme: light !important;
}

/* Sfondo sempre bianco */
html, body { background: #ffffff !important; }
[data-testid="stAppViewContainer"] { background: #ffffff !important; }
[data-testid="stHeader"] { background: #ffffff !important; }
[data-testid="stSidebar"] { background: #ffffff !important; }
[data-testid="stSidebarContent"] { background: #ffffff !important; }

/* Testi scuri */
body, p, li, label, span, div { color: #111827 !important; }

/* Riduci padding top e rendi pagina più "pulita" */
.block-container { padding-top: 2.2rem; }

/* Titolo più grande */
.pg-title {
  font-size: 3.8rem;
  font-weight: 900;
  line-height: 1.02;
  margin: 0.6rem 0 0.5rem 0;
}

/* Bottoni base */
div.stButton > button {
  border-radius: 12px;
  border: 1px solid rgba(17,24,39,0.18) !important;
  background: #ffffff !important;
  color: #111827 !important;
}
div.stButton > button:hover {
  border-color: rgba(17,24,39,0.35) !important;
}

/* Primary button (Calcola Gini): chiaro con testo scuro, sempre leggibile */
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="baseButton-primary"] {
  background: #e5e7eb !important;
  color: #111827 !important;
  border: 1px solid rgba(17,24,39,0.25) !important;
}
div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="baseButton-primary"]:hover {
  background: #d1d5db !important;
  border-color: rgba(17,24,39,0.35) !important;
}

/* Metric cards più leggibili */
[data-testid="stMetricValue"] { color: #111827 !important; }
[data-testid="stMetricLabel"] { color: rgba(17,24,39,0.75) !important; }

/* ===========================
   SEARCHBOX (dropdown) CHIARO
   =========================== */

/* input del searchbox – tutte le varianti BaseWeb */
div[data-baseweb="input"],
div[data-baseweb="base-input"],
div[data-baseweb="input-container"],
div[data-baseweb="select"] {
  background-color: #f9fafb !important;
  background: #f9fafb !important;
}
div[data-baseweb="input"] *,
div[data-baseweb="base-input"] *,
div[data-baseweb="input-container"] *,
div[data-baseweb="select"] * {
  background-color: transparent !important;
  color: #111827 !important;
}
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input,
div[data-baseweb="select"] input {
  background-color: transparent !important;
  background: transparent !important;
  color: #111827 !important;
  -webkit-text-fill-color: #111827 !important;
}
div[data-baseweb="input"] input::placeholder,
div[data-baseweb="base-input"] input::placeholder,
div[data-baseweb="select"] input::placeholder {
  color: rgba(17,24,39,0.55) !important;
  -webkit-text-fill-color: rgba(17,24,39,0.55) !important;
}

/* bordo dell'input */
div[data-baseweb="input"],
div[data-baseweb="base-input"] {
  border-color: rgba(17,24,39,0.18) !important;
}

/* popover e menu dei suggerimenti */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div {
  background: #f3f4f6 !important;
  background-color: #f3f4f6 !important;
  color: #111827 !important;
  border: 1px solid rgba(17,24,39,0.18) !important;
  box-shadow: 0 10px 24px rgba(17,24,39,0.12) !important;
}

div[data-baseweb="menu"],
div[data-baseweb="menu"] ul {
  background: #f3f4f6 !important;
  background-color: #f3f4f6 !important;
  color: #111827 !important;
}

div[data-baseweb="menu"] * {
  color: #111827 !important;
}

div[data-baseweb="menu"] [role="option"],
div[data-baseweb="menu"] li {
  background: #f3f4f6 !important;
  background-color: #f3f4f6 !important;
}

div[data-baseweb="menu"] [role="option"]:hover,
div[data-baseweb="menu"] [role="option"][aria-selected="true"],
div[data-baseweb="menu"] li:hover {
  background: #e5e7eb !important;
  background-color: #e5e7eb !important;
}

/* Tag / chip selezionato nella searchbox */
div[data-baseweb="tag"],
span[data-baseweb="tag"] {
  background-color: #e5e7eb !important;
  color: #111827 !important;
}

/* Icone SVG dentro la searchbox */
div[data-baseweb="input"] svg,
div[data-baseweb="select"] svg {
  fill: rgba(17,24,39,0.55) !important;
  color: rgba(17,24,39,0.55) !important;
}

/* Decorazioni laterali (sinistra) */
.metro-decor {
  position: fixed;
  left: 10px;
  top: 120px;
  width: 18px;
  z-index: 2;
  opacity: 0.85;
  pointer-events: none;
}
.metro-decor .pill {
  width: 10px;
  margin: 6px auto;
  border-radius: 999px;
  border: 1px solid rgba(17,24,39,0.18);
}
.metro-decor .pill.small { height: 10px; }
.metro-decor .pill.med   { height: 16px; }
.metro-decor .pill.long  { height: 24px; }

/* Nascondi decorazione su schermi piccoli (mobile) */
@media (max-width: 768px) {
  .metro-decor {
    display: none !important;
  }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ============================================================
# DECORAZIONI (SINISTRA, PICCOLE, COLORATE, ALMENO 2 LINEE DIVERSE)
# ============================================================
def render_left_decor():
    # Genera i colori una sola volta al caricamento della pagina
    if "decor_colors" not in st.session_state:
        keys = list(LINE_COLORS.keys())
        chosen = random.sample(keys, k=2)
        if random.random() < 0.45:
            chosen.append(random.choice([k for k in keys if k not in chosen]))

        sizes = ["small", "med", "small", "long", "small", "med"]
        colors = []
        for i in range(len(sizes)):
            if i % 3 == 0:
                colors.append(WALK_COLOR)
            else:
                colors.append(LINE_COLORS[chosen[i % len(chosen)]])
        
        st.session_state.decor_colors = colors
    
    colors = st.session_state.decor_colors
    sizes = ["small", "med", "small", "long", "small", "med"]
    
    pills = "\n".join(
        [f"<div class='pill {sizes[i]}' style='background:{colors[i]}'></div>" for i in range(len(sizes))]
    )
    st.markdown(f"<div class='metro-decor'>{pills}</div>", unsafe_allow_html=True)


render_left_decor()

# ============================================================
# API: Géoplateforme - Autocompletion (IGN)
# ============================================================
COMPLETION_URL = "https://data.geopf.fr/geocodage/completion/"


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def geopf_completion(
    text: str,
    terr: str = "75",
    maximumResponses: int = 8,
    types: str = "StreetAddress,PositionOfInterest",
):
    text = (text or "").strip()
    if not text:
        return []

    r = requests.get(
        COMPLETION_URL,
        params={
            "text": text,
            "terr": terr,
            "type": types,
            "maximumResponses": maximumResponses,
        },
        timeout=10,
        headers={"User-Agent": "streamlit-pariGINI/1.0"},
    )
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "OK":
        return []
    return data.get("results", []) or []


def make_search_fn(map_key: str):
    def _search(searchterm: str):
        if not searchterm or len(searchterm.strip()) < 3:
            st.session_state[map_key] = {}
            return []

        results = geopf_completion(searchterm, terr="75", maximumResponses=8)
        mp = {}
        opts = []
        for r in results:
            ft = r.get("fulltext")
            x = r.get("x")
            y = r.get("y")
            if ft and x is not None and y is not None:
                opts.append(ft)
                mp[ft] = (float(x), float(y))

        st.session_state[map_key] = mp
        return opts

    return _search


def address_autocomplete(label: str, key: str, placeholder: str):
    map_key = f"{key}__map"
    search_fn = make_search_fn(map_key)

    selected = st_searchbox(
        search_fn,
        placeholder=placeholder,
        label=label,
        key=key,
        clear_on_submit=False,
    )

    coords = None
    if selected:
        coords = (st.session_state.get(map_key) or {}).get(selected)

    return selected, coords


# ============================================================
# ROUTE BARS (HTML/CSS)
# ============================================================
def _norm_line_for_color(line):
    if line is None:
        return None
    s = str(line).strip().lower().replace(" ", "")
    if s in {"3b", "3bis"}:
        return "3bis"
    if s in {"7b", "7bis"}:
        return "7bis"
    return str(line).strip()


def compress_edges_to_line_segments(edges):
    segs = []
    cur_line = None
    cur_t = 0.0

    for e in edges:
        line = e.get("line")
        t = float(e.get("time_min", 0.0) or 0.0)

        if cur_line is None:
            cur_line = line
            cur_t = t
            continue

        if line == cur_line:
            cur_t += t
        else:
            segs.append({"kind": "metro", "line": cur_line, "time_min": float(cur_t)})
            cur_line = line
            cur_t = t

    if cur_line is not None:
        segs.append({"kind": "metro", "line": cur_line, "time_min": float(cur_t)})

    return segs


def _get_walk_split(details):
    if not isinstance(details, dict):
        return 0.0, 0.0

    keys_start = [
        "walk_time_start_min",
        "walk_start_time_min",
        "walk_start_min",
        "walk_min_start",
        "walk_time_min_start",
    ]
    keys_end = [
        "walk_time_end_min",
        "walk_end_time_min",
        "walk_end_min",
        "walk_min_end",
        "walk_time_min_end",
    ]

    w_start = None
    w_end = None

    for k in keys_start:
        if k in details and details.get(k) is not None:
            w_start = float(details.get(k) or 0.0)
            break
    for k in keys_end:
        if k in details and details.get(k) is not None:
            w_end = float(details.get(k) or 0.0)
            break

    total_walk = float(details.get("walk_time_min", 0.0) or 0.0)
    edges = details.get("edges", []) or []
    has_metro = len(edges) > 0

    if w_start is not None or w_end is not None:
        return float(w_start or 0.0), float(w_end or 0.0)

    if total_walk <= 0:
        return 0.0, 0.0

    if has_metro:
        return total_walk / 2.0, total_walk / 2.0
    return total_walk, 0.0


def build_segments_for_friend(details):
    mode = details.get("mode", "metro_walk")

    if mode == "walk_only":
        w = float(details.get("walk_time_min", 0.0) or 0.0)
        return [{"kind": "walk", "time_min": w}]

    segs = []
    w_start, w_end = _get_walk_split(details)

    if w_start > 0:
        segs.append({"kind": "walk", "time_min": float(w_start)})

    edges = details.get("edges", []) or []
    segs.extend(compress_edges_to_line_segments(edges))

    if w_end > 0:
        segs.append({"kind": "walk", "time_min": float(w_end)})

    return segs


def render_routes_html(results_df):
    ok_df = results_df[results_df["ok"] == True].copy()
    if ok_df.empty:
        st.info("Nessun percorso disponibile da visualizzare.")
        return

    ok_df = ok_df.sort_values("i")

    max_total = 0.0
    precomp = []
    used_lines = set()

    for _, r in ok_df.iterrows():
        details = r["details"]
        total = float(details.get("total_time_min", r.get("total_time_min", 0.0)) or 0.0)
        max_total = max(max_total, total)

        segs = build_segments_for_friend(details)
        for s in segs:
            if s["kind"] == "metro":
                lk = _norm_line_for_color(s.get("line")) or "?"
                used_lines.add(lk)

        precomp.append((int(r["i"]), total, segs))

    max_total = max(max_total, 1e-9)

    def _legend_pill(label, color):
        return f"""<span class="pill" style="background:{color}"></span><span class="pilltxt">{label}</span>"""

    def _line_sort_key(x):
        try:
            return (0, float(str(x).replace("bis", ".5")))
        except Exception:
            return (1, str(x))

    legend_html = f"""
    <div class="legend">
      <div class="legend-item">{_legend_pill("Camminata", WALK_COLOR)}</div>
      {"".join([f'<div class="legend-item">{_legend_pill("Metro " + str(l), LINE_COLORS.get(l, "#666"))}</div>' for l in sorted(list(used_lines), key=_line_sort_key)])}
    </div>
    """

    rows = []
    for i, total, segs in precomp:
        name = f"Amico {i+1}"
        seg_html = ""

        for s in segs:
            dt = float(s.get("time_min", 0.0) or 0.0)
            if dt <= 0:
                continue

            w_pct = (dt / max_total) * 100.0
            if s["kind"] == "walk":
                color = WALK_COLOR
                title = f"Camminata: {fmt_min(dt)} min"
            else:
                lk = _norm_line_for_color(s.get("line")) or "?"
                color = LINE_COLORS.get(lk, "#666666")
                title = f"Metro {lk}: {fmt_min(dt)} min"

            seg_html += f"""
              <div class="seg" title="{title}" style="width:{w_pct:.4f}%; background:{color};"></div>
            """

        rows.append(
            f"""
            <div class="r">
              <div class="who">{name}</div>
              <div class="bar">{seg_html}</div>
              <div class="tot">{fmt_min(total)} min</div>
            </div>
            """
        )

    iframe_height = int(min(380, 95 + 32 * len(rows)))

    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  :root {{
    --text: rgba(17,24,39,0.92);
    --muted: rgba(17,24,39,0.70);
    --border: rgba(17,24,39,0.18);
  }}

  .wrap {{
    width: 48vw;
    max-width: 680px;
    min-width: 340px;
  }}

  .legend {{
    display:flex;
    flex-wrap: wrap;
    gap: 10px 14px;
    align-items: center;
    margin: 0 0 10px 0;
  }}
  .legend-item {{
    display:flex; gap:8px; align-items:center;
    font: 13px/1.2 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
    color: var(--muted);
  }}
  .pill {{
    width: 12px;
    height: 12px;
    border-radius: 999px;
    border: 1px solid var(--border);
    display:inline-block;
  }}
  .pilltxt {{ white-space: nowrap; }}

  .r {{
    display: grid;
    grid-template-columns: 80px 1fr 74px;
    gap: 10px;
    align-items: center;
    margin: 6px 0;
  }}

  .who {{
    font: 14px/1.2 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
    color: var(--text);
    font-weight: 700;
  }}

  .tot {{
    font: 13px/1.2 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
    color: var(--muted);
    text-align: right;
    white-space: nowrap;
  }}

  .bar {{
    display:flex;
    align-items:center;
    gap: 4px;
    height: 14px;
  }}

  .seg {{
    height: 14px;
    border-radius: 999px;
    min-width: 6px;
    border: 1px solid var(--border);
    box-sizing: border-box;
  }}
</style>
</head>
<body style="margin:0; background:transparent;">
  <div class="wrap">
    {legend_html}
    {''.join(rows)}
  </div>
</body>
</html>
"""
    components.html(html, height=iframe_height, scrolling=False)


# ============================================================
# GINI BAR (components.html)
# ============================================================
def gini_to_color_hex(v):
    v = float(np.clip(v, 0.0, 1.0))
    green = np.array([34, 197, 94])   # #22c55e
    amber = np.array([245, 158, 11])  # #f59e0b
    red = np.array([239, 68, 68])     # #ef4444

    if v <= 0.55:
        t = v / 0.55 if 0.55 else 0
        rgb = green + (amber - green) * t
    else:
        t = (v - 0.55) / (1 - 0.55)
        rgb = amber + (red - amber) * t

    rgb = np.clip(rgb.round().astype(int), 0, 255)
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


def render_gini_bar(gini_value: float):
    st.header("Indice di Gini (Disuguaglianza)")

    if not np.isfinite(gini_value):
        st.warning("Valore Gini non disponibile.")
        return

    v = float(np.clip(gini_value, 0.0, 1.0))
    pct = v * 100.0
    color = gini_to_color_hex(v)

    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  .wrap {{
    max-width: 980px;
    font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
    color: rgba(17,24,39,0.92);
  }}
  .bar {{
    position: relative;
    height: 18px;
    border-radius: 999px;
    background: linear-gradient(90deg, #22c55e 0%, #f59e0b 55%, #ef4444 100%);
    border: 1px solid rgba(17,24,39,0.18);
  }}
  .marker {{
    position:absolute;
    left:{pct:.2f}%;
    top:-20px;
    transform: translateX(-50%);
    font-weight: 900;
    font-size: 16px;
    color: rgba(17,24,39,0.95);
  }}
  .tick {{
    position:absolute;
    left:{pct:.2f}%;
    top:0px;
    transform: translateX(-50%);
    width: 2px;
    height: 18px;
    background: rgba(17,24,39,0.95);
  }}
  .labels {{
    display:flex;
    justify-content:space-between;
    margin-top:6px;
    font-size: 13px;
    color: rgba(17,24,39,0.78);
  }}
  .value {{
    margin-top: 10px;
    font-size: 22px;
    font-weight: 900;
    color: {color};
  }}
</style>
</head>
<body style="margin:0;background:transparent;">
  <div class="wrap">
    <div class="bar">
      <div class="marker">▼</div>
      <div class="tick"></div>
    </div>
    <div class="labels">
      <div>Massima uguaglianza</div>
      <div>Massima disuguaglianza</div>
    </div>
    <div class="value">{v:.4f}</div>
  </div>
</body>
</html>
"""
    components.html(html, height=115, scrolling=False)


# ============================================================
# HEADER
# ============================================================
st.markdown("<div class='pg-title'>pariGINI</div>", unsafe_allow_html=True)
st.markdown(
    """
I tuoi amici ti propongono un bar troppo lontano? Calcola quanto è equa la scelta.  
Misura la disuguaglianza dei tempi di spostamento in metro usando il Gini Index.  
Inserisci da dove partite (minimo 2 persone) e dove andate: il Gini viene calcolato automaticamente.
"""
)

# ============================================================
# LOAD GRAPH (cache)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_graph():
    G_ = build_graph_from_edgelist("./timed_edgelist.geojson")
    idx_ = build_node_index(G_)
    return G_, idx_


with st.spinner("Caricamento rete metro..."):
    try:
        G, node_index = load_graph()
    except Exception as e:
        st.error(f"Errore nel caricamento rete: {e}")
        st.stop()

# ============================================================
# PARAMETRI ROUTING FISSI (non mostrati)
# ============================================================
max_line_changes = 1
change_penalty_min = 2.0


# ============================================================
# INPUT: ORIGINS (AMICI) - SOLO DA JSON + MULTI IMAGE PICKER (st-img-picker)
# ============================================================
from pathlib import Path
from st_img_picker import img_picker  # pip install st-img-picker

try:
    friends_map = load_friends_addresses("friends_address.json")  # {"Nome": (lon, lat)}
except Exception as e:
    st.error(f"Errore nel caricamento di friends_address.json: {e}")
    st.stop()

# stato: selezione amici (multi)
if "selected_friends" not in st.session_state:
    st.session_state.selected_friends = []  # lista di nomi

# --- immagini ---
IMGS_DIR = Path("./imgs")
PLACEHOLDER_IMG = IMGS_DIR / "placeholder.png"

def friend_img_path(name: str) -> str:
    p = IMGS_DIR / f"{name}.png"
    if p.exists():
        return str(p)
    return str(PLACEHOLDER_IMG)

if not PLACEHOLDER_IMG.exists():
    st.error("Manca il file ./imgs/placeholder.png (serve come fallback).")
    st.stop()

st.header("Chi viene oggi di omett?")

names = list(friends_map.keys())
images = [friend_img_path(n) for n in names]
captions = names

# preselezione (indices) coerente con la sessione
selected_set = set(st.session_state.selected_friends)
preselected_idx = [i for i, n in enumerate(names) if n in selected_set]

st.markdown(
    """
<style>
/* Scope SOLO al blocco del picker */
#friends-picker .image-box {
  height: 20rem !important;
  min-width: 10rem !important;   /* lascia uguale se vuoi solo più alto */
}

/* Mantieni l'immagine che riempie il box */
#friends-picker .image {
  height: 100% !important;
}

/* (opzionale) se vuoi anche più larghi i tile */
#friends-picker .item {
  width: 10rem !important;       /* prova 11rem o 12rem */
}
</style>
""",
    unsafe_allow_html=True,
)

# Multi-select: ritorna lista di indici (grazie a return_value="index")
st.markdown("<div id='friends-picker'>", unsafe_allow_html=True)

picked_idx = img_picker(
    label="",
    images=images,
    captions=captions,
    index=preselected_idx,
    return_value="index",
    use_container_width=True,
    allow_multiple=True,
    key="friends_img_picker",
)

st.markdown("</div>", unsafe_allow_html=True)


# Normalizza e aggiorna sessione solo se cambia
picked_idx = picked_idx or []
picked_idx = sorted([int(i) for i in picked_idx if i is not None])
new_selected = [names[i] for i in picked_idx if 0 <= i < len(names)]

if new_selected != st.session_state.selected_friends:
    st.session_state.selected_friends = new_selected

# Mostra selezionati
if st.session_state.selected_friends:
    st.caption("Selezionati: " + ", ".join(st.session_state.selected_friends))
else:
    st.caption("Seleziona almeno 2 persone.")

# I punti di partenza corrispondono ai selezionati
starts = [friends_map[n] for n in st.session_state.selected_friends if n in friends_map]



# ============================================================
# INPUT: CONFRONTA TUTTI I BAR
# ============================================================
st.header("Confronta tutti i bar")

# carica bar
try:
    bars_map = load_bars_addresses("bar_address.json")  # {"Nome bar": (lon, lat)}
except Exception as e:
    st.error(f"Errore nel caricamento di bar_address.json: {e}")
    st.stop()

bar_names = list(bars_map.keys())
targets = [bars_map[n] for n in bar_names]  # lista (lon, lat) nello stesso ordine di bar_names

# ============================================================
# VALIDAZIONE (solo min 2 amici) + UI feedback
# ============================================================
ready = (len(starts) >= 2)

if not ready:
    st.warning("Manca: almeno 2 amici (con indirizzo selezionato).")
else:
    st.success("Tutto ok: puoi confrontare i bar.")


# ============================================================
# ANALYSIS & RESULTS
# ============================================================

# Firma per capire se è cambiato l'input (amici selezionati)
# (se vuoi più robusto includi anche max_line_changes ecc.)
signature = tuple(st.session_state.selected_friends)  # oppure tuple(starts)


prev_sig = st.session_state.get("bars_metrics_signature", None)

if prev_sig != signature:
    st.session_state["bars_metrics_df"] = None
    st.session_state["bars_metrics_signature"] = signature
    st.session_state["picked_bar"] = None

clicked = st.button(
    "CONFRONTA BAR",
    type="primary",
    use_container_width=True,
    disabled=not ready,
)

# Se cliccato, calcola e salva
if clicked:
    st.divider()
    with st.spinner("Calcolo indicatori per tutti i bar..."):
        try:
            metrics_df = accessibility_inequality_to_targets(
                G,
                starts_lonlat=starts,
                targets_lonlat=targets,
                node_index=node_index,
                max_line_changes=max_line_changes,
                change_penalty_min=change_penalty_min,
                max_walk_min_start=15.0,
                max_walk_min_end=15.0,
                max_candidate_stations=25,
                allow_walk_only=True,
                keep_details=False,
                return_per_target_df=False,
            )
        except Exception as e:
            st.error(f"Errore nel calcolo multi-target: {e}")
            st.stop()

    # --- aggancia nomi bar ---
    name_by_id = {i: bar_names[i] for i in range(len(bar_names))}
    metrics_df["bar_name"] = metrics_df["target_id"].map(name_by_id)

        # --- Normalizza Gini (dividi per max) ---
    if "gini_time" in metrics_df.columns:
        try:
             gmax = float(metrics_df["gini_time"].max(skipna=True))
        except Exception:
             gmax = np.nan

        # if np.isfinite(gmax) and gmax > 0:
        #     metrics_df["gini_time_norm"] = (metrics_df["gini_time"] / gmax).clip(0.0, 1.0)
        # else:
        #     # if all values equal or invalid, set normalized to 0.0 for defined values
        #     metrics_df["gini_time_norm"] = metrics_df["gini_time"].apply(lambda v: 0.0 if (v is not None and not (isinstance(v, float) and np.isnan(v))) else np.nan)

        st.session_state["bars_gini_max"] = gmax

    # --- ordine colonne ---
    cols_front = ["bar_name", "gini_time", "mean_time_min", "min_time_min", "max_time_min", "n_ok", "n_total"]
    cols_existing = [c for c in cols_front if c in metrics_df.columns]
    other_cols = [c for c in metrics_df.columns if c not in cols_existing]
    metrics_df = metrics_df[cols_existing + other_cols]

    # --- ordina per equità ---
    if "gini_time" in metrics_df.columns:
        metrics_df = metrics_df.sort_values("gini_time", ascending=True).reset_index(drop=True)

    # salva in sessione
    st.session_state.bars_metrics_df = metrics_df

    # set default picked bar (il migliore)
    if len(metrics_df):
        st.session_state.picked_bar = metrics_df.iloc[0]["bar_name"]

# Se ho risultati salvati, li mostro (anche dopo rerun!)
metrics_df = st.session_state.bars_metrics_df
if metrics_df is not None and len(metrics_df) > 0:
    st.divider()

    # ============================================================
    # OUTPUT VISIVO: confronto
    # ============================================================
    st.subheader("Classifica bar (più equo → meno equo)")

    # --- TOP INFO: mostra subito il bar col Gini minimo ---
    if "gini_time" in metrics_df.columns and metrics_df["gini_time"].notna().any():
        best_row = metrics_df.loc[metrics_df["gini_time"].idxmin()]
        best_name = best_row["bar_name"].title()
        best_gini = float(best_row["gini_time"])

        # Banner grande e visibile (hero)
        st.markdown(
            f"""
            <div style="
                padding: 18px 18px;
                border-radius: 16px;
                border: 1px solid rgba(17,24,39,0.18);
                background: #f3f4f6;
                margin: 10px 0 14px 0;
            ">
            <div style="
                font-size: 26px;
                font-weight: 600;
                line-height: 1.15;
                color: rgba(17,24,39,0.95);
            ">
                Il bar dove vi conviene andare è:
                <span style="
                    font-size: 34px;
                    font-weight: 950;
                    padding: 2px 10px;
                    border-radius: 12px;
                    background: #ffffff;
                    border: 1px solid rgba(17,24,39,0.18);
                    display: inline-block;
                    margin-left: 8px;
                ">
                {best_name}
                </span>
            </div>
            <div style="
                margin-top: 8px;
                font-size: 18px;
                font-weight: 800;
                color: rgba(17,24,39,0.78);
            ">
                Gini: <span style="color: rgba(17,24,39,0.95);">{best_gini:.4f}</span>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="
                padding: 18px 18px;
                border-radius: 16px;
                border: 1px solid rgba(17,24,39,0.18);
                background: #fef3c7;
                margin: 10px 0 14px 0;
                font-size: 22px;
                font-weight: 900;
                color: rgba(17,24,39,0.95);
            ">
            Non è disponibile il Gini per determinare il bar migliore.
            </div>
            """,
            unsafe_allow_html=True,
        )


    # Show only normalized gini in display
    if 'gini_time' in metrics_df.columns:
        metrics_df_display = metrics_df.copy()[['bar_name', 'gini_time', 'mean_time_min']]
        metrics_df_display = metrics_df_display.rename(columns={
            'bar_name': 'Bar',
            'gini_time': 'Gini',
            'mean_time_min': 'Tempo Medio (min)',
        })
    else:
        metrics_df_display = metrics_df.copy()[['bar_name', 'gini_time', 'mean_time_min']]
        metrics_df_display = metrics_df_display.rename(columns={
            'bar_name': 'Bar',
            'gini_time': 'Gini',
            'mean_time_min': 'Tempo Medio (min)',
        })
    metrics_df_display['Bar'] = metrics_df_display['Bar'].str.title()

    st.dataframe(metrics_df_display, use_container_width=True, hide_index=True)

    if "gini_time" in metrics_df.columns:
        st.subheader("Confronto Gini per bar")
        # prefer normalized values for chart when available
        if 'gini_time' in metrics_df.columns:
            chart_df = metrics_df[["bar_name", "gini_time"]].set_index("bar_name")
        st.bar_chart(chart_df, height=320)

    # ============================================================
    # DRILL-DOWN (persistente)
    # ============================================================
    st.subheader("Dettaglio di un bar")

    options = list(metrics_df["bar_name"])
    # scegli indice in base al valore in session_state
    cur = st.session_state.picked_bar
    if cur not in options:
        cur = options[0]
        st.session_state.picked_bar = cur
    cur_index = options.index(cur)

    picked_bar = st.selectbox(
        "Seleziona un bar",
        options=options,
        index=cur_index,
        key="picked_bar_selectbox",  # key diverso da st.session_state.picked_bar
    )

    # aggiorna session_state quando cambia
    if picked_bar != st.session_state.picked_bar:
        st.session_state.picked_bar = picked_bar

    picked_coords = bars_map.get(st.session_state.picked_bar)
    if picked_coords is None:
        st.warning("Coordinate del bar non trovate.")
    else:
        with st.spinner(f"Calcolo dettagli percorso verso: {st.session_state.picked_bar} ..."):
            try:
                results_df, metrics = accessibility_inequality_to_target(
                    G,
                    starts,
                    picked_coords,
                    node_index=node_index,
                    max_line_changes=max_line_changes,
                    change_penalty_min=change_penalty_min,
                    max_walk_min_start=15.0,
                    max_walk_min_end=15.0,
                    max_candidate_stations=25,
                    allow_walk_only=True,
                    keep_details=True,
                )
            except Exception as e:
                st.error(f"Errore nel calcolo dettagli: {e}")
                st.stop()

        st.caption(f"Destinazione: **{st.session_state.picked_bar}**")

        # normalize single-target gini using previously computed global min/max
        gini_value = metrics.get("gini_time", None)

        render_gini_bar(gini_value)

        render_routes_html(results_df)

        st.subheader("Statistiche tempi di percorrenza")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Minimo", f"{fmt_min(metrics.get('min_time_min', np.nan))} min")
        with c2:
            st.metric("Medio", f"{fmt_min(metrics.get('mean_time_min', np.nan))} min")
        with c3:
            st.metric("Massimo", f"{fmt_min(metrics.get('max_time_min', np.nan))} min")

        with st.expander("Esporta risultati", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                csv_all = metrics_df.to_csv(index=False)
                st.download_button(
                    label="Scarica CSV (confronto bar)",
                    data=csv_all,
                    file_name="bar_comparison_metrics.csv",
                    mime="text/csv",
                )

            with col2:
                summary = {
                    "picked_bar": st.session_state.picked_bar,
                    "picked_bar_lon": float(picked_coords[0]),
                    "picked_bar_lat": float(picked_coords[1]),
                    "picked_bar_metrics": {
                        k: (float(v) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else v)
                        for k, v in metrics.items()
                    },
                    "routing": {
                        "max_line_changes": max_line_changes,
                        "change_penalty_min": change_penalty_min,
                        "max_walk_min_start": 15.0,
                        "max_walk_min_end": 15.0,
                        "max_candidate_stations": 25,
                        "allow_walk_only": True,
                    },
                }
                json_data = json.dumps(summary, indent=2)
                st.download_button(
                    label="Scarica JSON (dettaglio bar selezionato)",
                    data=json_data,
                    file_name="picked_bar_detail.json",
                    mime="application/json",
                )



# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(
    """
---
**pariGINI**

Dati: RATP Metro Network
Autocomplete: Géoplateforme (IGN) - completion

Francesco Farina e Francesco Paolo Savatteri. Per omett e per tutt3
"""
)