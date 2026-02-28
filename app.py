import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from groq import Groq
import json
import re
import numpy as np
from scipy import stats as scipy_stats
import os
import glob
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataWhisper — AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("API Key not found! Please check your .env file.")
    st.stop()

# Safe check for Kaggle API
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()
    KAGGLE_AVAILABLE = True
except Exception as e:
    KAGGLE_AVAILABLE = False
    print(f"Kaggle Error: {e}")

# Safe check for PDF Generation
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    print("FPDF not installed. PDF export disabled.")


# ─────────────────────────────────────────────────────────────
# SESSION STATE & NAVIGATION LOGIC
# ─────────────────────────────────────────────────────────────
for key, default in {
    "view": "home",
    "theme": "light",
    "messages": [], "df": None, "df_name": None,
    "insights": [], "selected_insight": None,
    "dataset_summary": None, "llm_history": [],
    "insight_messages": [], "last_insight_idx": None,
    "kaggle_results": None,
    "pdf_bytes": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def set_view_insight(idx):
    if st.session_state.last_insight_idx != idx:
        st.session_state.insight_messages = []
        st.session_state.last_insight_idx = idx
    st.session_state.selected_insight = idx
    st.session_state.view = "insight"

def set_view_dashboard():
    st.session_state.selected_insight = None
    st.session_state.view = "dashboard"

def go_home():
    st.session_state.view = "home"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"


# ─────────────────────────────────────────────────────────────
# DYNAMIC THEME VARIABLES 
# ─────────────────────────────────────────────────────────────
is_dark = st.session_state.theme == "dark"

bg_color      = "#0e1117"  if is_dark else "#f4f6f9"
text_color    = "#e8eeff"  if is_dark else "#1e293b"   
subtext_color = "#a8b8d8"  if is_dark else "#475569"   
sidebar_bg    = "#111827"  if is_dark else "#ffffff"
card_bg       = "#1a2035"  if is_dark else "#ffffff"
border_color  = "#2e3f60"  if is_dark else "#e2e8f0"
button_bg     = "#2a3f6f"  if is_dark else "#ffffff"   
button_text   = "#e8eeff"  if is_dark else "#1e293b"   
button_hover  = "#3a5290"  if is_dark else "#f8fafc"   
heading_color = "#ffffff"  if is_dark else "#0f172a"
input_bg      = "#0e1117"  if is_dark else "#ffffff" # Deep dark for inputs
shadow_css    = "rgba(0,0,0,0.45)" if is_dark else "rgba(0,0,0,0.06)"
plotly_template = "plotly_dark" if is_dark else "plotly_white"


# ─────────────────────────────────────────────────────────────
# THEME CSS & ANIMATIONS
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

@keyframes fadeIn {{
    0% {{ opacity: 0; transform: translateY(15px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes slideDown {{
    0% {{ opacity: 0; transform: translateY(-20px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes float {{
    0% {{ transform: translateY(0px); }}
    50% {{ transform: translateY(-8px); }}
    100% {{ transform: translateY(0px); }}
}}
@keyframes gradientShift {{
    0% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}

/* ── App Background Forces ── */
html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    font-family: 'Inter', sans-serif;
    background-color: {bg_color} !important;
    color: {text_color} !important;
    transition: background-color 0.4s ease, color 0.4s ease;
}}

/* ── Broad text override: kills Streamlit's native grey in dark mode ── */
p, span, li, div, td, th, small, strong, em,
.stMarkdown p, .stMarkdown li, .stMarkdown span,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {{
    color: {text_color} !important;
}}
[data-testid="stCaptionContainer"] p {{ color: {subtext_color} !important; }}

[data-testid="stSidebar"] {{
    background: {sidebar_bg} !important;
    border-right: 1px solid {border_color};
    transition: background 0.4s ease;
}}
[data-testid="stSidebar"] * {{ color: {text_color} !important; }}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{ color: {heading_color} !important; }}

/* ── Buttons (Insight cases & general buttons) ── */
div[data-testid="stButton"] > button {{
    background-color: {button_bg} !important;
    border: 1px solid {border_color} !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    box-shadow: 0 2px 4px {shadow_css};
}}
div[data-testid="stButton"] > button p,
div[data-testid="stButton"] > button span,
div[data-testid="stButton"] > button div {{
    color: {button_text} !important;
}}
div[data-testid="stButton"] > button:hover {{
    border-color: #4f6ef7 !important;
    background-color: {button_hover} !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 15px rgba(79, 110, 247, 0.25);
}}
div[data-testid="stButton"] > button:hover p,
div[data-testid="stButton"] > button:hover span,
div[data-testid="stButton"] > button:hover div {{
    color: #ffffff !important;
}}

.main .block-container {{
    padding: 1rem 2rem 2rem;
    max-width: 100%;
    animation: fadeIn 0.6s ease-out forwards;
}}

h1 {{ font-weight: 700 !important; letter-spacing: -0.5px; color: {heading_color} !important; }}
h2, h3 {{ font-weight: 600 !important; color: {heading_color} !important; }}

/* ── Animated Logo & Title ── */
.title-container {{ text-align: center; margin-bottom: 10px; }}
.animated-logo {{
    display: inline-block; font-size: 3.5rem;
    animation: float 3s ease-in-out infinite;
    margin-right: 15px; vertical-align: middle;
}}
.animated-title {{
    display: inline-block; font-size: 4rem !important; font-weight: 800 !important;
    background: linear-gradient(270deg, #4f6ef7, #00c4b4, #9b59b6, #4f6ef7);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientShift 6s ease infinite;
    margin-bottom: 0; vertical-align: middle;
    letter-spacing: -1.5px !important;
}}

/* ── Metric Cards ── */
[data-testid="stMetric"] {{
    background: {card_bg} !important;
    border-radius: 10px;
    padding: 16px !important;
    border: 1px solid {border_color} !important;
    box-shadow: 0 4px 8px {shadow_css};
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}
[data-testid="stMetric"]:hover {{
    transform: translateY(-4px);
    box-shadow: 0 8px 20px {shadow_css};
}}
[data-testid="stMetricValue"] {{ font-weight: 800 !important; color: {heading_color} !important; }}
[data-testid="stMetricLabel"] {{ font-size: 13px !important; color: {subtext_color} !important; font-weight: 500; }}

/* ── Hero Section ── */
.hero-subtitle-container {{
    display: flex;
    justify-content: center;
    width: 100%;
    margin-bottom: 30px;
}}
.hero-subtitle {{
    font-size: 1.2rem; color: {subtext_color} !important;
    text-align: center !important; max-width: 700px;
    line-height: 1.6;
}}

/* ── Analysis Panel ── */
.analysis-panel {{
    background: {card_bg} !important;
    border-left: 5px solid #4f6ef7;
    padding: 25px;
    border-radius: 0 12px 12px 0;
    height: 100%;
    box-shadow: 0 4px 10px {shadow_css};
    transition: background 0.4s ease;
}}

/* ── Top Navigation Bar ── */
.top-nav-container {{
    display: flex; align-items: center; justify-content: space-between;
    background: {card_bg};
    padding: 12px 25px;
    border-radius: 12px;
    border: 1px solid {border_color};
    box-shadow: 0 4px 10px {shadow_css};
    margin-bottom: 30px;
    animation: slideDown 0.5s ease-out forwards;
    transition: all 0.4s ease;
}}

/* ── STREAMLIT NATIVE UI AGGRESSIVE FIXES FOR DARK MODE ── */

/* File Uploader Dropzone FIX */
[data-testid="stFileUploadDropzone"] {{
    background-color: {input_bg} !important;
    border: 2px dashed {border_color} !important;
}}
[data-testid="stFileUploadDropzone"] div,
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] small {{
    color: {text_color} !important;
}}

/* Chat Input FIX */
[data-testid="stChatInput"] {{
    background-color: {input_bg} !important;
    border: 1px solid {border_color} !important;
}}
[data-testid="stChatInput"] * {{
    color: {text_color} !important;
}}

/* Text Inputs / Text Areas FIX */
.stTextInput input, .stTextArea textarea,
[data-baseweb="input"], [data-baseweb="input"] input,
[data-testid="stChatInputTextArea"] {{
    background-color: {input_bg} !important;
    color: {text_color} !important;
    border-color: {border_color} !important;
    -webkit-text-fill-color: {text_color} !important;
}}

/* Tabs FIX */
[data-testid="stTabs"] button[role="tab"] {{ background-color: transparent !important; }}
[data-testid="stTabs"] button[role="tab"] p {{ color: {subtext_color} !important; font-size: 1.05rem; }}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {{ color: {heading_color} !important; font-weight: 700; }}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{ border-bottom-color: #4f6ef7 !important; }}

/* Sidebar collapse button (<<) FIX */
[data-testid="collapsedControl"] {{
    background-color: {card_bg} !important;
    border: 1px solid {border_color} !important;
    border-radius: 8px;
    box-shadow: 0 2px 4px {shadow_css};
    transition: all 0.3s ease;
    z-index: 100;
}}
[data-testid="collapsedControl"] svg {{
    fill: {text_color} !important;
    color: {text_color} !important;
}}

::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-thumb {{ background: {border_color}; border-radius: 3px; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TOP NAVIGATION BAR
# ─────────────────────────────────────────────────────────────
st.markdown("<div class='top-nav-container'>", unsafe_allow_html=True)
nav_col1, nav_col2, nav_col3 = st.columns([1, 8, 1])
with nav_col1:
    st.button("🏠 Home", on_click=go_home, use_container_width=True)
with nav_col2:
    st.markdown(
        f"<div style='text-align:center;'><h3 style='margin:0; font-size:1.4rem; color:{heading_color};'>⚡ DataWhisper</h3></div>",
        unsafe_allow_html=True
    )
with nav_col3:
    theme_icon = "🌙 Dark" if st.session_state.theme == "light" else "☀️ Light"
    st.button(theme_icon, on_click=toggle_theme, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# COLOR PALETTE & CHART THEME
# ─────────────────────────────────────────────────────────────
PALETTE = ["#4f6ef7","#00c4b4","#f5a623","#e74c3c","#9b59b6",
           "#1abc9c","#e67e22","#3498db","#e91e63","#00bcd4"]

CHART_THEME = dict(
    font=dict(family="Inter, sans-serif", size=12, color=text_color),
    margin=dict(l=10, r=10, t=40, b=10),
    colorway=PALETTE,
    legend=dict(borderwidth=1, font=dict(size=11), bordercolor=border_color),
    autosize=True,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)"
)

PTYPE_ICONS = {
    "distribution":"〜", "correlation":"⟷", "trend":"↗",
    "comparison":"⇄",   "outlier":"◉",      "composition":"⬡", "ranking":"▤"
}


# ─────────────────────────────────────────────────────────────
# DATASET PROFILER
# ─────────────────────────────────────────────────────────────
def profile_dataset(df: pd.DataFrame) -> dict:
    numeric_cols  = df.select_dtypes(include=np.number).columns.tolist()
    category_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

    for col in list(category_cols):
        try:
            sample = df[col].dropna().head(50)
            parsed = pd.to_datetime(sample, format='mixed', errors="coerce")
            if parsed.notna().mean() > 0.7:
                datetime_cols.append(col); category_cols.remove(col)
        except Exception:
            pass

    col_stats = {}
    for col in df.columns:
        s = {"name": col, "dtype": str(df[col].dtype),
             "null_count": int(df[col].isnull().sum()),
             "null_pct": round(df[col].isnull().mean()*100, 1),
             "unique": int(df[col].nunique())}
        if col in numeric_cols:
            s.update({
                "mean":   round(float(df[col].mean()),   4),
                "median": round(float(df[col].median()), 4),
                "std":    round(float(df[col].std()),    4),
                "min":    round(float(df[col].min()),    4),
                "max":    round(float(df[col].max()),    4),
                "skew":   round(float(df[col].skew()),   3),
                "q25":    round(float(df[col].quantile(0.25)), 4),
                "q75":    round(float(df[col].quantile(0.75)), 4),
            })
        elif col in category_cols:
            vc = df[col].value_counts()
            s["top_values"] = vc.head(5).to_dict()
            s["top_value"]  = vc.index[0] if len(vc) else None
        col_stats[col] = s

    corr_matrix = None
    strong_corrs = []
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().round(3).to_dict()
        for c1 in numeric_cols:
            for c2 in numeric_cols:
                if c1 >= c2: continue
                val = corr_matrix[c1].get(c2, 0)
                if abs(val) >= 0.4:
                    strong_corrs.append({"col1":c1,"col2":c2,"r":val})
        strong_corrs.sort(key=lambda x: abs(x["r"]), reverse=True)

    return {
        "shape": df.shape,
        "numeric_cols": numeric_cols,
        "category_cols": category_cols,
        "datetime_cols": datetime_cols,
        "col_stats": col_stats,
        "corr_matrix": corr_matrix,
        "strong_corrs": strong_corrs[:10],
        "memory_mb": round(df.memory_usage(deep=True).sum()/1e6, 2),
        "duplicate_rows": int(df.duplicated().sum()),
    }


# ─────────────────────────────────────────────────────────────
# AI STREAMING GENERATOR FUNCTION
# ─────────────────────────────────────────────────────────────
def get_groq_stream(prompt_messages, api_key):
    client = Groq(api_key=api_key)
    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=prompt_messages,
        temperature=0.4,
        max_tokens=600,
        stream=True
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


# ─────────────────────────────────────────────────────────────
# AI: DESCRIBE DATASET
# ─────────────────────────────────────────────────────────────
def describe_dataset(df: pd.DataFrame, filename: str, api_key: str) -> str:
    try:
        client = Groq(api_key=api_key)
    except Exception:
        return ""

    profile = profile_dataset(df)
    prompt = f"""You are a senior data analyst. A user uploaded "{filename}".
Write 4-6 sentences describing this dataset: what it seems to be about, its structure,
data quality, and 1-2 notable initial observations with specific numbers.

Facts:
- {df.shape[0]:,} rows × {df.shape[1]} columns
- Numeric ({len(profile['numeric_cols'])}): {profile['numeric_cols']}
- Categorical ({len(profile['category_cols'])}): {profile['category_cols']}
- Missing: {sum(s['null_count'] for s in profile['col_stats'].values())} values
- Duplicates: {profile['duplicate_rows']}

End with: "I'll now identify the most interesting patterns to visualize."
Be specific. Cite numbers. Do NOT be generic."""

    try:
        r = client.chat.completions.create(
            messages=[{"role":"system","content":"Expert data analyst. Concise and specific."},
                      {"role":"user","content":prompt}],
            model="llama-3.3-70b-versatile", temperature=0.3, max_tokens=450)
        return r.choices[0].message.content
    except Exception:
        return f"Dataset **{filename}** loaded: {df.shape[0]:,} rows × {df.shape[1]} columns. I'll now identify the most interesting patterns to visualize."


# ─────────────────────────────────────────────────────────────
# AI: GENERATE INSIGHTS
# ─────────────────────────────────────────────────────────────
def generate_insights(df: pd.DataFrame, api_key: str) -> list:
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Groq init failed: {e}"); return []

    profile = profile_dataset(df)
    prompt = f"""You are a senior Data Scientist. Identify exactly 5 genuinely interesting phenomena to visualize.

DATASET: {df.shape[0]} rows × {df.shape[1]} columns
ALL COLUMNS: {list(df.columns)}
NUMERIC: {profile['numeric_cols']}
CATEGORICAL: {profile['category_cols']}

NUMERIC STATS:
{json.dumps({c:{k:v for k,v in profile['col_stats'][c].items() if k in ['mean','std','min','max','skew','q25','q75','null_pct']} for c in profile['numeric_cols']}, indent=2)}

Return ONLY a JSON array of exactly 5 objects (no markdown):
{{
  "title": "Specific insightful title",
  "description": "2-3 sentences: what the pattern is, why it matters",
  "phenomenon_type": "distribution"|"correlation"|"trend"|"comparison"|"outlier"|"composition"|"ranking",
  "chart_type": "scatter"|"bar"|"line"|"histogram"|"box"|"violin"|"heatmap"|"area"|"bubble"|"pie"|"scatter_3d",
  "x_col": "exact column name or null",
  "y_col": "exact column name or null",
  "z_col": "exact column name or null",
  "color_col": "exact column name or null",
  "size_col": "exact column name or null",
  "aggregation": "none"|"mean"|"sum"|"count"|"median",
  "sort_by_value": true|false,
  "top_n": null|integer
}}
Column names MUST exactly match. Return pure JSON only."""

    try:
        resp = client.chat.completions.create(
            messages=[{"role":"system","content":"Precise data analyst. Return valid JSON arrays only."},
                      {"role":"user","content":prompt}],
            model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=2500)
        text = resp.choices[0].message.content
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if m: return json.loads(m.group(0))
    except Exception as e:
        st.error(f"Insight error: {e}")
    return []


# ─────────────────────────────────────────────────────────────
# CHART BUILDER (NOW WITH FORCED COLORS)
# ─────────────────────────────────────────────────────────────
def build_chart(ins: dict, df: pd.DataFrame) -> go.Figure | None:
    ct  = ins.get("chart_type","bar")
    xc  = ins.get("x_col")
    yc  = ins.get("y_col")
    zc  = ins.get("z_col")
    cc  = ins.get("color_col")
    sc  = ins.get("size_col")
    agg = ins.get("aggregation","none")
    sv  = ins.get("sort_by_value", False)
    tn  = ins.get("top_n")

    valid = set(df.columns)
    xc = xc if xc in valid else None
    yc = yc if yc in valid else None
    zc = zc if zc in valid else None
    cc = cc if cc in valid else None
    sc = sc if sc in valid else None

    if ct in ["histogram", "box", "violin"]:
        agg = "none"
        if ct == "histogram" and xc is None and yc is not None:
            xc, yc = yc, None

    if ct == "pie" and xc is None and yc is not None:
        xc, yc = yc, None

    pf = df.copy()

    if ct in ["line", "area", "pie"] and agg == "none" and xc:
        if yc and pf[xc].duplicated().any() and pd.api.types.is_numeric_dtype(pf.get(yc, pd.Series())):
            agg = "mean" if ct != "pie" else "sum"
    elif ct == "pie" and not yc and agg == "none":
        agg = "count"

    if agg != "none" and xc:
        if agg == "count":
            grp = [xc]+([cc] if cc else [])
            pf  = pf.groupby(grp, observed=True).size().reset_index(name="count"); yc="count"
        elif yc and yc in pf.columns:
            fn  = {"mean":"mean","sum":"sum","median":"median"}.get(agg,"mean")
            grp = [xc]+([cc] if cc else [])
            pf  = pf.groupby(grp, observed=True)[yc].agg(fn).reset_index()

    if sv and yc and yc in pf.columns:
        pf = pf.sort_values(yc, ascending=False)
    if tn and xc and xc in pf.columns:
        top = (pf.groupby(xc, observed=True)[yc].sum().nlargest(tn).index
               if (yc and yc in pf.columns) else pf[xc].value_counts().nlargest(tn).index)
        pf = pf[pf[xc].isin(top)]

    if ct in ["line", "area"] and xc:
        pf = pf.sort_values(xc)
        
    # --- AUTO-COLORIZATION ---
    if ct in ["bar", "box", "violin"] and not cc and xc in pf.columns:
        if pf[xc].nunique() <= 20:
            cc = xc

    hov    = [c for c in pf.columns.tolist()[:8] if c not in [xc, yc, zc]]
    title  = ins.get("title","")
    fig    = None
    profile = profile_dataset(df)

    try:
        if not xc and not yc:
            return None

        # Add color_discrete_sequence=PALETTE to ensure charts are always colorful
        if ct == "scatter":
            if not xc or not yc: return None
            needs_tl = (xc and yc
                        and pd.api.types.is_numeric_dtype(pf.get(xc, pd.Series()))
                        and pd.api.types.is_numeric_dtype(pf.get(yc, pd.Series()))
                        and not cc)
            fig = px.scatter(pf, x=xc, y=yc, color=cc, size=sc, hover_data=hov,
                             opacity=0.75, title=title, color_discrete_sequence=PALETTE)
            if needs_tl:
                try:
                    mask = pf[[xc, yc]].notna().all(axis=1)
                    sl, ic, r, p, _ = scipy_stats.linregress(pf[mask][xc], pf[mask][yc])
                    fig.add_trace(go.Scatter(
                        x=pf[mask][xc], y=sl*pf[mask][xc]+ic,
                        mode='lines', name='Trendline',
                        line=dict(color='#f5a623', width=3), hoverinfo='skip'
                    ))
                    fig.add_annotation(
                        text=f"R² = {r**2:.3f}  |  p = {p:.3g}",
                        xref="paper", yref="paper", x=0.02, y=0.97,
                        showarrow=False, font=dict(size=11, color="#f5a623"),
                        bgcolor=card_bg, bordercolor=border_color, borderwidth=1, borderpad=6)
                except Exception:
                    pass

        elif ct == "scatter_3d":
            if not xc or not yc or not zc: return None
            fig = px.scatter_3d(pf, x=xc, y=yc, z=zc, color=cc, size=sc, hover_data=hov,
                                opacity=0.75, title=title, color_discrete_sequence=PALETTE)

        elif ct == "bubble":
            if not xc or not yc: return None
            fig = px.scatter(pf, x=xc, y=yc, color=cc, size=sc, hover_data=hov,
                             opacity=0.72, title=title, color_discrete_sequence=PALETTE)

        elif ct == "bar":
            fig = px.bar(pf, x=xc, y=yc, color=cc,
                         barmode="group" if cc else "relative",
                         text_auto=".3s", title=title, color_discrete_sequence=PALETTE)
            fig.update_traces(textposition="outside", textfont_size=10, marker_line_width=0)

        elif ct == "line":
            fig = px.line(pf, x=xc, y=yc, color=cc, markers=True, title=title, color_discrete_sequence=PALETTE)
            fig.update_traces(line_width=2.5)

        elif ct == "area":
            fig = px.area(pf, x=xc, y=yc, color=cc, title=title, color_discrete_sequence=PALETTE)

        elif ct == "pie":
            if not xc: return None
            fig = px.pie(pf, names=xc, values=yc, title=title, hole=0.3, color_discrete_sequence=PALETTE)
            fig.update_traces(textposition='inside', textinfo='percent+label')

        elif ct == "histogram":
            if not xc: return None
            fig = px.histogram(pf, x=xc, color=cc,
                               nbins=min(50, max(10, int(pf.shape[0]**0.5))),
                               opacity=0.82, barmode="overlay", title=title, color_discrete_sequence=PALETTE)

        elif ct == "box":
            fig = px.box(pf, x=xc, y=yc, color=cc, points="outliers", notched=True, title=title, color_discrete_sequence=PALETTE)

        elif ct == "violin":
            fig = px.violin(pf, x=xc, y=yc, color=cc, box=True, points="outliers", title=title, color_discrete_sequence=PALETTE)

        elif ct == "heatmap":
            num_cols = profile["numeric_cols"]
            if xc in num_cols and yc in num_cols:
                cols_h = [c for c in num_cols if df[c].nunique()>1][:16]
                corr   = df[cols_h].corr().round(2)
                fig    = go.Figure(go.Heatmap(
                    z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                    colorscale=[[0,"#e74c3c"],[0.5,card_bg],[1,"#4f6ef7"]], zmid=0,
                    text=corr.values.round(2), texttemplate="%{text}",
                    textfont=dict(size=10, color=text_color),
                    hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z}<extra></extra>",
                    colorbar=dict(title="r", thickness=12)))
                fig.update_layout(title=title)

        if fig:
            fig.update_layout(template=plotly_template, **CHART_THEME)
            grid_color = "rgba(150,150,150,0.1)" if is_dark else "rgba(0,0,0,0.05)"
            fig.update_xaxes(automargin=True, autorange=True, gridcolor=grid_color)
            fig.update_yaxes(automargin=True, autorange=True, gridcolor=grid_color)

    except Exception as e:
        st.warning(f"Note: This specific chart variation could not be drawn properly ({e})")
        return None

    return fig


# ─────────────────────────────────────────────────────────────
# OVERVIEW CHARTS (auto, no LLM)
# ─────────────────────────────────────────────────────────────
def render_overview_charts(df: pd.DataFrame, profile: dict):
    num_cols = profile["numeric_cols"]
    cat_cols = profile["category_cols"]

    if len(num_cols) >= 3:
        cols_h = [c for c in num_cols if df[c].nunique()>1][:14]
        corr   = df[cols_h].corr().round(2)
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale=[[0,"#e74c3c"],[0.5,card_bg],[1,"#4f6ef7"]], zmid=0,
            text=corr.values.round(2), texttemplate="%{text}", 
            textfont=dict(size=11, color=text_color),
            hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z}<extra></extra>",
            colorbar=dict(title="r", thickness=10, len=0.9)))
        fig.update_layout(template=plotly_template, **CHART_THEME, title="Correlation Matrix — Numeric Features", height=400)
        fig.update_xaxes(automargin=True); fig.update_yaxes(automargin=True)
        st.plotly_chart(fig, use_container_width=True)

    num_show = num_cols[:6]
    if num_show:
        n = len(num_show); ncols = min(3,n); nrows = (n+ncols-1)//ncols
        fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=num_show,
                            vertical_spacing=0.14, horizontal_spacing=0.08)
        for i, col in enumerate(num_show):
            r, c = divmod(i, ncols)
            data = df[col].dropna()
            fig.add_trace(go.Histogram(x=data, name=col,
                                        marker_color=PALETTE[i%len(PALETTE)],
                                        opacity=0.82, nbinsx=30, showlegend=False),
                          row=r+1, col=c+1)
            fig.add_vline(x=data.mean(), line_dash="dash", line_color="#f5a623",
                          line_width=1.2, row=r+1, col=c+1)  # type: ignore
        fig.update_layout(template=plotly_template, **CHART_THEME, title="Distributions of Numeric Features",
                          height=max(300, 260*nrows))
        fig.update_annotations(font=dict(size=11, color=text_color))
        st.plotly_chart(fig, use_container_width=True)

    cat_show = cat_cols[:4]
    if cat_show:
        ncols = min(2, len(cat_show)); nrows = (len(cat_show)+ncols-1)//ncols
        fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=cat_show,
                            vertical_spacing=0.18, horizontal_spacing=0.1)
        for i, col in enumerate(cat_show):
            r, c = divmod(i, ncols)
            vc = df[col].value_counts().head(12)
            fig.add_trace(go.Bar(x=vc.values[::-1], y=vc.index.astype(str).tolist()[::-1],
                                  orientation="h", marker_color=PALETTE[i%len(PALETTE)],
                                  showlegend=False, name=col), row=r+1, col=c+1)
        fig.update_layout(template=plotly_template, **CHART_THEME, title="Categorical Columns — Value Counts",
                          height=max(300, 260*nrows))
        fig.update_annotations(font=dict(size=11, color=text_color))
        st.plotly_chart(fig, use_container_width=True)

    null_s = df.isnull().sum()
    null_s = null_s[null_s>0].sort_values(ascending=False)
    if not null_s.empty:
        fig = go.Figure(go.Bar(
            x=null_s.index.tolist(), y=null_s.values, marker_color="#e74c3c",
            text=[f"{v/len(df)*100:.1f}%" for v in null_s.values], textposition="outside"))
        fig.update_layout(template=plotly_template, **CHART_THEME, title="Missing Values by Column",
                          yaxis_title="Missing Count", height=320)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PDF REPORT GENERATOR
# ─────────────────────────────────────────────────────────────
def create_pdf_report(df_name, profile, summary_text, insights, df):
    if not FPDF_AVAILABLE:
        return None

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    PRIMARY_BLUE = (79, 110, 247)
    DARK_SLATE = (26, 43, 76)
    TEXT_BLACK = (50, 50, 50)
    BG_LIGHT = (240, 244, 250)

    pdf.set_font("Arial", 'B', 22)
    pdf.set_text_color(*PRIMARY_BLUE)
    pdf.cell(0, 15, txt="DataWhisper Analysis Report", ln=True, align='C')
    pdf.ln(5)

    pdf.set_fill_color(*BG_LIGHT)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(*DARK_SLATE)
    pdf.cell(0, 10, txt=f" Dataset: {df_name}", ln=True, fill=True)
    pdf.set_font("Arial", '', 10)
    pdf.set_text_color(*TEXT_BLACK)
    pdf.cell(0, 10, txt=f" Rows: {profile['shape'][0]:,} | Columns: {profile['shape'][1]}", ln=True, fill=True)
    pdf.ln(8)

    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(*PRIMARY_BLUE)
    pdf.cell(0, 10, txt="Executive Summary", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.set_text_color(*TEXT_BLACK)
    clean_summary = summary_text.encode('latin-1', 'ignore').decode('latin-1')
    clean_summary = clean_summary.replace('**', '')
    pdf.multi_cell(0, 6, txt=clean_summary)
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(*PRIMARY_BLUE)
    pdf.cell(0, 10, txt="Key Insights & Phenomenons", ln=True)
    pdf.set_text_color(*TEXT_BLACK)
    pdf.ln(5)

    for i, ins in enumerate(insights):
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(*DARK_SLATE)
        clean_title = ins.get('title', '').encode('latin-1', 'ignore').decode('latin-1')
        pdf.cell(0, 10, txt=f"{i+1}. {clean_title}", ln=True)

        pdf.set_font("Arial", '', 10)
        pdf.set_text_color(*TEXT_BLACK)
        clean_desc = ins.get('description', '').encode('latin-1', 'ignore').decode('latin-1')
        clean_desc = clean_desc.replace('**', '')
        pdf.multi_cell(0, 6, txt=clean_desc)

        fig = build_chart(ins, df)
        if fig:
            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="#f0f4fa",
                plot_bgcolor="#f0f4fa",
                font=dict(color="#1a2b4c"),
                colorway=PALETTE, # Forces the graphs in PDF to use your custom colors!
                margin=dict(l=20, r=20, t=40, b=20)
            )
            fig.update_xaxes(gridcolor="rgba(26, 43, 76, 0.1)", title_font=dict(color="#1a2b4c"), tickfont=dict(color="#1a2b4c"))
            fig.update_yaxes(gridcolor="rgba(26, 43, 76, 0.1)", title_font=dict(color="#1a2b4c"), tickfont=dict(color="#1a2b4c"))
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    fig.write_image(tmpfile.name, format="png", engine="kaleido", width=700, height=400)
                    pdf.image(tmpfile.name, w=170)
                    os.remove(tmpfile.name)
            except ValueError:
                pdf.set_font("Arial", 'I', 9)
                pdf.set_text_color(200, 0, 0)
                pdf.cell(0, 7, txt=f"[ Chart hidden. Install 'kaleido' via pip to embed images ]", ln=True)
                pdf.set_text_color(*TEXT_BLACK)
            except Exception as e:
                pdf.set_font("Arial", 'I', 9)
                pdf.cell(0, 7, txt=f"[ Interactive {ins.get('chart_type')} chart available in the dashboard ]", ln=True)
        pdf.ln(10)

    pdf_out = pdf.output(dest='S')
    return pdf_out.encode('latin-1') if isinstance(pdf_out, str) else bytes(pdf_out)


# ─────────────────────────────────────────────────────────────
# FILE PROCESSING HELPER
# ─────────────────────────────────────────────────────────────
def handle_file_upload(uploaded_file):
    if uploaded_file:
        fname = uploaded_file.name
        if st.session_state.df_name != fname:
            try:
                sep = "\t" if fname.endswith(".tsv") else ","
                st.session_state.df = pd.read_csv(uploaded_file, sep=sep)
                st.session_state.df_name = fname
                st.session_state.insights = []
                st.session_state.selected_insight = None
                st.session_state.llm_history = []
                st.session_state.messages = []
                st.session_state.insight_messages = []
                st.session_state.pdf_bytes = None
                st.session_state.view = "dashboard"
            except Exception as e:
                st.error(f"Error: {e}")


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ DataWhisper")
    st.caption("AI-powered data analysis")
    st.divider()

    st.markdown("### 📂 Active Dataset")
    sidebar_file = st.file_uploader("Upload new dataset", type=["csv","tsv"], key="sb_upload")
    if sidebar_file: handle_file_upload(sidebar_file)

    if st.session_state.df is not None:
        df = st.session_state.df
        profile = profile_dataset(df)
        st.markdown(f"""
        <div style='background:{card_bg};border:1px solid {border_color};border-radius:10px; padding:15px; margin-top:8px; box-shadow: 0 2px 5px {shadow_css};'>
        <b style='color:{heading_color}; font-size:14px;'>{st.session_state.df_name}</b><br>
        <span style='color:{subtext_color}; font-size:12px;'>{df.shape[0]:,} rows · {df.shape[1]} cols · {profile['memory_mb']} MB</span><br>
        <div style='margin-top:8px; font-size:12px;'>
            <span style='color:#4f6ef7; font-weight:600;'>{len(profile['numeric_cols'])} numeric</span>
            <span style='color:{border_color}'> | </span>
            <span style='color:#00c4b4; font-weight:600;'>{len(profile["category_cols"])} categorical</span>
        </div></div>""", unsafe_allow_html=True)

        if st.session_state.insights:
            st.divider()
            st.markdown("### 💡 AI Insights")
            for idx, ins in enumerate(st.session_state.insights):
                icon = PTYPE_ICONS.get(ins.get("phenomenon_type",""), "◈")
                st.button(f"{icon}  {ins['title']}", key=f"sb_btn_{idx}",
                          use_container_width=True, on_click=set_view_insight, args=(idx,))

            st.divider()
            st.markdown("### 📥 Export Report")
            if FPDF_AVAILABLE:
                if st.button("📄 Generate PDF Report", use_container_width=True):
                    with st.spinner("Generating PDF (saving charts... this takes a few seconds)"):
                        summary_msg = next((m["content"] for m in st.session_state.messages if m["role"] == "assistant" and "interesting phenomena" not in m["content"]), "No summary generated.")
                        st.session_state.pdf_bytes = create_pdf_report(
                            st.session_state.df_name, profile, summary_msg,
                            st.session_state.insights, st.session_state.df
                        )

                if st.session_state.get("pdf_bytes"):
                    st.download_button(
                        label="⬇️ Download PDF Now",
                        data=st.session_state.pdf_bytes,
                        file_name=f"DataWhisper_Report_{st.session_state.df_name.replace('.csv', '')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary"
                    )
            else:
                st.info("💡 Tip: Install `fpdf` and `kaleido` via pip to enable PDF downloads.")


# ─────────────────────────────────────────────────────────────
# MAIN APP ROUTING
# ─────────────────────────────────────────────────────────────

# VIEW 1: HOME PAGE
if st.session_state.df is None or st.session_state.view == "home":
    st.markdown("""
    <div class='title-container'>
        <span class='animated-logo'>🌌</span>
        <h1 class='animated-title'>DataWhisper</h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='hero-subtitle-container'>
        <p class='hero-subtitle'>Transform raw datasets into compelling visual stories. Chat with your data, let us uncover hidden phenomenons automatically, and generate professional visual reports in seconds.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""<div style='text-align:center; margin-bottom: 10px;'><h3 style='color:{heading_color};'>1. Upload your own dataset (Primary)</h3></div>""", unsafe_allow_html=True)
        home_file = st.file_uploader("Drag and drop your CSV or TSV file here", type=["csv","tsv"], key="home_upload")
        if home_file:
            handle_file_upload(home_file)
            st.rerun()

        st.markdown(f"""<div style='text-align:center; margin: 30px 0;'><b style='color:{subtext_color};'>— OR —</b></div>""", unsafe_allow_html=True)
        st.markdown(f"""<div style='text-align:center; margin-bottom: 10px;'><h3 style='color:{heading_color};'>2. Let AI Propose Datasets</h3></div>""", unsafe_allow_html=True)

        if KAGGLE_AVAILABLE:
            st.markdown(f"<p style='text-align:center; color:{subtext_color};'>Describe what you want to analyze, and the AI will find the best datasets for you.</p>", unsafe_allow_html=True)
            user_topic = st.text_input("Your Topic / Interest:", placeholder="e.g., 'I want to analyze mental health in students'")

            if st.button("✨ Find & Propose Datasets", type="primary", use_container_width=True):
                if user_topic:
                    with st.spinner("🧠 Please wait while we searching..."):
                        try:
                            client = Groq(api_key=api_key)
                            prompt = f"Convert this user interest into a 2-4 word precise search query for Kaggle datasets. Return ONLY the keywords separated by spaces, no quotes, no explanation: '{user_topic}'"
                            resp = client.chat.completions.create(
                                messages=[{"role": "user", "content": prompt}],
                                model="llama-3.3-70b-versatile",
                                temperature=0.1, max_tokens=15
                            )
                            search_keywords = resp.choices[0].message.content.strip(" '\"\n")
                            results = kaggle_api.dataset_list(search=search_keywords, sort_by='votes', file_type='csv')
                            st.session_state.kaggle_results = results[:4]
                            if results:
                                st.success(f"**Optimized Search Query used:** `{search_keywords}`")
                            else:
                                st.warning(f"No datasets found for `{search_keywords}`. Try another topic.")
                        except Exception as e:
                            st.error(f"Search failed. Error: {e}")
                else:
                    st.warning("Please enter a topic first.")

            if st.session_state.get('kaggle_results'):
                st.markdown("<br>**🏆 Proposed Datasets:**", unsafe_allow_html=True)
                for ds in st.session_state.kaggle_results:
                    with st.container():
                        dataset_ref   = getattr(ds, 'ref', 'Unknown Ref')
                        dataset_title = getattr(ds, 'title', dataset_ref)
                        dataset_size  = getattr(ds, 'size', 'Unknown size')

                        st.markdown(f"""
                        <div style='background:{card_bg}; border:1px solid {border_color}; border-radius:10px; padding:20px; margin-bottom:12px; box-shadow:0 4px 6px {shadow_css};'>
                            <h4 style='margin:0; color:{heading_color};'>{dataset_title}</h4>
                            <p style='margin:5px 0 0; color:{subtext_color}; font-size:13px;'>
                                <b>Ref:</b> <code>{dataset_ref}</code> &nbsp;|&nbsp; <b>Size:</b> 📦 {dataset_size}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        if st.button(f"📥 Load Dataset", key=f"btn_{dataset_ref}", use_container_width=True):
                            with st.spinner("Downloading and loading dataset... This might take a moment."):
                                try:
                                    download_dir = "kaggle_downloads"
                                    os.makedirs(download_dir, exist_ok=True)
                                    kaggle_api.dataset_download_files(dataset_ref, path=download_dir, unzip=True)
                                    csv_files = glob.glob(os.path.join(download_dir, "*.csv"))
                                    if csv_files:
                                        target_csv = max(csv_files, key=os.path.getsize)
                                        st.session_state.df = pd.read_csv(target_csv)
                                        st.session_state.df_name = f"Kaggle: {dataset_title}"
                                        st.session_state.insights = []
                                        st.session_state.selected_insight = None
                                        st.session_state.llm_history = []
                                        st.session_state.messages = []
                                        st.session_state.insight_messages = []
                                        st.session_state.pdf_bytes = None
                                        st.session_state.view = "dashboard"
                                        st.rerun()
                                    else:
                                        st.error("No CSV file found in this Kaggle dataset.")
                                except Exception as e:
                                    st.error(f"Error loading Kaggle dataset: {e}")
                        st.markdown(f"<hr style='margin: 15px 0; border: none; border-top: 1px solid {border_color};'>", unsafe_allow_html=True)
        else:
            st.info("Kaggle API is not configured. Add credentials to your `.env` file.")


# VIEW 2: DEDICATED INSIGHT REPORT
elif st.session_state.view == "insight" and st.session_state.selected_insight is not None:
    idx = st.session_state.selected_insight
    ins = st.session_state.insights[idx]
    df  = st.session_state.df
    profile = profile_dataset(df)

    st.button("← Back to Chat & Dashboard", on_click=set_view_dashboard)

    st.markdown("<br>", unsafe_allow_html=True)
    icon = PTYPE_ICONS.get(ins.get("phenomenon_type",""), "◈")
    st.markdown(f"<h1>{icon} {ins['title']}</h1>", unsafe_allow_html=True)

    report_col1, report_col2 = st.columns([2.2, 1])

    with report_col1:
        fig = build_chart(ins, df)
        if fig:
            fig.update_layout(height=600, title="")
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False})
        else:
            st.error("Chart could not be rendered completely.")

    with report_col2:
        st.markdown(f"""
        <div class='analysis-panel'>
            <h3 style='margin-top: 0; color: {heading_color}; font-size: 1.2rem; font-weight: 600;'>📝 AI Deep Analysis</h3>
            <p style='color: {text_color}; line-height: 1.6;'>{ins.get("description", "No description provided.")}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📊 Key Statistics")
        for col_name in [ins.get("x_col"), ins.get("y_col"), ins.get("z_col")]:
            if col_name and col_name in df.columns and col_name in profile["numeric_cols"]:
                cs = profile["col_stats"][col_name]
                st.markdown(f"**{col_name}**")
                st.code(f"Mean: {cs['mean']}\nStd Dev: {cs['std']}\nRange: {cs['min']} to {cs['max']}")

    st.markdown("---")
    st.markdown("### 💬 Discuss this Graph")

    for msg in st.session_state.insight_messages:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"]=="assistant" else "👤"):
            st.markdown(msg["content"])

    if insight_prompt := st.chat_input("Ask a question specific to this graph..."):
        st.session_state.insight_messages.append({"role": "user", "content": insight_prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(insight_prompt)

        with st.chat_message("assistant", avatar="🤖"):
            xc = ins.get("x_col"); yc = ins.get("y_col"); cc = ins.get("color_col")
            chart_stats = {col: profile['col_stats'][col] for col in [xc, yc, cc] if col in profile.get('col_stats', {})}

            system = f"""You are an expert AI Data Analyst.
            The user is viewing: {ins.get('title')} ({ins.get('chart_type')}).
            Axes: X={xc}, Y={yc}.
            Stats: {json.dumps(chart_stats)}
            RULES: Answer directly. Use bullet points. Cite exact numbers."""

            msgs = ([{"role":"system","content":system}]
                    + [{"role": m["role"], "content": m["content"]} for m in st.session_state.insight_messages[:-1]]
                    + [{"role":"user","content":insight_prompt}])

            stream_gen = get_groq_stream(msgs, api_key)
            reply = st.write_stream(stream_gen)

        st.session_state.insight_messages.append({"role": "assistant", "content": reply})


# VIEW 3: CHAT & DASHBOARD
elif st.session_state.view == "dashboard":
    st.markdown("# 📊 Dashboard")
    st.caption("Chat with your AI analyst on the left, and view your dataset overview on the right.")

    col_chat, col_viz = st.columns([1, 1.35], gap="medium")

    df      = st.session_state.df
    profile = profile_dataset(df) if df is not None else None

    with col_chat:
        if df is not None and not st.session_state.messages and api_key:
            with st.spinner("🔍 Analyzing dataset…"):
                desc = describe_dataset(df, st.session_state.df_name, api_key)
            st.session_state.messages.append({"role":"assistant","content":desc})
            st.session_state.llm_history.append({"role":"assistant","content":desc})

        if df is not None and st.session_state.messages and not st.session_state.insights and api_key:
            with st.spinner("💡 Finding interesting phenomena…"):
                insights = generate_insights(df, api_key)
            if insights:
                st.session_state.insights = insights
                lines = "I found **5 interesting phenomena** in your dataset. Click any button below to explore it:\n\n"
                for ins in insights:
                    icon = PTYPE_ICONS.get(ins.get("phenomenon_type",""), "◈")
                    lines += f"**{icon} {ins['title']}**\n_{ins['description']}_\n\n"
                lines += "You can also ask me anything about your data in the chat below."
                st.session_state.messages.append({"role":"assistant","content":lines})
                st.session_state.llm_history.append({"role":"assistant","content":lines})
                st.rerun()

        chat_box = st.container(height=520)
        with chat_box:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"], avatar="🤖" if msg["role"]=="assistant" else "👤"):
                    st.markdown(msg["content"])

        if st.session_state.insights:
            btn_cols = st.columns(min(3, len(st.session_state.insights)))
            for idx, ins in enumerate(st.session_state.insights):
                icon  = PTYPE_ICONS.get(ins.get("phenomenon_type",""), "◈")
                label = f"{icon} {ins['title']}"
                short = label[:30]+"…" if len(label)>30 else label
                with btn_cols[idx % 3]:
                    st.button(short, key=f"qi_{idx}", use_container_width=True,
                              help=ins.get("description",""), on_click=set_view_insight, args=(idx,))

        if prompt := st.chat_input("Ask anything about your data…" if df is not None else "Upload a dataset first…", disabled=(df is None)):
            st.session_state.messages.append({"role":"user","content":prompt})
            st.session_state.llm_history.append({"role":"user","content":prompt})

            with chat_box:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(prompt)

                with st.chat_message("assistant", avatar="🤖"):
                    system = f"""You are an expert AI Data Analyst.
                    Dataset: {df.shape[0]:,} rows, columns: {list(df.columns)}
                    Key stats: {json.dumps({c:{k:v for k,v in profile['col_stats'][c].items() if k in ['mean','std','min','max','null_pct']} for c in list(df.columns)[:8]})}
                    RULES: Brief, clear, bullet points. Cite exact numbers. Under 4 sentences."""

                    msgs = [{"role":"system","content":system}] + st.session_state.llm_history[-8:]

                    stream_gen = get_groq_stream(msgs, api_key)
                    reply = st.write_stream(stream_gen)

            st.session_state.messages.append({"role":"assistant","content":reply})
            st.session_state.llm_history.append({"role":"assistant","content":reply})
            st.rerun()

    with col_viz:
        st.markdown("### 📋 Dataset Overview")
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Rows",    f"{df.shape[0]:,}")
        m2.metric("Columns", df.shape[1])
        m3.metric("Numeric", len(profile["numeric_cols"]))
        m4.metric("Missing", f"{df.isnull().sum().sum():,}")

        tab_charts, tab_data, tab_stats = st.tabs(["📊 Overview Charts","🗂 Data Preview","📈 Statistics"])
        with tab_charts:
            render_overview_charts(df, profile)
        with tab_data:
            st.dataframe(df.head(200), use_container_width=True, height=420)
        with tab_stats:
            st.dataframe(df.describe(include="all").T, use_container_width=True)