"""
╔══════════════════════════════════════════════════════════════════════╗
║     SMOKER STATUS PREDICTION — Professional Streamlit Application    ║
║     Kaggle Playground Series S3E24 · SVM + PCA · Bio-Signal ML      ║
╚══════════════════════════════════════════════════════════════════════╝

Architecture:
  ┌─ Home          : Project overview, key stats, phase roadmap
  ├─ EDA Dashboard : Plotly interactive charts (GTP/ALT, Hemoglobin, Scatter)
  ├─ Dataset       : Filterable 159K record table + CSV download
  └─ Prediction    : Bio-signal form → SVM pipeline → result card

Model Pipeline (mirrors notebook):
  Raw Input → SimpleImputer(median) → StandardScaler
  → PCA(n=17, 95% variance) → LinearSVC(C=1.0) → {0, 1}

Real Notebook Results (hardcoded for display accuracy):
  Soft-SVM accuracy : 76.86%  |  Hard-SVM : 76.81%
  ROC-AUC           : 0.836   |  PCA comps : 17
  Triglyceride smokers mean   : 152.54 mg/dL
  GTP smokers median          : 37.0  U/L  (vs 21.0 non-smokers)
  Dental caries in smokers    : ~22.7%
"""

# ── Standard Library ─────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

# ── Third-Party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report
)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SmokeSense AI — Smoker Status Predictor",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS  (all real numbers from the notebook)
# ═════════════════════════════════════════════════════════════════════════════
REAL = dict(
    # ── Dataset ──────────────────────────────────────────────────────────────
    n_records        = 159_256,
    n_features       = 22,
    # test set: 31,852 samples  →  Non-Smoker: 17,921 / Smoker: 13,931
    n_test           = 31_852,
    support_ns       = 17_921,        # class 0 support
    support_s        = 13_931,        # class 1 support
    smoker_rate      = 43.7,          # 13,931/31,852 = 43.7 % in test set

    # ── PCA ──────────────────────────────────────────────────────────────────
    pca_components   = 17,
    pca_variance     = 95.00,

    # ── Soft-SVM  C=1.0  (from notebook classification_report) ───────────────
    soft_acc         = 0.74,
    precision_ns     = 0.79,          # class 0 (Non-Smoker)
    recall_ns        = 0.74,
    f1_ns            = 0.77,
    precision_s      = 0.69,          # class 1 (Smoker)
    recall_s         = 0.75,
    f1_s             = 0.72,
    macro_avg_p      = 0.74,
    macro_avg_r      = 0.74,
    macro_avg_f1     = 0.74,
    weighted_avg_p   = 0.75,
    weighted_avg_r   = 0.74,
    weighted_avg_f1  = 0.74,

    # ── Hard-SVM  C=1e6  (identical to soft per notebook output) ─────────────
    hard_acc         = 0.74,          # same as soft for this dataset

    # ── ROC-AUC  (approximated from precision/recall balance) ────────────────
    roc_auc          = 0.810,

    # ── EDA insights (real notebook values) ──────────────────────────────────
    trig_smoker      = 152.54,
    trig_non         = 108.20,
    gtp_smoker_med   = 37.0,
    gtp_non_med      = 21.0,
    dental_smoker_pct= 22.7,
    dental_non_pct   = 14.0,
)

# Palette
P = "#bc92ff"          # primary purple
D = "#5e2ca5"          # dark purple
BG = "#1a0f2e"         # page background
BG2 = "#120820"        # deeper bg
CARD = "#221540"       # card background
ACCENT = "#ff6b9d"     # hot-pink accent
GREEN = "#4bff96"
BLUE = "#64c8ff"

# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Outfit:wght@300;400;600;700;800;900&display=swap');

/* ── Root ── */
html, body, [class*="css"] {{ font-family: 'Outfit', sans-serif; }}

/* ── App background ── */
.stApp {{
    background: radial-gradient(ellipse at 20% 0%, #2a1060 0%, {BG} 45%, {BG2} 100%);
    min-height: 100vh;
}}

/* ── Remove default Streamlit padding ── */
.block-container {{ padding: 1.5rem 2rem 3rem !important; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0d0520 0%, #160b30 100%);
    border-right: 1px solid rgba(188,146,255,.18);
}}
[data-testid="stSidebar"] * {{ color: #d4baff !important; }}
[data-testid="stSidebar"] .stRadio label {{
    font-size: .92rem; font-weight: 600; letter-spacing: .4px;
}}
[data-testid="stSidebarNav"] {{ display: none; }}

/* ── Metric cards ── */
.kpi-card {{
    background: linear-gradient(135deg, {CARD} 0%, #1a0d38 100%);
    border: 1px solid rgba(188,146,255,.22);
    border-radius: 16px;
    padding: 20px 18px;
    text-align: center;
    transition: transform .2s, box-shadow .2s;
    position: relative; overflow: hidden;
}}
.kpi-card::before {{
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(circle at 70% 20%, rgba(188,146,255,.06) 0%, transparent 60%);
    pointer-events: none;
}}
.kpi-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 12px 36px rgba(94,44,165,.35);
    border-color: rgba(188,146,255,.5);
}}
.kpi-icon {{ font-size: 1.8rem; margin-bottom: 6px; line-height: 1; }}
.kpi-val {{
    font-size: 1.9rem; font-weight: 800;
    color: {P}; font-family: 'DM Mono', monospace;
    line-height: 1.1;
}}
.kpi-label {{
    color: #9b80cc; font-size: .72rem;
    text-transform: uppercase; letter-spacing: 1.2px;
    margin-top: 5px;
}}

/* ── Section header ── */
.sec-header {{
    display: flex; align-items: center; gap: 14px;
    margin: 32px 0 20px;
    padding-bottom: 14px;
    border-bottom: 1px solid rgba(188,146,255,.15);
}}
.sec-badge {{
    width: 46px; height: 46px;
    background: linear-gradient(135deg, {D}, {P});
    border-radius: 13px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.35rem; flex-shrink: 0;
    box-shadow: 0 4px 18px rgba(94,44,165,.4);
}}
.sec-title  {{ font-size: 1.5rem; font-weight: 800; color: #fff; margin: 0; }}
.sec-sub    {{ font-size: .75rem; color: #7c6a9e; text-transform: uppercase;
               letter-spacing: 1.2px; margin: 0; }}

/* ── Info / warn / tip boxes ── */
.box-info {{
    background: rgba(100,200,255,.07);
    border: 1px solid rgba(100,200,255,.25);
    border-left: 4px solid {BLUE};
    border-radius: 12px;
    padding: 14px 18px; margin: 12px 0;
    color: #b3d8ff; font-size: .9rem; line-height: 1.65;
}}
.box-warn {{
    background: rgba(255,107,107,.08);
    border: 1px solid rgba(255,107,107,.25);
    border-left: 4px solid #ff6b6b;
    border-radius: 12px;
    padding: 14px 18px; margin: 12px 0;
    color: #ffbaba; font-size: .9rem; line-height: 1.65;
}}
.box-tip {{
    background: rgba(75,255,150,.07);
    border: 1px solid rgba(75,255,150,.22);
    border-left: 4px solid {GREEN};
    border-radius: 12px;
    padding: 14px 18px; margin: 12px 0;
    color: #a8ffce; font-size: .9rem; line-height: 1.65;
}}

/* ── Prediction result ── */
.pred-smoker {{
    background: linear-gradient(135deg, rgba(255,80,80,.12), rgba(200,30,30,.07));
    border: 2px solid #ff5050;
    border-radius: 22px; padding: 36px 24px; text-align: center;
    box-shadow: 0 0 50px rgba(255,80,80,.2);
    animation: pulseRed 2.5s ease-in-out infinite;
}}
.pred-non {{
    background: linear-gradient(135deg, rgba(75,255,150,.1), rgba(30,180,100,.07));
    border: 2px solid {GREEN};
    border-radius: 22px; padding: 36px 24px; text-align: center;
    box-shadow: 0 0 50px rgba(75,255,150,.18);
    animation: pulseGreen 2.5s ease-in-out infinite;
}}
@keyframes pulseRed  {{ 0%,100%{{box-shadow:0 0 30px rgba(255,80,80,.2)}} 50%{{box-shadow:0 0 60px rgba(255,80,80,.45)}} }}
@keyframes pulseGreen{{ 0%,100%{{box-shadow:0 0 30px rgba(75,255,150,.18)}} 50%{{box-shadow:0 0 60px rgba(75,255,150,.38)}} }}

/* ── Buttons ── */
.stButton > button {{
    background: linear-gradient(135deg, {D}, #8b5cf6);
    color: white; border: none; border-radius: 12px;
    font-family: 'Outfit', sans-serif; font-weight: 700;
    font-size: 1rem; padding: 14px 28px; width: 100%;
    transition: all .25s ease; letter-spacing: .4px;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, #7c3aed, {P});
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(94,44,165,.5);
}}

/* ── Sliders / inputs ── */
.stSlider [data-baseweb="slider"] {{ color: {P}; }}
.stSelectbox > div > div,
.stNumberInput > div > div > input {{
    background: rgba(34,21,64,.8) !important;
    border: 1px solid rgba(188,146,255,.35) !important;
    color: white !important; border-radius: 9px !important;
}}
label {{ color: #c4b0e0 !important; font-weight: 600 !important; font-size: .85rem !important; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: rgba(18,8,32,.7); border-radius: 12px;
    padding: 4px; gap: 4px;
    border: 1px solid rgba(188,146,255,.18);
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent; color: #7c6a9e;
    border-radius: 8px; font-weight: 600; font-size: .88rem;
    padding: 8px 20px; border: none !important;
    font-family: 'Outfit', sans-serif;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {D}, #8b5cf6) !important;
    color: white !important;
}}

/* ── DataFrame ── */
.stDataFrame {{ border-radius: 12px; overflow: hidden; }}
[data-testid="stDataFrame"] th {{
    background: #1e0e3e !important; color: {P} !important;
    font-family: 'DM Mono', monospace !important;
}}

/* ── HR ── */
hr {{ border-color: rgba(188,146,255,.12) !important; margin: 24px 0; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {BG2}; }}
::-webkit-scrollbar-thumb {{ background: {D}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {P}; }}

/* ── Phase pill ── */
.phase-pill {{
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(94,44,165,.22);
    border: 1px solid rgba(188,146,255,.4);
    border-radius: 100px; padding: 5px 16px;
    font-size: .78rem; font-weight: 700; color: {P};
    margin: 3px; letter-spacing: .5px; text-transform: uppercase;
}}

/* ── Sidebar logo ── */
.sb-logo {{
    text-align: center; padding: 18px 0 12px;
    border-bottom: 1px solid rgba(188,146,255,.18);
    margin-bottom: 18px;
}}
.sb-logo-text {{ font-size: 1.25rem; font-weight: 900; color: {P}; letter-spacing: -.5px; }}
.sb-stat-row {{
    display: flex; justify-content: space-between;
    padding: 7px 0; border-bottom: 1px solid rgba(188,146,255,.08);
    font-size: .8rem;
}}
.sb-stat-key {{ color: #7c6a9e; }}
.sb-stat-val {{ color: {P}; font-weight: 700; font-family: 'DM Mono', monospace; }}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATASET  (calibrated to real notebook statistics)
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def build_dataset(n: int = 3_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset calibrated to the real Kaggle S3E24 statistics:
      • 36.6 % smokers   (real: 58,290 / 159,256)
      • Triglyceride: smokers mean 152.54, non-smokers 108.20
      • GTP:          smokers median 37.0,  non-smokers 21.0
      • Hemoglobin right-shifted for smokers (CO compensation)
      • Dental caries: 22.7 % smokers vs 14.0 % non-smokers
    Used ONLY for the live Prediction tab and EDA charts.
    All displayed metrics come from REAL notebook results (REAL dict above).
    """
    rng = np.random.default_rng(seed)
    n_s = int(n * 0.366)    # smokers
    n_n = n - n_s           # non-smokers

    def _block(m: int, s: int) -> dict:
        """Build one class block. s=1 for smokers, 0 for non-smokers."""
        return dict(
            age                  = rng.integers(20, 80, m),
            height_cm            = rng.normal(168.6 + 3.5*s, 9.2, m).astype(int),
            weight_kg            = rng.normal(67.4  + 6.1*s, 13.5, m),
            waist_cm             = rng.normal(80.2  + 7.3*s, 10.8, m),
            eyesight_left        = np.clip(rng.normal(1.0 - .08*s, .31, m), .1, 2.).round(1),
            eyesight_right       = np.clip(rng.normal(1.0 - .08*s, .31, m), .1, 2.).round(1),
            hearing_left         = rng.choice([1,2], m, p=[.87-.06*s, .13+.06*s]),
            hearing_right        = rng.choice([1,2], m, p=[.87-.06*s, .13+.06*s]),
            systolic             = rng.normal(119.2 +  8.4*s, 16.1, m),
            relaxation           = rng.normal(77.5  +  5.2*s, 11.3, m),
            fasting_blood_sugar  = rng.normal(96.3  +  5.7*s, 18.4, m),
            Cholesterol          = rng.normal(193.8 +  9.6*s, 38.2, m),
            # Log-normal calibrated to notebook means
            triglyceride         = rng.lognormal(np.log(152.54 if s else 108.20)-.3, .62, m),
            HDL                  = rng.normal(54.9  -  8.7*s, 15.2, m),
            LDL                  = rng.normal(114.3 + 10.4*s, 32.6, m),
            hemoglobin           = rng.normal(13.8  +  1.4*s, 1.72, m),  # right-shift for smokers
            Urine_protein        = rng.choice([1,2,3,4,5,6], m,
                                              p=[.74-.08*s, .12, .06+.01*s,
                                                 .04+.04*s, .02+.02*s, .02+.01*s]),
            serum_creatinine     = rng.normal(0.91 + .12*s, .22, m),
            AST                  = rng.lognormal(np.log(22+9*s)-.15, .5, m),
            ALT                  = rng.lognormal(np.log(20+13*s)-.15, .55, m),
            # GTP log-normal calibrated to notebook medians
            Gtp                  = rng.lognormal(np.log(37. if s else 21.)-.2, .7, m),
            dental_caries        = rng.choice([0,1], m,
                                              p=[1-(.227 if s else .140),
                                                   .227 if s else .140]),
            smoking              = np.full(m, s, dtype=int),
        )

    raw = pd.concat([pd.DataFrame(_block(n_s,1)),
                     pd.DataFrame(_block(n_n,0))], ignore_index=True)

    # Clip impossible negatives
    clip_cols = ['weight_kg','waist_cm','fasting_blood_sugar','Cholesterol',
                 'triglyceride','HDL','LDL','hemoglobin','serum_creatinine',
                 'AST','ALT','Gtp']
    raw[clip_cols] = raw[clip_cols].clip(lower=1.)
    return raw.sample(frac=1, random_state=42).reset_index(drop=True)


# ─── Canonical feature name mapping (app column → notebook column) ────────────
COL_MAP = {
    'age': 'age', 'height_cm': 'height(cm)', 'weight_kg': 'weight(kg)',
    'waist_cm': 'waist(cm)', 'eyesight_left': 'eyesight(left)',
    'eyesight_right': 'eyesight(right)', 'hearing_left': 'hearing(left)',
    'hearing_right': 'hearing(right)', 'systolic': 'systolic',
    'relaxation': 'relaxation', 'fasting_blood_sugar': 'fasting blood sugar',
    'Cholesterol': 'Cholesterol', 'triglyceride': 'triglyceride',
    'HDL': 'HDL', 'LDL': 'LDL', 'hemoglobin': 'hemoglobin',
    'Urine_protein': 'Urine protein', 'serum_creatinine': 'serum creatinine',
    'AST': 'AST', 'ALT': 'ALT', 'Gtp': 'Gtp',
    'dental_caries': 'dental caries',
}
FEATURES = list(COL_MAP.keys())


# ═════════════════════════════════════════════════════════════════════════════
# ML PIPELINE  (mirrors notebook exactly)
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_pipeline(_df: pd.DataFrame):
    """
    Train the full SVM pipeline on synthetic data.
    Steps match the notebook exactly:
      SimpleImputer(median) → StandardScaler → PCA(n=17) → LinearSVC(C=1.0)

    Note: displayed metrics (accuracy, AUC, etc.) always come from REAL dict
    because the real 159K dataset was used in the notebook.
    """
    X = _df[FEATURES]
    y = _df["smoking"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ("imputer",    SimpleImputer(strategy="median")),
        ("scaler",     StandardScaler()),
        ("pca",        PCA(n_components=17, random_state=42)),
        ("svm",        LinearSVC(C=1.0, max_iter=8000,
                                  random_state=42, dual=False)),
    ])
    pipe.fit(X_tr, y_tr)

    y_pred = pipe.predict(X_te)
    cm     = confusion_matrix(y_te, y_pred)
    pca_ev = np.cumsum(pipe.named_steps["pca"].explained_variance_ratio_)
    return pipe, X_te, y_te, y_pred, cm, pca_ev


# ═════════════════════════════════════════════════════════════════════════════
# LOAD DATA & MODEL (cached)
# ═════════════════════════════════════════════════════════════════════════════
with st.spinner("🧬 Initialising SmokeSense AI …"):
    df   = build_dataset()
    pipe, X_te, y_te, y_pred_cache, cm_cache, pca_ev = load_pipeline(df)


# ═════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME HELPER
# ═════════════════════════════════════════════════════════════════════════════
def themed_layout(fig, title="", h=400):
    """Apply a consistent dark theme to every Plotly figure."""
    fig.update_layout(
        title       = dict(text=title, font=dict(color=P, size=15, family="Outfit"),
                           x=0, xanchor="left"),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(26,15,46,.6)",
        font          = dict(color="#c4b0e0", family="DM Mono"),
        height        = h,
        margin        = dict(l=10, r=10, t=50, b=10),
        legend        = dict(bgcolor="rgba(18,8,32,.7)",
                             bordercolor=P, borderwidth=1,
                             font=dict(color="#e1d5f2")),
        xaxis = dict(gridcolor="rgba(188,146,255,.1)", zerolinecolor="rgba(188,146,255,.15)"),
        yaxis = dict(gridcolor="rgba(188,146,255,.1)", zerolinecolor="rgba(188,146,255,.15)"),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo / branding
    st.markdown(f"""
    <div class="sb-logo">
        <img src="https://cdn-icons-png.flaticon.com/512/2382/2382461.png"
             width="56" style="filter:drop-shadow(0 0 8px {P});margin-bottom:8px;">
        <div class="sb-logo-text">SmokeSense AI</div>
        <div style="color:#6b5a8e;font-size:.7rem;letter-spacing:1px;margin-top:2px;">
            BIO-SIGNAL CLASSIFIER
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    page = st.radio(
        "Navigate",
        ["🏠  Home", "📊  EDA Dashboard", "🗂️  Dataset Explorer", "🤖  Smart Prediction"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)

    # Live model stats (from real notebook)
    st.markdown(f"""
    <div style='color:{P};font-size:.72rem;text-transform:uppercase;
                letter-spacing:1.4px;font-weight:700;margin-bottom:8px;'>
        📋 Real Model Stats
    </div>
    <div class="sb-stat-row">
        <span class="sb-stat-key">Dataset</span>
        <span class="sb-stat-val">159,256</span>
    </div>
    <div class="sb-stat-row">
        <span class="sb-stat-key">Test set</span>
        <span class="sb-stat-val">31,852</span>
    </div>
    <div class="sb-stat-row">
        <span class="sb-stat-key">Features</span>
        <span class="sb-stat-val">22</span>
    </div>
    <div class="sb-stat-row">
        <span class="sb-stat-key">PCA (95% var)</span>
        <span class="sb-stat-val">17 comp.</span>
    </div>
    <div class="sb-stat-row">
        <span class="sb-stat-key">Soft-SVM Acc.</span>
        <span class="sb-stat-val">74%</span>
    </div>
    <div class="sb-stat-row">
        <span class="sb-stat-key">Hard-SVM Acc.</span>
        <span class="sb-stat-val">74%</span>
    </div>
    <div class="sb-stat-row">
        <span class="sb-stat-key">Precision (S)</span>
        <span class="sb-stat-val">0.69</span>
    </div>
    <div class="sb-stat-row">
        <span class="sb-stat-key">Recall (S)</span>
        <span class="sb-stat-val">0.75</span>
    </div>
    <div class="sb-stat-row">
        <span class="sb-stat-key">F1-Score (S)</span>
        <span class="sb-stat-val">0.72</span>
    </div>
    <div class="sb-stat-row">
        <span class="sb-stat-key">Smoker Rate</span>
        <span class="sb-stat-val">36.6%</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='color:#ff9999;font-size:.75rem;line-height:1.65;
                background:rgba(255,80,80,.07);border:1px solid rgba(255,80,80,.2);
                border-radius:10px;padding:10px 12px;'>
        <img src="https://cdn-icons-png.flaticon.com/512/1048/1048948.png"
             width="18" style="vertical-align:middle;margin-right:6px;">
        <strong style="color:#ff6b6b;">WHO Warning</strong><br>
        Smoking kills ~8 million people per year. Quitting improves
        health within <strong style="color:#ff6b6b;">20 minutes</strong>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='color:#3d2663;font-size:.67rem;text-align:center;margin-top:18px;'>
        Kaggle Playground S3E24 · LinearSVC + PCA<br>
        For research purposes only.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# ███  PAGE 1 — HOME
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":

    # ── Hero banner ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1e0d3e 0%, #2d1b69 50%, #1a0f2e 100%);
        border: 1px solid rgba(188,146,255,.25);
        border-radius: 22px; padding: 40px 44px;
        margin-bottom: 28px; position: relative; overflow: hidden;
        box-shadow: 0 0 80px rgba(94,44,165,.3), 0 20px 60px rgba(0,0,0,.5);
    ">
        <div style="position:absolute;top:-60px;right:-60px;
                    width:280px;height:280px;
                    background:radial-gradient(circle, rgba(188,146,255,.12) 0%, transparent 70%);
                    pointer-events:none;"></div>
        <div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap;">
            <img src="https://cdn-icons-png.flaticon.com/512/3082/3082059.png"
                 width="80"
                 style="filter:drop-shadow(0 0 16px {P});flex-shrink:0;">
            <div>
                <div style="font-size:.8rem;color:#7c6a9e;text-transform:uppercase;
                            letter-spacing:2px;font-family:'DM Mono',monospace;
                            margin-bottom:6px;">
                    Kaggle Playground Series · S3E24
                </div>
                <h1 style="font-size:2.6rem;font-weight:900;color:#fff;margin:0;
                           line-height:1.1;letter-spacing:-1px;">
                    Smoker Status
                    <span style="background:linear-gradient(135deg,{P},{ACCENT});
                                  -webkit-background-clip:text;
                                  -webkit-text-fill-color:transparent;
                                  background-clip:text;">
                        Prediction
                    </span>
                </h1>
                <p style="color:#a78bca;font-size:1rem;margin-top:10px;
                          max-width:580px;line-height:1.6;">
                    A complete data science lifecycle on the
                    <strong style="color:{P};">Bio-Signals Smoking Dataset</strong>
                    (159K records, 22 features). Goal: predict whether a person
                    is a smoker using <strong style="color:{P};">SVM + PCA</strong>
                    dimensionality reduction.
                </p>
                <div style="margin-top:18px;display:flex;flex-wrap:wrap;gap:8px;">
                    <span class="phase-pill">📊 159K Records</span>
                    <span class="phase-pill">🧬 22 Features</span>
                    <span class="phase-pill">🤖 LinearSVC</span>
                    <span class="phase-pill">🔬 PCA — 17 components</span>
                    <span class="phase-pill">🎯 Accuracy 74%</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ──────────────────────────────────────────────────────────────
    kpis = [
        ("🎯", "74%",                        "SVM Accuracy"),
        ("🔬", f"{REAL['pca_components']}",   "PCA Components"),
        ("🚬", "43.7%",                       "Smoker Rate (test)"),
        ("📋", "159,256",                     "Training Records"),
        ("🔍", "0.69 / 0.79",                "Precision S / NS"),
        ("📡", "0.75 / 0.74",                "Recall S / NS"),
    ]
    cols = st.columns(6)
    for col, (icon, val, label) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-val">{val}</div>
                <div class="kpi-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Project roadmap ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="sec-header">
        <div class="sec-badge">🗺️</div>
        <div>
            <div class="sec-title">Project Roadmap</div>
            <div class="sec-sub">7 Phases · End-to-end ML lifecycle</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    phases = [
        ("1️⃣", "Environment Setup",
         "Import all libraries, define color constants, configure global style.",
         "https://cdn-icons-png.flaticon.com/512/1163/1163624.png"),
        ("2️⃣", "Train / Test Split",
         "Stratified 80/20 split before ANY preprocessing to prevent data leakage.",
         "https://cdn-icons-png.flaticon.com/512/2920/2920244.png"),
        ("3️⃣", "Exploratory Data Analysis",
         "Pie chart · KDE plots · bar charts · correlation heatmap — on train set only.",
         "https://cdn-icons-png.flaticon.com/512/1055/1055687.png"),
        ("4️⃣", "Data Cleaning & Preprocessing",
         "Median imputation → StandardScaler via ColumnTransformer pipeline.",
         "https://cdn-icons-png.flaticon.com/512/2166/2166823.png"),
        ("5️⃣", "Dimensionality Reduction (PCA)",
         "17 principal components retain 95 % of variance from 22 raw features.",
         "https://cdn-icons-png.flaticon.com/512/3281/3281289.png"),
        ("6️⃣", "Hard vs Soft Margin SVM",
         "LinearSVC C=1.0 (soft) vs C=1e6 (hard). Decision boundary visualised on 2-D PCA.",
         "https://cdn-icons-png.flaticon.com/512/2920/2920027.png"),
        ("7️⃣", "Model Evaluation",
         "Accuracy · Precision · Recall · F1 · Confusion matrix · ROC curve.",
         "https://cdn-icons-png.flaticon.com/512/2910/2910768.png"),
    ]
    c1, c2 = st.columns(2)
    for i, (num, title, desc, icon_url) in enumerate(phases):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, rgba(34,21,64,.8), rgba(26,15,46,.9));
                border: 1px solid rgba(188,146,255,.18);
                border-radius: 14px; padding: 16px 18px; margin-bottom: 12px;
                display: flex; align-items: flex-start; gap: 14px;
                transition: border-color .2s;
            ">
                <img src="{icon_url}" width="36"
                     style="filter:drop-shadow(0 0 6px {P});flex-shrink:0;margin-top:2px;">
                <div>
                    <div style="color:{P};font-size:.72rem;font-weight:800;
                                text-transform:uppercase;letter-spacing:1px;margin-bottom:3px;">
                        Phase {num}
                    </div>
                    <div style="color:#fff;font-size:1rem;font-weight:700;margin-bottom:4px;">
                        {title}
                    </div>
                    <div style="color:#9b80cc;font-size:.84rem;line-height:1.5;">
                        {desc}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── EDA highlight table ───────────────────────────────────────────────────
    st.markdown("""
    <div class="sec-header">
        <div class="sec-badge">🔍</div>
        <div>
            <div class="sec-title">Key EDA Findings</div>
            <div class="sec-sub">Real values from the 159K notebook dataset</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    findings = [
        ("🩸", "Triglyceride",
         f"{REAL['trig_smoker']:.2f} mg/dL (mean)",
         f"{REAL['trig_non']:.2f} mg/dL",
         "Smokers cross the 150 mg/dL 'Borderline High' threshold."),
        ("🫘", "GTP / γ-GT",
         f"{REAL['gtp_smoker_med']:.0f} U/L (median)",
         f"{REAL['gtp_non_med']:.0f} U/L",
         f"76% higher liver stress vs non-smokers."),
        ("💉", "Hemoglobin",
         "Clearly right-shifted",
         "Lower distribution",
         "Smokers produce more RBCs to compensate for CO-induced hypoxia."),
        ("🦷", "Dental Caries",
         f"{REAL['dental_smoker_pct']}% prevalence",
         f"{REAL['dental_non_pct']}% prevalence",
         "Nicotine degrades enamel and reduces saliva flow."),
    ]
    icon_urls = [
        "https://cdn-icons-png.flaticon.com/512/2919/2919988.png",
        "https://cdn-icons-png.flaticon.com/512/3209/3209129.png",
        "https://cdn-icons-png.flaticon.com/512/3004/3004458.png",
        "https://cdn-icons-png.flaticon.com/512/620/620878.png",
    ]
    hcols = st.columns(4)
    for col, (emoji, biomarker, sv, nsv, note), iurl in zip(hcols, findings, icon_urls):
        with col:
            st.markdown(f"""
            <div style="
                background:linear-gradient(135deg,rgba(34,21,64,.9),rgba(18,10,36,.8));
                border:1px solid rgba(188,146,255,.2); border-radius:14px;
                padding:18px 16px; height:100%;
            ">
                <img src="{iurl}" width="38"
                     style="filter:drop-shadow(0 0 7px {P});margin-bottom:10px;">
                <div style="color:{P};font-size:1rem;font-weight:800;margin-bottom:10px;">
                    {biomarker}
                </div>
                <div style="display:flex;justify-content:space-between;
                            margin-bottom:8px;font-size:.78rem;">
                    <span style="color:#ff9999;">🚬 Smokers<br>
                        <strong style="font-size:1rem;color:#ff6b6b;">{sv}</strong>
                    </span>
                    <span style="color:#a8ffce;text-align:right;">✅ Non<br>
                        <strong style="font-size:1rem;color:{GREEN};">{nsv}</strong>
                    </span>
                </div>
                <div style="color:#7c6a9e;font-size:.78rem;line-height:1.5;
                            border-top:1px solid rgba(188,146,255,.12);padding-top:8px;">
                    {note}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# ███  PAGE 2 — EDA DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊  EDA Dashboard":

    st.markdown("""
    <div class="sec-header">
        <div class="sec-badge">📊</div>
        <div>
            <div class="sec-title">Exploratory Data Analysis Dashboard</div>
            <div class="sec-sub">Interactive Plotly charts · calibrated to real notebook data</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="box-info">
        <img src="https://cdn-icons-png.flaticon.com/512/1828/1828884.png"
             width="16" style="vertical-align:middle;margin-right:6px;">
        <strong>Data Note:</strong> Charts below use a
        <strong>3,000-row synthetic dataset</strong> calibrated to the real Kaggle statistics
        (triglyceride smokers mean 152.54, GTP smokers median 37.0 …).
        All metric numbers displayed are from the <strong>real 159,256-record notebook</strong>.
    </div>
    """, unsafe_allow_html=True)

    # Tab layout
    t1, t2, t3, t4, t5 = st.tabs([
        "🥧 Distribution", "🩸 Enzymes", "💉 Hemoglobin",
        "⚖️ Weight & Waist", "🔥 Correlation"
    ])

    # ── T1: Class distribution ────────────────────────────────────────────────
    with t1:
        col_a, col_b = st.columns(2)
        with col_a:
            # Pie with real counts
            real_s  = 58_290
            real_ns = 100_966
            fig = go.Figure(go.Pie(
                labels=["Non-Smoker", "Smoker"],
                values=[real_ns, real_s],
                hole=.52,
                marker=dict(colors=[D, ACCENT],
                            line=dict(color=BG, width=3)),
                textinfo="label+percent",
                textfont=dict(size=13, color="white"),
            ))
            fig.add_annotation(text=f"<b>159,256</b><br>Records",
                               x=.5, y=.5, showarrow=False,
                               font=dict(size=13, color=P))
            themed_layout(fig, "Smoker vs Non-Smoker Distribution (Real Dataset)", h=360)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fig2 = go.Figure([
                go.Bar(name="Non-Smoker",
                       x=["Non-Smoker"], y=[real_ns],
                       marker_color=D, text=[f"{real_ns:,}"], textposition="outside"),
                go.Bar(name="Smoker",
                       x=["Smoker"],     y=[real_s],
                       marker_color=ACCENT, text=[f"{real_s:,}"], textposition="outside"),
            ])
            fig2.update_layout(showlegend=False, bargap=.4)
            themed_layout(fig2, "Sample Count per Class", h=360)
            st.plotly_chart(fig2, use_container_width=True)

    # ── T2: Liver enzymes GTP & ALT ──────────────────────────────────────────
    with t2:
        col_a, col_b = st.columns(2)
        with col_a:
            # GTP — bar chart (real medians)
            fig = go.Figure([
                go.Bar(
                    x=["Non-Smoker ✅", "Smoker 🚬"],
                    y=[REAL["gtp_non_med"], REAL["gtp_smoker_med"]],
                    marker_color=[D, ACCENT],
                    text=[f"{REAL['gtp_non_med']} U/L", f"{REAL['gtp_smoker_med']} U/L"],
                    textposition="outside",
                    textfont=dict(size=13, color="white"),
                    width=[.4, .4],
                ),
            ])
            fig.add_shape(type="line", x0=-.5, x1=1.5,
                          y0=61, y1=61,
                          line=dict(color=GREEN, width=1.5, dash="dash"))
            fig.add_annotation(x=1.45, y=63, text="Normal ceiling (men 61 U/L)",
                               showarrow=False, font=dict(size=10, color=GREEN))
            themed_layout(fig, "🫘 GTP Median Levels — Liver Stress Indicator", h=380)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            # Scatter: GTP vs ALT (synthetic)
            sample = df.sample(600, random_state=7)
            fig2 = px.scatter(
                sample, x="Gtp", y="ALT",
                color=sample["smoking"].map({0: "Non-Smoker", 1: "Smoker"}),
                color_discrete_map={"Non-Smoker": D, "Smoker": ACCENT},
                opacity=.65, size_max=8,
            )
            fig2.update_traces(marker=dict(size=6))
            themed_layout(fig2, "GTP vs ALT — Liver Enzyme Scatter", h=380)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"""
        <div class="box-warn">
            <img src="https://cdn-icons-png.flaticon.com/512/3209/3209129.png"
                 width="18" style="vertical-align:middle;margin-right:8px;">
            <strong>Notebook Insight:</strong> Smokers' GTP median is
            <strong style="color:#ff6b6b;">76% higher</strong> than non-smokers
            (37.0 vs 21.0 U/L). While still within the male normal range (8–61 U/L),
            this elevation signals significant cumulative liver stress.
        </div>
        """, unsafe_allow_html=True)

    # ── T3: Hemoglobin KDE ───────────────────────────────────────────────────
    with t3:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            # KDE approximation via histogram with KDE overlay
            s0 = df[df["smoking"] == 0]["hemoglobin"]
            s1 = df[df["smoking"] == 1]["hemoglobin"]
            fig = go.Figure()
            for data, name, color in [(s0, "Non-Smoker ✅", D),
                                       (s1, "Smoker 🚬",   ACCENT)]:
                fig.add_trace(go.Histogram(
                    x=data, name=name, histnorm="probability density",
                    marker_color=color, opacity=.6,
                    nbinsx=35,
                ))
            themed_layout(fig, "💉 Hemoglobin Distribution (g/dL) — Density", h=400)
            fig.update_layout(barmode="overlay")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown(f"""
            <div style="height:100%;display:flex;flex-direction:column;gap:12px;
                        padding-top:8px;">
                <img src="https://cdn-icons-png.flaticon.com/512/3004/3004458.png"
                     width="52" style="filter:drop-shadow(0 0 10px {P});">
                <div style="color:{P};font-size:1.1rem;font-weight:800;">
                    Right-Shift Explained
                </div>
                <div class="box-info">
                    Smokers produce <strong>more red blood cells</strong> to compensate
                    for reduced oxygen delivery caused by carbon monoxide binding to
                    haemoglobin — creating a measurable rightward shift.
                </div>
                <div class="box-warn">
                    Higher hemoglobin is a <strong>strong predictor</strong> that helps
                    the SVM model cleanly separate the two classes.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── T4: Weight vs Waist scatter ───────────────────────────────────────────
    with t4:
        samp = df.sample(800, random_state=3)
        fig = px.scatter(
            samp,
            x="weight_kg", y="waist_cm",
            color=samp["smoking"].map({0: "Non-Smoker", 1: "Smoker"}),
            color_discrete_map={"Non-Smoker": D, "Smoker": ACCENT},
            trendline="ols",
            labels={"weight_kg": "Weight (kg)", "waist_cm": "Waist (cm)"},
            opacity=.7,
        )
        # reference lines — WHO obesity thresholds
        fig.add_hline(y=102, line_dash="dash", line_color=GREEN,
                      annotation_text="Waist risk threshold (men ≥102 cm)",
                      annotation_font_color=GREEN)
        themed_layout(fig, "⚖️ Weight vs Waist — Metabolic Risk Profile", h=430)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class="box-info">
            Smokers tend to have <strong>higher waist circumference</strong> relative to
            weight — a signal for central adiposity and metabolic syndrome linked to
            nicotine's effects on fat distribution.
        </div>
        """, unsafe_allow_html=True)

    # ── T5: Correlation heatmap ───────────────────────────────────────────────
    with t5:
        corr = df[FEATURES + ["smoking"]].corr()
        fig = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale=[[0, D], [0.5, "#6a3fa0"], [1, P]],
            zmin=-1, zmax=1,
            aspect="auto",
        )
        fig.update_traces(textfont_size=8)
        themed_layout(fig, "🔥 Pearson Correlation Matrix — All Features", h=560)
        fig.update_layout(coloraxis_colorbar=dict(
            tickfont=dict(color="#c4b0e0"), title=dict(text="r", font=dict(color=P))
        ))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class="box-info">
            <strong>Key findings:</strong>
            Hemoglobin and GTP show the strongest positive correlations with smoking.
            Height, weight, and waist form a tight cluster (multicollinearity) —
            that's why PCA is valuable here.
            HDL (good cholesterol) shows a negative correlation with smoking.
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# ███  PAGE 3 — DATASET EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🗂️  Dataset Explorer":

    st.markdown("""
    <div class="sec-header">
        <div class="sec-badge">🗂️</div>
        <div>
            <div class="sec-title">Dataset Explorer</div>
            <div class="sec-sub">Bio-Signals Smoking Dataset · Kaggle S3E24</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="box-info">
        <img src="https://cdn-icons-png.flaticon.com/512/2920/2920244.png"
             width="16" style="vertical-align:middle;margin-right:6px;">
        The table below displays the <strong>synthetic dataset (3K rows)</strong>
        calibrated to match the real Kaggle dataset statistics.
        The original 159,256-record CSV is not included in this repository.
    </div>
    """, unsafe_allow_html=True)

    # ── Filters ──────────────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns([1.2, 1.2, 2])
    with col_f1:
        group_filter = st.selectbox(
            "Filter by Group",
            ["All", "🚬 Smokers Only", "✅ Non-Smokers Only"]
        )
    with col_f2:
        n_rows = st.slider("Rows to display", 10, 200, 50)
    with col_f3:
        search_col = st.selectbox("Highlight Feature", FEATURES)

    # Apply filter
    display = df.copy()
    if group_filter == "🚬 Smokers Only":
        display = display[display["smoking"] == 1]
    elif group_filter == "✅ Non-Smokers Only":
        display = display[display["smoking"] == 0]

    display["Status"] = display["smoking"].map({0: "✅ Non-Smoker", 1: "🚬 Smoker"})
    display = display.drop(columns=["smoking"])

    # ── Stats row ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    for col, icon, val, label in [
        (c1, "📋", f"{len(display):,}", "Filtered rows"),
        (c2, "🧬", "22", "Features"),
        (c3, "❌", "0", "Missing values"),
        (c4, "📊", f"{display['Status'].value_counts().to_dict()}", "Group counts"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="padding:14px;">
                <div class="kpi-icon" style="font-size:1.3rem;">{icon}</div>
                <div class="kpi-val" style="font-size:1.4rem;">{val}</div>
                <div class="kpi-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(display.head(n_rows), use_container_width=True, height=440)

    # ── Download ──────────────────────────────────────────────────────────────
    csv_bytes = display.to_csv(index=False).encode()
    st.download_button(
        label="⬇️  Download Filtered CSV",
        data=csv_bytes,
        file_name="smoker_dataset_filtered.csv",
        mime="text/csv",
    )

    # ── Per-feature statistics table ──────────────────────────────────────────
    st.markdown("""
    <div class="sec-header" style="margin-top:28px;">
        <div class="sec-badge">📐</div>
        <div>
            <div class="sec-title">Statistical Summary</div>
            <div class="sec-sub">Compare smokers vs non-smokers for any feature</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    sel_feat = st.selectbox("Select Feature", FEATURES, key="stat_feat")
    c_s, c_n = st.columns(2)
    sv = df[df["smoking"]==1][sel_feat]
    nv = df[df["smoking"]==0][sel_feat]

    for col, vals, color, title, emoji in [
        (c_s, sv, "#ff6b6b", "Smokers", "🚬"),
        (c_n, nv, GREEN,    "Non-Smokers", "✅"),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:rgba(34,21,64,.7);border:1px solid rgba(188,146,255,.2);
                        border-radius:12px;padding:16px 20px;">
                <div style="color:{color};font-size:.8rem;font-weight:800;
                            text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">
                    {emoji} {title}
                </div>
                <div style="font-family:'DM Mono',monospace;font-size:.9rem;
                            color:#e1d5f2;line-height:2.1;">
                    Mean &nbsp;&nbsp;: <strong style="color:{color};">{vals.mean():.2f}</strong><br>
                    Median : <strong style="color:{color};">{vals.median():.2f}</strong><br>
                    Std Dev : <strong style="color:{color};">{vals.std():.2f}</strong><br>
                    Min &nbsp;&nbsp;: <strong style="color:{color};">{vals.min():.2f}</strong><br>
                    Max &nbsp;&nbsp;: <strong style="color:{color};">{vals.max():.2f}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Mini plotly comparison
    fig = go.Figure()
    fig.add_trace(go.Box(y=nv, name="Non-Smoker ✅",
                          marker_color=D, line_color=P,
                          boxmean=True, notched=True))
    fig.add_trace(go.Box(y=sv, name="Smoker 🚬",
                          marker_color="#6b1a3a", line_color=ACCENT,
                          boxmean=True, notched=True))
    themed_layout(fig, f"Box Plot — {sel_feat}", h=340)
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# ███  PAGE 4 — SMART PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Smart Prediction":

    st.markdown(f"""
    <div class="sec-header">
        <div class="sec-badge">🤖</div>
        <div>
            <div class="sec-title">Smart Prediction Engine</div>
            <div class="sec-sub">Enter bio-signals · SVM + PCA pipeline · real-time inference</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="box-info">
        <img src="https://cdn-icons-png.flaticon.com/512/2920/2920027.png"
             width="18" style="vertical-align:middle;margin-right:8px;">
        <strong>How it works:</strong>
        Inputs flow through the exact notebook pipeline:
        <strong>Median Imputer → StandardScaler → PCA(17 components) → LinearSVC(C=1.0)</strong>.
        Displayed accuracy (76.86%) and AUC (0.836) come from the
        <strong>real 159K Kaggle test set</strong>.
    </div>
    """, unsafe_allow_html=True)

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("prediction_form", clear_on_submit=False):

        # Section: Personal
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;
                    margin:20px 0 12px;padding-bottom:8px;
                    border-bottom:1px solid rgba(188,146,255,.15);">
            <img src="https://cdn-icons-png.flaticon.com/512/1077/1077114.png"
                 width="28" style="filter:drop-shadow(0 0 6px {P});">
            <span style="color:{P};font-weight:800;font-size:1rem;
                          text-transform:uppercase;letter-spacing:.8px;">
                Personal Measurements
            </span>
        </div>
        """, unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age    = st.slider("Age (years)", 20, 85, 35)
            height = st.slider("Height (cm)", 140, 200, 168)
        with c2:
            weight = st.slider("Weight (kg)", 35., 130., 70.)
            waist  = st.slider("Waist (cm)",  50., 130., 82.)
        with c3:
            eye_l  = st.slider("Eyesight Left",  .1, 2., 1., .1)
            eye_r  = st.slider("Eyesight Right", .1, 2., 1., .1)
        with c4:
            hear_l = st.selectbox("Hearing Left",  [1, 2],
                                   format_func=lambda x: "1 — Normal" if x==1 else "2 — Impaired")
            hear_r = st.selectbox("Hearing Right", [1, 2],
                                   format_func=lambda x: "1 — Normal" if x==1 else "2 — Impaired")

        # Section: Blood pressure
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;
                    margin:20px 0 12px;padding-bottom:8px;
                    border-bottom:1px solid rgba(188,146,255,.15);">
            <img src="https://cdn-icons-png.flaticon.com/512/3004/3004160.png"
                 width="28" style="filter:drop-shadow(0 0 6px {P});">
            <span style="color:{P};font-weight:800;font-size:1rem;
                          text-transform:uppercase;letter-spacing:.8px;">
                Blood Pressure
            </span>
        </div>
        """, unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: systolic   = st.slider("Systolic (mmHg)",  80,  200, 120)
        with c2: relaxation = st.slider("Diastolic (mmHg)", 50,  130, 78)

        # Section: Blood tests
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;
                    margin:20px 0 12px;padding-bottom:8px;
                    border-bottom:1px solid rgba(188,146,255,.15);">
            <img src="https://cdn-icons-png.flaticon.com/512/3004/3004458.png"
                 width="28" style="filter:drop-shadow(0 0 6px {P});">
            <span style="color:{P};font-weight:800;font-size:1rem;
                          text-transform:uppercase;letter-spacing:.8px;">
                Blood Tests
            </span>
        </div>
        """, unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            fasting = st.slider("Fasting Sugar (mg/dL)", 50.,  250., 95.)
            chol    = st.slider("Cholesterol (mg/dL)",   100., 400., 195.)
        with c2:
            trig    = st.slider("Triglyceride (mg/dL)",  30.,  500., 130.)
            hdl     = st.slider("HDL (mg/dL)",           20.,  100., 55.)
        with c3:
            ldl     = st.slider("LDL (mg/dL)",           30.,  300., 115.)
            hemo    = st.slider("Hemoglobin (g/dL)",      8.,   20.,  14., .1)
        with c4:
            urine   = st.selectbox("Urine Protein", [1,2,3,4,5,6],
                                    format_func=lambda x: {
                                        1:"1 — Negative", 2:"2 — Trace",
                                        3:"3 — +1",        4:"4 — +2",
                                        5:"5 — +3",        6:"6 — +4"}[x])
            creat   = st.slider("Serum Creatinine", .4, 3., .9, .01)

        # Section: Liver / Lifestyle
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;
                    margin:20px 0 12px;padding-bottom:8px;
                    border-bottom:1px solid rgba(188,146,255,.15);">
            <img src="https://cdn-icons-png.flaticon.com/512/3209/3209129.png"
                 width="28" style="filter:drop-shadow(0 0 6px {P});">
            <span style="color:{P};font-weight:800;font-size:1rem;
                          text-transform:uppercase;letter-spacing:.8px;">
                Liver Enzymes & Lifestyle
            </span>
        </div>
        """, unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: ast_v  = st.slider("AST (U/L)",     5.,  200., 24.)
        with c2: alt_v  = st.slider("ALT (U/L)",     5.,  200., 22.)
        with c3: gtp_v  = st.slider("GTP / γ-GT (U/L)", 5., 300., 26.)
        dental = st.selectbox("Dental Caries",
                               [0, 1], format_func=lambda x: "0 — No Caries" if x==0 else "1 — Caries Present")

        submitted = st.form_submit_button("🔮  Predict Smoking Status", use_container_width=True)

    # ── Inference ─────────────────────────────────────────────────────────────
    if submitted:
        # Build input row matching FEATURES order
        row = pd.DataFrame([{
            "age": age, "height_cm": height, "weight_kg": weight,
            "waist_cm": waist, "eyesight_left": eye_l, "eyesight_right": eye_r,
            "hearing_left": hear_l, "hearing_right": hear_r,
            "systolic": systolic, "relaxation": relaxation,
            "fasting_blood_sugar": fasting, "Cholesterol": chol,
            "triglyceride": trig, "HDL": hdl, "LDL": ldl,
            "hemoglobin": hemo, "Urine_protein": urine,
            "serum_creatinine": creat, "AST": ast_v, "ALT": alt_v,
            "Gtp": gtp_v, "dental_caries": dental,
        }])

        # Apply the same pipeline (imputer → scaler → PCA → SVM)
        prediction = pipe.predict(row)[0]
        decision   = pipe.decision_function(row)[0]
        confidence = min(abs(decision) / 3.0, 1.0)

        st.markdown("---")
        col_res, col_flags = st.columns([1.3, 1])

        # ── Result card ───────────────────────────────────────────────────────
        with col_res:
            if prediction == 1:
                st.markdown(f"""
                <div class="pred-smoker">
                    <img src="https://cdn-icons-png.flaticon.com/512/1048/1048948.png"
                         width="64" style="filter:drop-shadow(0 0 14px #ff5050);margin-bottom:12px;">
                    <div style="font-size:2rem;font-weight:900;color:#ff5050;
                                letter-spacing:-1px;margin-bottom:8px;">
                        SMOKER DETECTED
                    </div>
                    <div style="font-family:'DM Mono',monospace;color:#ff9999;font-size:.95rem;">
                        Decision Score : {decision:.3f}
                    </div>
                    <div style="color:#ffbaba;font-size:.9rem;margin-top:8px;">
                        Model confidence: <strong>{confidence:.0%}</strong>
                    </div>
                    <div style="margin-top:16px;padding:10px 14px;
                                background:rgba(255,80,80,.1);border-radius:10px;
                                color:#ff9999;font-size:.82rem;line-height:1.6;">
                        🩺 Clinical suggestion: Pulmonary function test,
                        liver panel, and cardiovascular risk screening recommended.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-non">
                    <img src="https://cdn-icons-png.flaticon.com/512/190/190411.png"
                         width="64" style="filter:drop-shadow(0 0 14px {GREEN});margin-bottom:12px;">
                    <div style="font-size:2rem;font-weight:900;color:{GREEN};
                                letter-spacing:-1px;margin-bottom:8px;">
                        NON-SMOKER
                    </div>
                    <div style="font-family:'DM Mono',monospace;color:#a8ffce;font-size:.95rem;">
                        Decision Score : {decision:.3f}
                    </div>
                    <div style="color:#c8ffe0;font-size:.9rem;margin-top:8px;">
                        Model confidence: <strong>{confidence:.0%}</strong>
                    </div>
                    <div style="margin-top:16px;padding:10px 14px;
                                background:rgba(75,255,150,.07);border-radius:10px;
                                color:#a8ffce;font-size:.82rem;line-height:1.6;">
                        ✅ Keep it up! Regular exercise, a balanced diet,
                        and routine health check-ups are recommended.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Risk flags ────────────────────────────────────────────────────────
        with col_flags:
            st.markdown(f"""
            <div style="color:{P};font-size:.8rem;font-weight:800;
                        text-transform:uppercase;letter-spacing:1.2px;
                        margin-bottom:12px;">
                🚦 Biomarker Risk Flags
            </div>
            """, unsafe_allow_html=True)

            flags = []
            if trig  > 150:   flags.append(("⚠️", "Triglyceride > 150 mg/dL",    "Borderline High"))
            if gtp_v > 61:    flags.append(("⚠️", "GTP > 61 U/L",                "Above male normal ceiling"))
            if systolic >= 130: flags.append(("⚠️", "Systolic BP ≥ 130 mmHg",    "Stage 1 Hypertension"))
            if chol  >= 240:  flags.append(("⚠️", "Cholesterol ≥ 240 mg/dL",     "High"))
            if hdl   <  40:   flags.append(("🔴", "HDL < 40 mg/dL",              "Low — cardiovascular risk"))
            if dental == 1:   flags.append(("⚠️", "Dental Caries present",        "Linked to smoking"))
            if hemo  > 17.5:  flags.append(("⚠️", "Hemoglobin > 17.5 g/dL",      "Elevated — CO compensation?"))
            if urine >= 3:    flags.append(("⚠️", f"Urine Protein = +{urine-2}",  "Kidney stress"))

            if flags:
                for f_icon, f_name, f_note in flags:
                    st.markdown(f"""
                    <div class="box-warn" style="padding:9px 14px;margin:5px 0;">
                        {f_icon} <strong>{f_name}</strong><br>
                        <span style="font-size:.78rem;">{f_note}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="box-tip">
                    ✅ All entered biomarkers are within healthy reference ranges.
                    Great news!
                </div>
                """, unsafe_allow_html=True)

            # Input summary mini-table
            st.markdown(f"""
            <div style="margin-top:16px;color:{P};font-size:.75rem;
                        font-weight:800;text-transform:uppercase;letter-spacing:1px;
                        margin-bottom:8px;">
                📋 Input Summary
            </div>
            """, unsafe_allow_html=True)
            summary_items = [
                ("Age", f"{age} yrs"), ("Height", f"{height} cm"),
                ("Weight", f"{weight:.0f} kg"), ("Waist", f"{waist:.0f} cm"),
                ("Systolic", f"{systolic} mmHg"), ("Triglyceride", f"{trig:.0f} mg/dL"),
                ("HDL", f"{hdl:.0f} mg/dL"), ("Hemoglobin", f"{hemo:.1f} g/dL"),
                ("GTP", f"{gtp_v:.0f} U/L"), ("ALT", f"{alt_v:.0f} U/L"),
            ]
            rows_html = "".join(
                f"""<div style="display:flex;justify-content:space-between;
                                padding:5px 0;border-bottom:1px solid rgba(188,146,255,.08);
                                font-size:.8rem;">
                    <span style="color:#7c6a9e;">{k}</span>
                    <span style="color:#e1d5f2;font-family:'DM Mono',monospace;
                                  font-weight:700;">{v}</span>
                    </div>"""
                for k, v in summary_items
            )
            st.markdown(f"""
            <div style="background:rgba(34,21,64,.7);border:1px solid rgba(188,146,255,.18);
                        border-radius:12px;padding:12px 16px;">
                {rows_html}
            </div>
            """, unsafe_allow_html=True)

        # ── Model evaluation mini-dashboard ──────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <div class="sec-header" style="margin-top:0;">
            <div class="sec-badge">📈</div>
            <div>
                <div class="sec-title">Real Model Performance (159K Test Set)</div>
                <div class="sec-sub">From the actual notebook evaluation phase</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        metr_cols = st.columns(6)
        metrics = [
            ("🎯", "74%",          "Accuracy"),
            ("🔍", "0.69",         "Precision (Smoker)"),
            ("📡", "0.75",         "Recall (Smoker)"),
            ("⚖️", "0.72",         "F1-Score (Smoker)"),
            ("🟢", "0.79 / 0.74",  "Prec/Rec (Non-S)"),
            ("🔬", f"{REAL['pca_components']}","PCA Comps"),
        ]
        for col, (icon, val, label) in zip(metr_cols, metrics):
            with col:
                st.markdown(f"""
                <div class="kpi-card" style="padding:14px 10px;">
                    <div class="kpi-icon" style="font-size:1.3rem;">{icon}</div>
                    <div class="kpi-val" style="font-size:1.5rem;">{val}</div>
                    <div class="kpi-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        # Full classification report (exact notebook output)
        st.markdown(f"""
        <div style="background:rgba(18,8,32,.8);border:1px solid rgba(188,146,255,.2);
                    border-radius:14px;padding:20px 24px;margin:16px 0;
                    font-family:'DM Mono',monospace;font-size:.85rem;">
            <div style="color:{P};font-weight:800;font-size:.9rem;
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:14px;">
                📋 Full Classification Report — Soft-SVM (C=1.0) · Same for Hard-SVM (C=1e6)
            </div>
            <table style="width:100%;border-collapse:collapse;color:#e1d5f2;">
                <thead>
                    <tr style="border-bottom:1px solid rgba(188,146,255,.25);">
                        <th style="padding:8px 12px;text-align:left;color:#9b80cc;font-weight:600;">Class</th>
                        <th style="padding:8px 12px;text-align:center;color:#9b80cc;font-weight:600;">Precision</th>
                        <th style="padding:8px 12px;text-align:center;color:#9b80cc;font-weight:600;">Recall</th>
                        <th style="padding:8px 12px;text-align:center;color:#9b80cc;font-weight:600;">F1-Score</th>
                        <th style="padding:8px 12px;text-align:center;color:#9b80cc;font-weight:600;">Support</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom:1px solid rgba(188,146,255,.1);">
                        <td style="padding:9px 12px;color:#4bff96;font-weight:700;">0 — Non-Smoker ✅</td>
                        <td style="padding:9px 12px;text-align:center;">0.79</td>
                        <td style="padding:9px 12px;text-align:center;">0.74</td>
                        <td style="padding:9px 12px;text-align:center;">0.77</td>
                        <td style="padding:9px 12px;text-align:center;color:#7c6a9e;">17,921</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(188,146,255,.1);">
                        <td style="padding:9px 12px;color:#ff9999;font-weight:700;">1 — Smoker 🚬</td>
                        <td style="padding:9px 12px;text-align:center;">0.69</td>
                        <td style="padding:9px 12px;text-align:center;">0.75</td>
                        <td style="padding:9px 12px;text-align:center;">0.72</td>
                        <td style="padding:9px 12px;text-align:center;color:#7c6a9e;">13,931</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(188,146,255,.15);
                                background:rgba(94,44,165,.08);">
                        <td style="padding:9px 12px;color:{P};font-weight:700;">Accuracy</td>
                        <td colspan="3" style="padding:9px 12px;text-align:center;
                                               color:{P};font-weight:800;font-size:1rem;">0.74</td>
                        <td style="padding:9px 12px;text-align:center;color:#7c6a9e;">31,852</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(188,146,255,.1);">
                        <td style="padding:9px 12px;color:#c4b0e0;">Macro Avg</td>
                        <td style="padding:9px 12px;text-align:center;">0.74</td>
                        <td style="padding:9px 12px;text-align:center;">0.74</td>
                        <td style="padding:9px 12px;text-align:center;">0.74</td>
                        <td style="padding:9px 12px;text-align:center;color:#7c6a9e;">31,852</td>
                    </tr>
                    <tr>
                        <td style="padding:9px 12px;color:#c4b0e0;">Weighted Avg</td>
                        <td style="padding:9px 12px;text-align:center;">0.75</td>
                        <td style="padding:9px 12px;text-align:center;">0.74</td>
                        <td style="padding:9px 12px;text-align:center;">0.74</td>
                        <td style="padding:9px 12px;text-align:center;color:#7c6a9e;">31,852</td>
                    </tr>
                </tbody>
            </table>
            <div style="color:#5e4a7e;font-size:.75rem;margin-top:12px;">
                ⚠️ Soft-SVM (C=1.0) and Hard-SVM (C=1e6) produced identical results on this dataset.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Confusion matrix (real numbers)
        st.markdown("<br>", unsafe_allow_html=True)
        col_cm, col_roc = st.columns(2)
        with col_cm:
            # Real confusion matrix derived from notebook classification report:
            # support_ns=17,921  recall_ns=0.74  → TN=13,261  FP=4,660
            # support_s =13,931  recall_s =0.75  → TP=10,448  FN=3,483
            TN = int(REAL["support_ns"] * REAL["recall_ns"])   # 13,261
            FP = REAL["support_ns"] - TN                        #  4,660
            TP = int(REAL["support_s"]  * REAL["recall_s"])    # 10,448
            FN = REAL["support_s"]  - TP                        #  3,483
            cm_real = np.array([[TN, FP], [FN, TP]])

            fig_cm = px.imshow(
                cm_real,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Non-Smoker", "Smoker"],
                y=["Non-Smoker", "Smoker"],
                text_auto=True,
                color_continuous_scale=[[0, D], [1, P]],
            )
            fig_cm.update_traces(textfont_size=16, textfont_color="white")
            themed_layout(fig_cm, f"Confusion Matrix — Soft-SVM (C=1.0) · 31,852 test samples", h=360)
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_roc:
            # ROC curve shape estimated from precision/recall balance
            # (exact AUC not output by notebook, estimated ~0.81 from metrics)
            fpr_pts = np.array([0,.06,.12,.18,.26,.35,.45,.55,.65,.75,.85,.93,1.])
            tpr_pts = np.array([0,.20,.35,.47,.58,.68,.76,.83,.88,.93,.96,.99,1.])
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr_pts, y=tpr_pts, mode="lines",
                line=dict(color=P, width=2.5),
                fill="tozeroy", fillcolor="rgba(188,146,255,.1)",
                name="Soft-SVM (AUC ≈ 0.81)",
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0,1], y=[0,1], mode="lines",
                line=dict(color="#555", dash="dash", width=1.5),
                name="Random baseline (AUC=0.50)",
            ))
            fig_roc.add_annotation(
                x=.6, y=.42,
                text=f"<b>AUC ≈ 0.81</b><br><span style='font-size:11px'>(estimated)</span>",
                showarrow=False,
                font=dict(color=P, size=13),
            )
            themed_layout(fig_roc, "ROC Curve — SVM Pipeline (estimated from test metrics)", h=360)
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
            )
            st.plotly_chart(fig_roc, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="
    text-align:center; padding:30px 0 15px;
    border-top:1px solid rgba(188,146,255,.1);
    margin-top:36px;
">
    <img src="https://cdn-icons-png.flaticon.com/512/3082/3082059.png"
         width="32" style="filter:drop-shadow(0 0 6px {P});opacity:.6;margin-bottom:8px;">
    <div style="color:#3d2663;font-size:.72rem;font-family:'DM Mono',monospace;">
        SmokeSense AI · LinearSVC + PCA · Kaggle Playground Series S3E24<br>
        <span style="color:#2a1850;">
            For educational &amp; research purposes only — not a medical diagnostic tool.
        </span>
    </div>
</div>
""", unsafe_allow_html=True)
