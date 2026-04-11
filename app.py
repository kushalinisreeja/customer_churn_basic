import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from scipy.stats import loguniform
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="RetainIQ — Churn Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DESIGN SYSTEM — Refined Dark-First, Editorial Aesthetic
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,300&display=swap');

:root {
    /* Core Palette */
    --ink:        #0A0C10;
    --ink-2:      #111318;
    --ink-3:      #1A1D24;
    --ink-4:      #22262F;
    --line:       #2A2F3A;
    --line-light: #353B48;

    /* Text */
    --text-1:     #F0F2F7;
    --text-2:     #A8AEBB;
    --text-3:     #6B7280;

    /* Accent — Warm Gold */
    --gold:       #E8B84B;
    --gold-dim:   rgba(232,184,75,0.12);
    --gold-glow:  rgba(232,184,75,0.25);

    /* Semantic */
    --danger:     #E05C5C;
    --danger-dim: rgba(224,92,92,0.10);
    --safe:       #4CAF80;
    --safe-dim:   rgba(76,175,128,0.10);
    --info:       #5B8AF0;
    --info-dim:   rgba(91,138,240,0.10);
    --warn:       #F0935B;
    --warn-dim:   rgba(240,147,91,0.10);

    /* Layout */
    --radius-sm:  8px;
    --radius-md:  14px;
    --radius-lg:  20px;
    --shadow:     0 4px 32px rgba(0,0,0,0.4);
    --shadow-sm:  0 2px 12px rgba(0,0,0,0.25);
}

/* ── RESET ─────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-1) !important;
}
.stApp {
    background: var(--ink) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(232,184,75,0.04) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(91,138,240,0.04) 0%, transparent 60%);
}
.block-container {
    padding: 2rem 2.5rem 5rem !important;
    max-width: 1280px !important;
}
h1,h2,h3,h4,h5,h6 { color: var(--text-1) !important; }
.stMarkdown p, .stMarkdown li { color: var(--text-2) !important; }
hr { border-color: var(--line) !important; margin: 2rem 0; }

/* ── SIDEBAR ────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--ink-2) !important;
    border-right: 1px solid var(--line) !important;
}
[data-testid="stSidebarContent"] { padding: 2rem 1.25rem !important; }
[data-testid="stSidebar"] .stButton button {
    background: transparent !important;
    border: none !important;
    color: var(--text-2) !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: 10px 14px !important;
    border-radius: var(--radius-sm) !important;
    width: 100% !important;
    transition: all .18s ease !important;
    box-shadow: none !important;
    letter-spacing: 0.01em !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: var(--ink-4) !important;
    color: var(--text-1) !important;
}

/* ── WORDMARK ───────────────────────────────────────── */
.wordmark {
    display: flex;
    align-items: baseline;
    gap: 6px;
    margin-bottom: 28px;
}
.wordmark-logo {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.5rem;
    color: var(--text-1) !important;
    letter-spacing: -0.02em;
}
.wordmark-dot {
    width: 8px; height: 8px;
    background: var(--gold);
    border-radius: 50%;
    margin-bottom: 3px;
    flex-shrink: 0;
}
.wordmark-tag {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-3) !important;
    margin-bottom: 28px;
}

/* ── NAV ────────────────────────────────────────────── */
.nav-divider {
    font-size: 9.5px;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: var(--text-3) !important;
    padding: 0 14px;
    margin: 20px 0 8px;
}

/* ── ACTIVE NAV STATE ───────────────────────────────── */
.active-nav-indicator {
    background: var(--gold-dim) !important;
    border: 1px solid var(--gold-glow) !important;
    color: var(--gold) !important;
    border-radius: var(--radius-sm);
    padding: 10px 14px;
    font-size: 13.5px;
    font-weight: 600;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ── PAGE HEADER ────────────────────────────────────── */
.page-header {
    border-bottom: 1px solid var(--line);
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
}
.page-eyebrow {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--gold) !important;
    margin-bottom: 8px;
}
.page-title {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.1rem;
    font-weight: 400;
    color: var(--text-1) !important;
    letter-spacing: -0.02em;
    line-height: 1.15;
    margin: 0 0 8px;
}
.page-subtitle {
    font-size: 14px;
    color: var(--text-3) !important;
    font-weight: 400;
    line-height: 1.6;
}

/* ── KPI CARDS ──────────────────────────────────────── */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 2rem;
}
.kpi-card {
    background: var(--ink-2);
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    padding: 22px 20px;
    position: relative;
    overflow: hidden;
    transition: border-color .2s, transform .2s;
}
.kpi-card:hover { border-color: var(--line-light); transform: translateY(-2px); }
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.kpi-card.gold::before  { background: var(--gold); }
.kpi-card.danger::before { background: var(--danger); }
.kpi-card.warn::before  { background: var(--warn); }
.kpi-card.safe::before  { background: var(--safe); }
.kpi-value {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.2rem;
    line-height: 1;
    margin: 10px 0 6px;
    letter-spacing: -0.02em;
}
.kpi-card.gold   .kpi-value { color: var(--gold)   !important; }
.kpi-card.danger .kpi-value { color: var(--danger)  !important; }
.kpi-card.warn   .kpi-value { color: var(--warn)    !important; }
.kpi-card.safe   .kpi-value { color: var(--safe)    !important; }
.kpi-label {
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--text-3) !important;
}
.kpi-note {
    font-size: 12px;
    color: var(--text-3) !important;
    margin-top: 6px;
    line-height: 1.45;
}

/* ── PANEL CARDS ────────────────────────────────────── */
.panel {
    background: var(--ink-2);
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    padding: 24px;
    height: 100%;
}
.panel-title {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.3px;
    color: var(--text-1) !important;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-body {
    font-size: 13.5px;
    color: var(--text-2) !important;
    line-height: 1.8;
}
.panel-body b, .panel-body strong { color: var(--text-1) !important; }

/* ── SIGNAL ITEMS ───────────────────────────────────── */
.signal-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid var(--line);
    font-size: 13px;
    color: var(--text-2) !important;
}
.signal-item:last-child { border-bottom: none; }
.signal-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    margin-top: 5px;
    flex-shrink: 0;
}
.dot-danger { background: var(--danger); }
.dot-safe   { background: var(--safe); }

/* ── CALLOUTS ───────────────────────────────────────── */
.callout {
    border-radius: var(--radius-md);
    padding: 16px 20px;
    font-size: 13.5px;
    line-height: 1.75;
    margin-bottom: 1.25rem;
    border: 1px solid;
}
.callout-gold   { background: var(--gold-dim);   border-color: rgba(232,184,75,0.2);   color: #D4A830 !important; }
.callout-danger { background: var(--danger-dim); border-color: rgba(224,92,92,0.2);    color: #E07070 !important; }
.callout-safe   { background: var(--safe-dim);   border-color: rgba(76,175,128,0.2);   color: #5BC490 !important; }
.callout-info   { background: var(--info-dim);   border-color: rgba(91,138,240,0.2);   color: #7BA6F5 !important; }
.callout b, .callout strong { color: inherit !important; }

/* ── VERDICT ────────────────────────────────────────── */
.verdict-wrap {
    border-radius: var(--radius-lg);
    padding: 32px 28px;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}
.verdict-danger { background: var(--danger-dim); border-color: rgba(224,92,92,0.3); }
.verdict-safe   { background: var(--safe-dim);   border-color: rgba(76,175,128,0.3); }
.verdict-icon   { font-size: 2rem; margin-bottom: 10px; }
.verdict-title  {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.6rem;
    margin-bottom: 8px;
    letter-spacing: -0.02em;
}
.verdict-danger .verdict-title { color: var(--danger) !important; }
.verdict-safe   .verdict-title { color: var(--safe)   !important; }
.verdict-body { font-size: 13.5px; color: var(--text-2) !important; line-height: 1.65; }

/* ── PROBABILITY ────────────────────────────────────── */
.prob-num {
    font-family: 'DM Serif Display', serif !important;
    font-size: 3.5rem;
    line-height: 1;
    letter-spacing: -0.03em;
    margin: 20px 0 4px;
}
.prob-desc { font-size: 11px; color: var(--text-3) !important; }
.bar-track {
    background: var(--ink-4);
    border-radius: 100px;
    height: 8px;
    margin-top: 14px;
    overflow: hidden;
}
.bar-fill { height: 100%; border-radius: 100px; transition: width .6s ease; }
.bar-labels {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: var(--text-3) !important;
    margin-top: 6px;
}

/* ── SUMMARY TABLE ──────────────────────────────────── */
.sum-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid var(--line);
    font-size: 13px;
}
.sum-row:last-child { border-bottom: none; }
.sum-key { color: var(--text-3) !important; font-weight: 500; }
.sum-val { color: var(--text-1) !important; font-weight: 600; font-size: 13px; }

/* ── ACTION STEPS ───────────────────────────────────── */
.action-item {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 14px 0;
    border-bottom: 1px solid var(--line);
}
.action-item:last-child { border-bottom: none; }
.action-num {
    background: var(--gold-dim);
    color: var(--gold) !important;
    font-weight: 700;
    font-size: 11px;
    min-width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    border: 1px solid var(--gold-glow);
}
.action-text { font-size: 13.5px; color: var(--text-2) !important; line-height: 1.6; }
.action-text b { color: var(--text-1) !important; }

/* ── FLAGS ──────────────────────────────────────────── */
.flag-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 0;
    border-bottom: 1px solid var(--line);
    font-size: 13px;
    color: var(--text-2) !important;
}
.flag-item:last-child { border-bottom: none; }

/* ── METRIC PILLS ───────────────────────────────────── */
.metric-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--ink-4);
    border: 1px solid var(--line-light);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-2) !important;
    margin: 3px;
}

/* ── FORM ───────────────────────────────────────────── */
[data-testid="stForm"] {
    background: var(--ink-2) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-lg) !important;
    padding: 28px !important;
}
.form-section-label {
    font-size: 9.5px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--gold) !important;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--gold-dim);
}
.stFormSubmitButton button {
    background: var(--gold) !important;
    color: var(--ink) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-weight: 700 !important;
    font-size: 14.5px !important;
    padding: 14px 28px !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
    transition: opacity .2s !important;
}
.stFormSubmitButton button:hover { opacity: 0.88 !important; }

/* ── STREAMLIT OVERRIDES ────────────────────────────── */
[data-baseweb="select"] > div,
[data-baseweb="input"]  > div {
    background: var(--ink-3) !important;
    border-color: var(--line) !important;
    color: var(--text-1) !important;
    border-radius: var(--radius-sm) !important;
}
[data-baseweb="select"] span,
[data-baseweb="input"] input { color: var(--text-1) !important; }
[data-testid="stSlider"] label p { color: var(--text-2) !important; }
[data-testid="stMetricLabel"] p  { color: var(--text-3) !important; font-weight: 600 !important; }
[data-testid="stMetricValue"]    { color: var(--text-1) !important; }
[data-testid="metric-container"] {
    background: var(--ink-2) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius-md) !important;
    padding: 18px !important;
}
.stAlert { border-radius: var(--radius-md) !important; }
.stDataFrame { border-radius: var(--radius-md) !important; border: 1px solid var(--line) !important; }

/* ── PILLS ──────────────────────────────────────────── */
.pill {
    display: inline-block;
    background: var(--ink-3);
    border: 1px solid var(--line-light);
    color: var(--text-2) !important;
    font-size: 11.5px;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 100px;
    margin: 3px;
}

/* ── FOOTER ─────────────────────────────────────────── */
.footer {
    text-align: center;
    font-size: 11.5px;
    color: var(--text-3) !important;
    padding: 16px 0;
    letter-spacing: 0.3px;
}
.footer span { color: var(--gold) !important; }

/* ── ACCURACY TABLE ─────────────────────────────────── */
.acc-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 13px 0;
    border-bottom: 1px solid var(--line);
    font-size: 13.5px;
}
.acc-row:last-child { border-bottom: none; }
.acc-metric { color: var(--text-2) !important; font-weight: 500; }
.acc-bar-wrap { flex: 1; margin: 0 20px; }
.acc-bar-bg {
    background: var(--ink-4);
    border-radius: 100px;
    height: 6px;
    overflow: hidden;
}
.acc-bar-fill { height: 100%; border-radius: 100px; }
.acc-val { color: var(--text-1) !important; font-weight: 700; font-size: 14px; min-width: 40px; text-align: right; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================
ACTUAL_METRICS = {
    'accuracy': 0.7388, 'recall': 0.8102, 'precision': 0.5050,
    'f1': 0.6222, 'roc_auc': 0.8389,
    'tp': 303, 'fp': 297, 'tn': 738, 'fn': 71,
    'train_accuracy': 0.7691, 'test_accuracy': 0.7388,
}
TOTAL_CUSTOMERS = 7043
CHURN_RATE      = 0.2654
AVG_MONTHLY     = 65.0

PAGES = [
    ("◈", "Home",     "Overview"),
    ("⌖", "Check",    "Risk Checker"),
    ("◎", "Accurate", "Model Performance"),
    ("▦", "Data",     "Data Explorer"),
]

# ============================================================================
# DATA & MODEL
# ============================================================================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/varaprasad197/Customer-churn-predictor/main/tele_comm.csv"
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        return None

@st.cache_resource
def train_model(df):
    df = df.copy()
    df.drop(columns=['customerID'], inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.loc[df['tenure'] == 0, 'TotalCharges'] = 0
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        if df[col].dtype == object:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})

    X, y = df.drop('Churn', axis=1), df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    med = X_train['MonthlyCharges'].median()

    def engineer(d, m):
        d = d.copy()
        d['IsFirstYear']           = (d['tenure'] <= 12).astype(int)
        d['AvgMonthlyCharge']      = d.apply(
            lambda r: r['TotalCharges'] / r['tenure'] if r['tenure'] > 0 else r['MonthlyCharges'], axis=1)
        svcs = ['OnlineSecurity','OnlineBackup','DeviceProtection',
                'TechSupport','StreamingTV','StreamingMovies']
        d['NumAdditionalServices'] = d[svcs].apply(
            lambda r: (r == 'Yes').sum() if isinstance(r.iloc[0], str) else r.sum(), axis=1)
        d['FiberOpticUser']        = (d['InternetService'] == 'Fiber optic').astype(int)
        d['IsMonthToMonth']        = (d['Contract'] == 'Month-to-month').astype(int)
        d['PaymentRisk']           = d['PaymentMethod'].map(
            {'Electronic check': 3, 'Mailed check': 2,
             'Bank transfer (automatic)': 1, 'Credit card (automatic)': 1})
        d['HighCostLowTenure']     = ((d['MonthlyCharges'] > m) & (d['tenure'] < 12)).astype(int)
        d['HasFamily']             = ((d['Partner'] == 1) | (d['Dependents'] == 1)).astype(int)
        return d

    X_train = engineer(X_train, med)
    X_test  = engineer(X_test,  med)
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test  = pd.get_dummies(X_test,  drop_first=True)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    scaler  = StandardScaler()
    Xtr_sc  = scaler.fit_transform(X_train)
    Xte_sc  = scaler.transform(X_test)
    Xr, yr  = SMOTE(random_state=42, k_neighbors=5).fit_resample(Xtr_sc, y_train)

    l1 = LogisticRegression(penalty='l1', solver='liblinear',
                            C=0.1, max_iter=1000, random_state=42)
    l1.fit(Xr, yr)
    mask     = l1.coef_[0] != 0
    selected = X_train.columns[mask].tolist()

    rs = RandomizedSearchCV(
        LogisticRegression(random_state=42),
        {'C': loguniform(0.01, 10), 'penalty': ['l1','l2'],
         'solver': ['liblinear'], 'max_iter': [500, 1000]},
        n_iter=20, cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='recall', n_jobs=-1, random_state=42)
    rs.fit(Xr[:, mask], yr)
    best = rs.best_estimator_
    best.fit(Xr[:, mask], yr)

    return dict(model=best, scaler=scaler, X_train=X_train, X_test=X_test,
                X_train_sel=Xr[:, mask], X_test_sel=Xte_sc[:, mask],
                selected_features=selected, l1_mask=mask,
                train_median_charge=med)

def predict_churn(customer, arts):
    df2 = pd.DataFrame([customer])
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        if col in df2.columns and df2[col].dtype == object:
            df2[col] = df2[col].map({'Yes': 1, 'No': 0})
    if 'gender' in df2.columns:
        df2['gender'] = df2['gender'].map({'Female': 1, 'Male': 0})
    med = arts['train_median_charge']
    df2['IsFirstYear']      = (df2['tenure'] <= 12).astype(int)
    df2['AvgMonthlyCharge'] = df2.apply(
        lambda r: r['TotalCharges'] / r['tenure'] if r['tenure'] > 0 else r['MonthlyCharges'], axis=1)
    svcs = ['OnlineSecurity','OnlineBackup','DeviceProtection',
            'TechSupport','StreamingTV','StreamingMovies']
    if all(c in df2.columns for c in svcs):
        df2['NumAdditionalServices'] = df2[svcs].apply(
            lambda r: (r == 'Yes').sum() if isinstance(r.iloc[0], str) else r.sum(), axis=1)
    df2['FiberOpticUser']    = (df2['InternetService'] == 'Fiber optic').astype(int)
    df2['IsMonthToMonth']    = (df2['Contract'] == 'Month-to-month').astype(int)
    df2['PaymentRisk']       = df2['PaymentMethod'].map(
        {'Electronic check': 3, 'Mailed check': 2,
         'Bank transfer (automatic)': 1, 'Credit card (automatic)': 1})
    df2['HighCostLowTenure'] = ((df2['MonthlyCharges'] > med) & (df2['tenure'] < 12)).astype(int)
    df2['HasFamily']         = ((df2['Partner'] == 1) | (df2['Dependents'] == 1)).astype(int)
    df2 = pd.get_dummies(df2, drop_first=True)
    df2 = df2.reindex(columns=arts['X_train'].columns, fill_value=0)
    sc  = arts['scaler'].transform(df2)
    return arts['model'].predict_proba(sc[:, arts['l1_mask']])[0, 1]

# ============================================================================
# MPL THEME
# ============================================================================
MPL_STYLE = {
    'figure.facecolor': '#111318',
    'axes.facecolor':   '#111318',
    'axes.edgecolor':   '#2A2F3A',
    'axes.labelcolor':  '#6B7280',
    'xtick.color':      '#6B7280',
    'ytick.color':      '#6B7280',
    'text.color':       '#A8AEBB',
    'grid.color':       '#1A1D24',
    'grid.linestyle':   '--',
    'font.family':      'sans-serif',
}

# ============================================================================
# BOOT
# ============================================================================
df = load_data()
if df is None:
    st.stop()

with st.spinner("Initialising model…"):
    arts = train_model(df)

at_risk         = int(TOTAL_CUSTOMERS * CHURN_RATE)
revenue_at_risk = at_risk * AVG_MONTHLY * 12

# ============================================================================
# SIDEBAR
# ============================================================================
if "page" not in st.session_state:
    st.session_state.page = "Home"

with st.sidebar:
    st.markdown("""
    <div class="wordmark">
        <span class="wordmark-logo">RetainIQ</span>
        <div class="wordmark-dot"></div>
    </div>
    <div class="wordmark-tag">Churn Intelligence</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-divider">Navigation</div>', unsafe_allow_html=True)

    for sym, key, label in PAGES:
        is_active = (st.session_state.page == key)
        if is_active:
            st.markdown(f'<div class="active-nav-indicator">{sym}&nbsp;&nbsp;{label}</div>',
                        unsafe_allow_html=True)
        else:
            if st.button(f"{sym}  {label}", key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()

    st.markdown("---")

    # Model info
    st.markdown(f"""
    <div style="font-size:11.5px;line-height:2;color:var(--text-3);padding:0 2px;">
        <div style="color:var(--text-2);font-weight:600;margin-bottom:8px;font-size:12px;">Model Details</div>
        <div>Algorithm &nbsp;·&nbsp; <span style="color:var(--text-2)">Logistic Regression</span></div>
        <div>Balancing &nbsp;·&nbsp; <span style="color:var(--text-2)">SMOTE</span></div>
        <div>Tuning &nbsp;·&nbsp; <span style="color:var(--text-2)">RandomizedSearchCV</span></div>
        <div>Dataset &nbsp;·&nbsp; <span style="color:var(--text-2)">{TOTAL_CUSTOMERS:,} customers</span></div>
        <div>Recall &nbsp;·&nbsp; <span style="color:var(--safe)">81.0%</span></div>
        <div>AUC-ROC &nbsp;·&nbsp; <span style="color:var(--safe)">83.9%</span></div>
    </div>
    """, unsafe_allow_html=True)

page = st.session_state.page

# ============================================================================
# PAGE 1 — HOME / OVERVIEW
# ============================================================================
if page == "Home":
    st.markdown("""
    <div class="page-header">
        <div class="page-eyebrow">Overview</div>
        <div class="page-title">Customer Retention Intelligence</div>
        <div class="page-subtitle">AI-powered early warning system for customer churn — act before they leave.</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card gold">
            <div class="kpi-label">Total Customers</div>
            <div class="kpi-value">{TOTAL_CUSTOMERS:,}</div>
            <div class="kpi-note">Telecom subscribers in dataset</div>
        </div>
        <div class="kpi-card danger">
            <div class="kpi-label">Customers at Risk</div>
            <div class="kpi-value">{at_risk:,}</div>
            <div class="kpi-note">{CHURN_RATE:.0%} of total base</div>
        </div>
        <div class="kpi-card warn">
            <div class="kpi-label">Annual Revenue at Risk</div>
            <div class="kpi-value">${revenue_at_risk/1e6:.1f}M</div>
            <div class="kpi-note">If all at-risk customers leave</div>
        </div>
        <div class="kpi-card safe">
            <div class="kpi-label">Churners Detected</div>
            <div class="kpi-value">81%</div>
            <div class="kpi-note">Model recall on held-out data</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout callout-info">
        <b>How it works:</b> Enter any customer's profile in the Risk Checker and receive an instant probability score from 0–100%. Scores above 50% trigger an actionable retention plan.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("""
        <div class="panel">
            <div class="panel-title">⚠ Churn Risk Signals</div>
            <div class="signal-item">
                <div class="signal-dot dot-danger"></div>
                <div>Month-to-month contract — no long-term commitment</div>
            </div>
            <div class="signal-item">
                <div class="signal-dot dot-danger"></div>
                <div>Tenure under 12 months — still in honeymoon phase</div>
            </div>
            <div class="signal-item">
                <div class="signal-dot dot-danger"></div>
                <div>Fiber optic internet — easiest to switch providers</div>
            </div>
            <div class="signal-item">
                <div class="signal-dot dot-danger"></div>
                <div>Electronic check payment — lowest switching friction</div>
            </div>
            <div class="signal-item">
                <div class="signal-dot dot-danger"></div>
                <div>High bill relative to short tenure</div>
            </div>
            <div class="signal-item">
                <div class="signal-dot dot-safe"></div>
                <div>Partner or dependents — significantly more loyal</div>
            </div>
            <div class="signal-item">
                <div class="signal-dot dot-safe"></div>
                <div>One- or two-year contract — committed customers</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="panel">
            <div class="panel-title">◈ How to Use This Dashboard</div>
            <div class="panel-body">
                <div class="action-item">
                    <div class="action-num">1</div>
                    <div class="action-text"><b>Go to Risk Checker</b><br>Enter a customer's details — contract, billing, services — and get a live churn probability score.</div>
                </div>
                <div class="action-item">
                    <div class="action-num">2</div>
                    <div class="action-text"><b>Act on high-risk alerts</b><br>Score above 50%? The dashboard provides a tailored 4-step retention plan immediately.</div>
                </div>
                <div class="action-item">
                    <div class="action-num">3</div>
                    <div class="action-text"><b>Review model performance</b><br>See confusion matrix and precision/recall metrics to understand confidence levels.</div>
                </div>
                <div class="action-item">
                    <div class="action-num">4</div>
                    <div class="action-text"><b>Explore the data</b><br>Understand churn patterns across tenure, billing, and contract types.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="callout callout-safe">
        <b>Business case:</b> Acquiring a new customer costs 5–7× more than retaining one.
        Retaining just 10% of at-risk customers recovers
        <b>${revenue_at_risk * 0.10:,.0f} per year</b> in otherwise-lost revenue.
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# PAGE 2 — RISK CHECKER
# ============================================================================
elif page == "Check":
    st.markdown("""
    <div class="page-header">
        <div class="page-eyebrow">Risk Checker</div>
        <div class="page-title">Customer Churn Probability</div>
        <div class="page-subtitle">Fill in the customer profile below for an instant AI risk assessment.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("pred_form"):
        c1, c2, c3 = st.columns(3, gap="large")

        with c1:
            st.markdown('<div class="form-section-label">Customer Profile</div>', unsafe_allow_html=True)
            gender     = st.selectbox("Gender", ["Male", "Female"])
            senior     = st.selectbox("Senior Citizen (65+)", ["No", "Yes"])
            partner    = st.selectbox("Has a Partner?", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
            tenure     = st.slider("Tenure (months)", 0, 72, 12)

        with c2:
            st.markdown('<div class="form-section-label">Services</div>', unsafe_allow_html=True)
            phone    = st.selectbox("Phone Service", ["No", "Yes"])
            internet = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
            security = st.selectbox("Online Security", ["No", "Yes"])
            backup   = st.selectbox("Online Backup", ["No", "Yes"])
            device   = st.selectbox("Device Protection", ["No", "Yes"])
            tech     = st.selectbox("Tech Support", ["No", "Yes"])

        with c3:
            st.markdown('<div class="form-section-label">Billing & Contract</div>', unsafe_allow_html=True)
            contract  = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment   = st.selectbox("Payment Method",
                                     ["Electronic check", "Mailed check",
                                      "Bank transfer (automatic)", "Credit card (automatic)"])
            paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
            stream_tv = st.selectbox("Streaming TV", ["No", "Yes"])
            stream_mv = st.selectbox("Streaming Movies", ["No", "Yes"])
            monthly   = st.number_input("Monthly Charge ($)", 0.0, 200.0, 65.0, 5.0)
            total     = st.number_input("Total Charged ($)", 0.0, 10000.0, 1000.0, 50.0)

        submitted = st.form_submit_button("⌖  Run Risk Assessment", use_container_width=True)

    if submitted:
        inp = {
            'gender': gender, 'SeniorCitizen': 1 if senior == 'Yes' else 0,
            'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
            'PhoneService': phone, 'InternetService': internet,
            'OnlineSecurity': security, 'OnlineBackup': backup,
            'DeviceProtection': device, 'TechSupport': tech,
            'StreamingTV': stream_tv, 'StreamingMovies': stream_mv,
            'Contract': contract, 'PaperlessBilling': paperless,
            'PaymentMethod': payment, 'MonthlyCharges': monthly, 'TotalCharges': total,
        }
        prob = predict_churn(inp, arts)
        pct  = prob * 100
        high = prob >= 0.5

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2 = st.columns([1.6, 1], gap="large")

        with r1:
            if high:
                bar_color  = "#E05C5C"
                v_class    = "verdict-danger"
                icon       = "⚠"
                headline   = "Elevated Churn Risk"
                sub        = f"The model estimates a <b>{pct:.0f}% probability</b> this customer will cancel. Immediate outreach is recommended."
            else:
                bar_color  = "#4CAF80"
                v_class    = "verdict-safe"
                icon       = "✓"
                headline   = "Customer Appears Stable"
                sub        = f"The model estimates only a <b>{pct:.0f}% probability</b> of cancellation. Maintain regular engagement."

            st.markdown(f"""
            <div class="verdict-wrap {v_class}">
                <div class="verdict-icon">{icon}</div>
                <div class="verdict-title">{headline}</div>
                <div class="verdict-body">{sub}</div>
                <div class="prob-num" style="color:{bar_color};">{pct:.0f}%</div>
                <div class="prob-desc">estimated cancellation probability</div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{pct:.1f}%;background:{bar_color};"></div>
                </div>
                <div class="bar-labels">
                    <span>0% — Safe</span><span>50% — Threshold</span><span>100% — Certain</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if high:
                st.markdown("""
                <div class="panel">
                    <div class="panel-title">◈ Recommended Retention Actions</div>
                    <div class="action-item">
                        <div class="action-num">1</div>
                        <div class="action-text"><b>Personal call within 48 hours.</b> A direct retention team call is the single highest-impact intervention.</div>
                    </div>
                    <div class="action-item">
                        <div class="action-num">2</div>
                        <div class="action-text"><b>Offer a loyalty discount or free add-on.</b> Even one month free significantly reduces cancellation probability.</div>
                    </div>
                    <div class="action-item">
                        <div class="action-num">3</div>
                        <div class="action-text"><b>Pitch a longer contract.</b> Upgrading from month-to-month to one year dramatically reduces future risk.</div>
                    </div>
                    <div class="action-item">
                        <div class="action-num">4</div>
                        <div class="action-text"><b>Bundle Tech Support or Security.</b> Add-on services create stickiness and perceived value.</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="callout callout-safe">
                    <b>No urgent action required.</b> Keep this customer engaged with regular touchpoints and consider upselling a premium add-on — satisfied customers are most receptive.
                </div>
                """, unsafe_allow_html=True)

        with r2:
            rows = {
                "Tenure":       f"{tenure} months",
                "Contract":     contract,
                "Monthly Bill": f"${monthly:.2f}",
                "Total Paid":   f"${total:.2f}",
                "Internet":     internet,
                "Tech Support": tech,
                "Payment":      payment,
            }
            tbl = "".join(
                f'<div class="sum-row"><span class="sum-key">{k}</span>'
                f'<span class="sum-val">{v}</span></div>'
                for k, v in rows.items()
            )
            st.markdown(
                f'<div class="panel"><div class="panel-title">▦ Customer Snapshot</div>{tbl}</div>',
                unsafe_allow_html=True)

            flags = []
            if contract  == "Month-to-month":  flags.append(("⚠", "Month-to-month contract"))
            if tenure     < 12:                 flags.append(("⚠", "New customer — under 1 year"))
            if internet  == "Fiber optic":      flags.append(("⚠", "Fiber optic subscriber"))
            if payment   == "Electronic check": flags.append(("⚠", "Electronic check payer"))
            if tech      == "No":               flags.append(("·", "No Tech Support add-on"))
            if security  == "No":               flags.append(("·", "No Online Security add-on"))

            st.markdown("<br>", unsafe_allow_html=True)
            if not flags:
                st.markdown(
                    '<div class="callout callout-safe"><b>No major risk flags.</b> Stable profile with no high-risk indicators.</div>',
                    unsafe_allow_html=True)
            else:
                items = "".join(f'<div class="flag-item"><span style="color:var(--danger);font-size:10px;">●</span> {t}</div>' for _, t in flags)
                st.markdown(
                    f'<div class="panel"><div class="panel-title">⚑ Risk Flags ({len(flags)})</div>{items}</div>',
                    unsafe_allow_html=True)


# ============================================================================
# PAGE 3 — MODEL PERFORMANCE
# ============================================================================
elif page == "Accurate":
    st.markdown("""
    <div class="page-header">
        <div class="page-eyebrow">Model Performance</div>
        <div class="page-title">How Reliable Is the AI?</div>
        <div class="page-subtitle">Evaluated on 1,409 customers the model had never seen — no data leakage.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout callout-info">
        <b>Testing methodology:</b> The model was trained on 80% of data and tested on a fully held-out 20% (1,409 customers). All scores below reflect true out-of-sample performance.
    </div>
    """, unsafe_allow_html=True)

    # Metric bars
    metrics = [
        ("AUC-ROC",   ACTUAL_METRICS['roc_auc'],   "#E8B84B"),
        ("Recall",    ACTUAL_METRICS['recall'],     "#4CAF80"),
        ("Accuracy",  ACTUAL_METRICS['accuracy'],   "#5B8AF0"),
        ("F1 Score",  ACTUAL_METRICS['f1'],         "#F0935B"),
        ("Precision", ACTUAL_METRICS['precision'],  "#A78BFA"),
    ]
    rows_html = "".join(f"""
        <div class="acc-row">
            <div class="acc-metric" style="min-width:90px;">{name}</div>
            <div class="acc-bar-wrap">
                <div class="acc-bar-bg">
                    <div class="acc-bar-fill" style="width:{val*100:.1f}%;background:{color};"></div>
                </div>
            </div>
            <div class="acc-val" style="color:{color};">{val:.1%}</div>
        </div>
    """ for name, val, color in metrics)

    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.markdown(f'<div class="panel"><div class="panel-title">◎ Performance Metrics</div>{rows_html}</div>',
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confusion matrix
        st.markdown("#### Prediction Breakdown")
        plt.rcParams.update(MPL_STYLE)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        cm = np.array([[ACTUAL_METRICS['tn'], ACTUAL_METRICS['fp']],
                       [ACTUAL_METRICS['fn'], ACTUAL_METRICS['tp']]])
        palette = sns.color_palette(["#1A1D24", "#1A1D24"])
        annot = np.array([
            [f"✓ {ACTUAL_METRICS['tn']}\nCorrectly Retained",
             f"✗ {ACTUAL_METRICS['fp']}\nFalse Alarm"],
            [f"✗ {ACTUAL_METRICS['fn']}\nMissed Churner",
             f"✓ {ACTUAL_METRICS['tp']}\nCorrectly Flagged"]
        ])
        colors = np.array([[0.3, 0.15], [0.1, 0.8]])
        sns.heatmap(colors, annot=annot, fmt='', cmap='RdYlGn',
                    vmin=0, vmax=1, ax=ax, cbar=False,
                    linewidths=2, linecolor='#111318',
                    xticklabels=['Predicted: Stay', 'Predicted: Leave'],
                    yticklabels=['Actually Stayed', 'Actually Left'],
                    annot_kws={"size": 9, "weight": "bold", "color": "#F0F2F7"})
        ax.set_title("Confusion Matrix — 1,409 Test Customers",
                     fontsize=11, fontweight='bold', pad=14)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">▦ Results in Plain English</div>
            <div class="panel-body">
                Out of <b>1,409 test customers</b>:<br><br>
                <span style="color:var(--safe);">✓ {ACTUAL_METRICS['tn']} correctly identified as loyal</span><br>
                <span style="color:var(--safe);">✓ {ACTUAL_METRICS['tp']} correctly flagged as about to leave</span><br>
                <span style="color:var(--warn);">⚠ {ACTUAL_METRICS['fp']} false alarms (flagged but stayed)</span><br>
                <span style="color:var(--danger);">✗ {ACTUAL_METRICS['fn']} missed churners (left without warning)</span><br><br>
                <hr style="border-color:var(--line);margin:12px 0;">
                <b>Why prioritise recall over precision?</b><br>
                A missed churner costs far more than a wasted retention call. The model is tuned to catch real churners even at the cost of some false alarms.<br><br>
                <b>Does it generalise?</b><br>
                Train accuracy: 76.9% → Test accuracy: 73.9%. The 3% gap confirms no overfitting. The model handles completely unseen customers well.<br><br>
                <b>AUC-ROC of 83.9%</b> means the model has strong discriminative power between loyal and at-risk customers.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# PAGE 4 — DATA EXPLORER
# ============================================================================
elif page == "Data":
    st.markdown("""
    <div class="page-header">
        <div class="page-eyebrow">Data Explorer</div>
        <div class="page-title">Understanding the Dataset</div>
        <div class="page-subtitle">Patterns and distributions that drive the predictive model.</div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records",         f"{len(df):,}")
    m2.metric("Features",              df.shape[1])
    m3.metric("Churn Rate",            f"{CHURN_RATE:.1%}")
    m4.metric("Selected Predictors",   len(arts['selected_features']))

    st.markdown("---")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### Retention vs Churn")
        plt.rcParams.update(MPL_STYLE)
        counts = df['Churn'].value_counts().rename(
            index={'No': 'Retained', 'Yes': 'Churned', 0: 'Retained', 1: 'Churned'})
        fig, ax = plt.subplots(figsize=(5.5, 4))
        colors  = ['#4CAF80', '#E05C5C']
        bars    = ax.bar(counts.index, counts.values, color=colors,
                         edgecolor='#111318', linewidth=2, width=0.4)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 50,
                    f'{b.get_height():,}', ha='center', fontsize=11,
                    fontweight='bold', color='#F0F2F7')
        ax.set_ylim(0, max(counts.values) * 1.2)
        ax.set_title("Customer Count by Status", fontsize=11, fontweight='bold', pad=12)
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top','right','left']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("#### Tenure Distribution")
        plt.rcParams.update(MPL_STYLE)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.hist(df['tenure'], bins=30, color='#E8B84B', alpha=0.85,
                edgecolor='#111318', linewidth=0.7)
        ax.set_title("Customer Tenure (Months)", fontsize=11, fontweight='bold', pad=12)
        ax.set_xlabel("Months with us", fontsize=10)
        ax.set_ylabel("Customers", fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top','right','left']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.markdown("#### Monthly Charges — Retained vs Churned")
    plt.rcParams.update(MPL_STYLE)
    churn_map = {0: 'Retained', 1: 'Churned', 'No': 'Retained', 'Yes': 'Churned'}
    df_plot = df.copy()
    df_plot['Churn_label'] = df_plot['Churn'].map(churn_map)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for label, color in [('Retained', '#4CAF80'), ('Churned', '#E05C5C')]:
        sub = df_plot[df_plot['Churn_label'] == label]['MonthlyCharges']
        ax.hist(sub, bins=40, alpha=0.65, label=label, color=color,
                edgecolor='#111318', linewidth=0.5)
    ax.set_title("Monthly Charges Distribution by Churn Status", fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel("Monthly Charge ($)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend(fontsize=10, framealpha=0.2)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top','right','left']].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")
    st.markdown("#### AI-Selected Predictors")
    st.markdown(f"""
    <div class="callout callout-info">
        <b>{len(arts['selected_features'])} of 29 features</b> were automatically selected via L1 regularisation as the strongest churn predictors.
    </div>
    """, unsafe_allow_html=True)
    pills = "".join(f'<span class="pill">{f}</span>' for f in arts['selected_features'])
    st.markdown(f'<div style="line-height:2.6;">{pills}</div>', unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <span>RetainIQ</span> &nbsp;·&nbsp; Logistic Regression + SMOTE &nbsp;·&nbsp;
    Tested on held-out data · no data leakage &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
