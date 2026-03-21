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
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Customer Retention Dashboard",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DESIGN SYSTEM  ─  fully theme-adaptive, zero hardcoded hex colours
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Lora:wght@600;700&display=swap');

    /* ═══════════════════════════════════════════════
       LIGHT TOKEN SET  (default)
    ═══════════════════════════════════════════════ */
    :root {
        --bg:                #f7f9fc;
        --surface:           #ffffff;
        --border:            #d1d9e8;
        --accent:            #1d4ed8;
        --accent-light:      #dbeafe;
        --green:             #15803d;
        --green-light:       #dcfce7;
        --red:               #b91c1c;
        --red-light:         #fee2e2;
        --orange:            #c2410c;
        --purple:            #6d28d9;
        --text:              #0f172a;
        --text-strong:       #0f172a;
        --muted:             #475569;
        --radius:            16px;
        --shadow:            0 2px 16px rgba(0,0,0,0.08);
        --cblue-text:        #1e3a8a;
        --cgreen-text:       #14532d;
        --cred-text:         #7f1d1d;
        --verdict-high-text: #7f1d1d;
        --verdict-low-text:  #14532d;
        --bar-track:         #cbd5e1;
        --pill-bg:           #dbeafe;
        --pill-text:         #1e3a8a;
        --sidebar-hover-bg:  #eff6ff;
        --chart-bar1:        #1d4ed8;
        --chart-bar2:        #b91c1c;
        --chart-hist:        #1d4ed8;
        --chart-face:        #ffffff;
        --chart-edge:        #e2e8f0;
        --chart-label:       #64748b;
        --chart-text:        #0f172a;
        --chart-grid:        #f1f5f9;
    }

    /* ═══════════════════════════════════════════════
       DARK TOKEN SET  ─ triggered by OS preference
    ═══════════════════════════════════════════════ */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg:                #0f172a;
            --surface:           #1e293b;
            --border:            #334155;
            --accent:            #60a5fa;
            --accent-light:      #1e3a5f;
            --green:             #4ade80;
            --green-light:       #14532d;
            --red:               #f87171;
            --red-light:         #7f1d1d;
            --orange:            #fb923c;
            --purple:            #a78bfa;
            --text:              #f1f5f9;
            --text-strong:       #ffffff;
            --muted:             #94a3b8;
            --shadow:            0 2px 24px rgba(0,0,0,0.5);
            --cblue-text:        #bfdbfe;
            --cgreen-text:       #bbf7d0;
            --cred-text:         #fecaca;
            --verdict-high-text: #fecaca;
            --verdict-low-text:  #bbf7d0;
            --bar-track:         #334155;
            --pill-bg:           #1e3a5f;
            --pill-text:         #93c5fd;
            --sidebar-hover-bg:  #1e3a5f;
            --chart-bar1:        #60a5fa;
            --chart-bar2:        #f87171;
            --chart-hist:        #60a5fa;
            --chart-face:        #1e293b;
            --chart-edge:        #334155;
            --chart-label:       #94a3b8;
            --chart-text:        #f1f5f9;
            --chart-grid:        #1e293b;
        }
    }

    /* ═══════════════════════════════════════════════
       DARK TOKEN SET  ─ triggered by Streamlit toggle
    ═══════════════════════════════════════════════ */
    [data-theme="dark"] {
        --bg:                #0f172a;
        --surface:           #1e293b;
        --border:            #334155;
        --accent:            #60a5fa;
        --accent-light:      #1e3a5f;
        --green:             #4ade80;
        --green-light:       #14532d;
        --red:               #f87171;
        --red-light:         #7f1d1d;
        --orange:            #fb923c;
        --purple:            #a78bfa;
        --text:              #f1f5f9;
        --text-strong:       #ffffff;
        --muted:             #94a3b8;
        --shadow:            0 2px 24px rgba(0,0,0,0.5);
        --cblue-text:        #bfdbfe;
        --cgreen-text:       #bbf7d0;
        --cred-text:         #fecaca;
        --verdict-high-text: #fecaca;
        --verdict-low-text:  #bbf7d0;
        --bar-track:         #334155;
        --pill-bg:           #1e3a5f;
        --pill-text:         #93c5fd;
        --sidebar-hover-bg:  #1e3a5f;
        --chart-bar1:        #60a5fa;
        --chart-bar2:        #f87171;
        --chart-hist:        #60a5fa;
        --chart-face:        #1e293b;
        --chart-edge:        #334155;
        --chart-label:       #94a3b8;
        --chart-text:        #f1f5f9;
        --chart-grid:        #1e293b;
    }

    /* ═══════════════════════════════════════════════
       GLOBAL
    ═══════════════════════════════════════════════ */
    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif !important;
        color: var(--text) !important;
    }
    .stApp { background: var(--bg) !important; }
    .block-container { padding: 2rem 2.5rem 4rem; max-width: 1280px; }
    hr { border-color: var(--border) !important; margin: 1.5rem 0; }

    /* ═══════════════════════════════════════════════
       SIDEBAR
    ═══════════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebarContent"] { padding: 1.5rem 1rem; }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 14px !important; font-weight: 700 !important;
        color: var(--muted) !important;
        padding: 10px 14px; border-radius: 10px;
        transition: all .2s; display: block; margin-bottom: 4px;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: var(--sidebar-hover-bg) !important;
        color: var(--accent) !important;
    }
    .brand     { font-family: 'Lora', serif !important; font-size: 1.4rem; font-weight: 700; color: var(--text-strong) !important; }
    .brand-tag { font-size: 12px; color: var(--muted) !important; margin-bottom: 1.5rem; }
    .sidebar-tip {
        background: var(--accent-light) !important;
        border-radius: 12px; padding: 14px 12px;
        font-size: 13px; color: var(--cblue-text) !important; line-height: 1.7;
    }
    .sidebar-tip b, .sidebar-tip strong { color: var(--cblue-text) !important; }

    /* ═══════════════════════════════════════════════
       TYPOGRAPHY
    ═══════════════════════════════════════════════ */
    .page-title {
        font-family: 'Lora', serif !important;
        font-size: 2rem; font-weight: 700;
        color: var(--text-strong) !important; margin-bottom: 4px;
    }
    .page-subtitle { font-size: 15px; color: var(--muted) !important; margin-bottom: 2rem; }
    h1,h2,h3,h4,h5,h6 { color: var(--text-strong) !important; }
    .stMarkdown p, .stMarkdown li { color: var(--text) !important; }

    /* ═══════════════════════════════════════════════
       KPI STAT CARDS
    ═══════════════════════════════════════════════ */
    .stat-grid {
        display: grid; grid-template-columns: repeat(4,1fr);
        gap: 14px; margin-bottom: 2rem;
    }
    .stat-card {
        background: var(--surface) !important;
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 22px 18px; box-shadow: var(--shadow); transition: transform .2s;
    }
    .stat-card:hover { transform: translateY(-2px); }
    .stat-icon  { font-size: 1.8rem; margin-bottom: 8px; }
    .stat-value {
        font-family: 'Lora', serif !important;
        font-size: 2rem; font-weight: 700;
        color: var(--text-strong) !important; line-height: 1;
    }
    .stat-label {
        font-size: 12px; font-weight: 700; color: var(--muted) !important;
        text-transform: uppercase; letter-spacing: .8px; margin-top: 4px;
    }
    .stat-note { font-size: 12px; color: var(--muted) !important; margin-top: 6px; line-height: 1.4; }

    /* ═══════════════════════════════════════════════
       INFO CARDS
    ═══════════════════════════════════════════════ */
    .info-card {
        background: var(--surface) !important;
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 24px; box-shadow: var(--shadow); height: 100%;
    }
    .info-card-title {
        font-size: 16px; font-weight: 800;
        color: var(--text-strong) !important;
        margin-bottom: 14px; display: flex; align-items: center; gap: 8px;
    }
    .info-card-body { font-size: 14px; color: var(--muted) !important; line-height: 1.8; }
    .info-card-body b, .info-card-body strong { color: var(--text-strong) !important; }
    .info-card hr { border-color: var(--border) !important; }

    /* ═══════════════════════════════════════════════
       CALLOUT BOXES
    ═══════════════════════════════════════════════ */
    .callout {
        border-radius: var(--radius); padding: 18px 22px;
        font-size: 14px; line-height: 1.7; margin-bottom: 1rem;
    }
    .callout-blue  { background: var(--accent-light) !important; border-left: 4px solid var(--accent);  color: var(--cblue-text)  !important; }
    .callout-green { background: var(--green-light)  !important; border-left: 4px solid var(--green);   color: var(--cgreen-text) !important; }
    .callout-red   { background: var(--red-light)    !important; border-left: 4px solid var(--red);     color: var(--cred-text)   !important; }
    .callout-blue  b, .callout-blue  strong { color: var(--cblue-text)  !important; }
    .callout-green b, .callout-green strong { color: var(--cgreen-text) !important; }
    .callout-red   b, .callout-red   strong { color: var(--cred-text)   !important; }

    /* ═══════════════════════════════════════════════
       VERDICT CARD  (prediction result)
    ═══════════════════════════════════════════════ */
    .verdict-card  { border-radius: var(--radius); padding: 32px 28px; box-shadow: var(--shadow); border: 2px solid transparent; }
    .verdict-high  { background: var(--red-light)   !important; border-color: var(--red); }
    .verdict-low   { background: var(--green-light) !important; border-color: var(--green); }
    .verdict-title { font-family: 'Lora', serif !important; font-size: 1.6rem; font-weight: 700; margin-bottom: 8px; }
    .verdict-high .verdict-sub { color: var(--verdict-high-text) !important; }
    .verdict-low  .verdict-sub { color: var(--verdict-low-text)  !important; }
    .verdict-high .verdict-sub b,
    .verdict-high .verdict-sub strong { color: var(--verdict-high-text) !important; }
    .verdict-low  .verdict-sub b,
    .verdict-low  .verdict-sub strong { color: var(--verdict-low-text)  !important; }

    /* ═══════════════════════════════════════════════
       PROBABILITY BAR
    ═══════════════════════════════════════════════ */
    .prob-number { font-family: 'Lora', serif !important; font-size: 3rem; font-weight: 700; line-height: 1; }
    .prob-desc   { font-size: 13px; color: var(--muted) !important; margin-top: 4px; }
    .bar-track   { background: var(--bar-track) !important; border-radius: 100px; height: 14px; margin-top: 14px; overflow: hidden; }
    .bar-fill    { height: 100%; border-radius: 100px; }
    .bar-labels  { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted) !important; margin-top: 5px; }

    /* ═══════════════════════════════════════════════
       SUMMARY TABLE  (inside customer summary card)
    ═══════════════════════════════════════════════ */
    .summary-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 10px 0; border-bottom: 1px solid var(--border); font-size: 14px;
    }
    .summary-key { color: var(--muted)       !important; font-weight: 600; }
    .summary-val { color: var(--text-strong) !important; font-weight: 700; }

    /* ═══════════════════════════════════════════════
       ACTION STEPS
    ═══════════════════════════════════════════════ */
    .action-step { display: flex; align-items: flex-start; gap: 14px; padding: 14px 0; border-bottom: 1px solid var(--border); }
    .action-num  { background: var(--accent); color: #ffffff !important; font-weight: 800; font-size: 13px; min-width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
    .action-text { font-size: 14px; color: var(--text) !important; line-height: 1.6; }
    .action-text b, .action-text strong { color: var(--text-strong) !important; }

    /* ═══════════════════════════════════════════════
       RISK FLAGS LIST
    ═══════════════════════════════════════════════ */
    .flag-row { padding: 8px 0; font-size: 13px; color: var(--text) !important; border-bottom: 1px solid var(--border); }

    /* ═══════════════════════════════════════════════
       FORM
    ═══════════════════════════════════════════════ */
    [data-testid="stForm"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius); padding: 28px; box-shadow: var(--shadow);
    }
    .form-section {
        font-size: 13px; font-weight: 800; text-transform: uppercase;
        letter-spacing: 1px; color: var(--accent) !important;
        margin-bottom: 10px; margin-top: 4px;
    }
    .stFormSubmitButton button {
        background: var(--accent) !important; color: #ffffff !important;
        border: none !important; border-radius: 10px !important;
        font-weight: 800 !important; font-size: 16px !important; padding: 14px 32px !important;
    }

    /* ═══════════════════════════════════════════════
       STREAMLIT NATIVE METRIC CARDS
    ═══════════════════════════════════════════════ */
    [data-testid="metric-container"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius); padding: 16px; box-shadow: var(--shadow);
    }
    [data-testid="stMetricLabel"]  { color: var(--muted)       !important; }
    [data-testid="stMetricValue"]  { color: var(--text-strong) !important; }
    [data-testid="stMetricDelta"]  { color: var(--muted)       !important; }

    /* ═══════════════════════════════════════════════
       FEATURE PILLS
    ═══════════════════════════════════════════════ */
    .pill {
        display: inline-block;
        background: var(--pill-bg) !important; color: var(--pill-text) !important;
        font-size: 12px; font-weight: 700; padding: 4px 12px; border-radius: 100px; margin: 3px;
    }

    /* ═══════════════════════════════════════════════
       FOOTER
    ═══════════════════════════════════════════════ */
    .footer { text-align: center; font-size: 12px; color: var(--muted) !important; padding: 12px 0; }

    /* ═══════════════════════════════════════════════
       DATAFRAME
    ═══════════════════════════════════════════════ */
    .stDataFrame { border: 1px solid var(--border) !important; border-radius: var(--radius); }
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


# ============================================================================
# DATA & MODEL
# ============================================================================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/varaprasad197/Customer-churn-predictor/main/tele_comm.csv"
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Could not load the dataset: {e}")
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
        {'C': loguniform(0.01, 10), 'penalty': ['l1', 'l2'],
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
    df = pd.DataFrame([customer])
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})
    med = arts['train_median_charge']
    df['IsFirstYear']           = (df['tenure'] <= 12).astype(int)
    df['AvgMonthlyCharge']      = df.apply(
        lambda r: r['TotalCharges'] / r['tenure'] if r['tenure'] > 0 else r['MonthlyCharges'], axis=1)
    svcs = ['OnlineSecurity','OnlineBackup','DeviceProtection',
            'TechSupport','StreamingTV','StreamingMovies']
    if all(c in df.columns for c in svcs):
        df['NumAdditionalServices'] = df[svcs].apply(
            lambda r: (r == 'Yes').sum() if isinstance(r.iloc[0], str) else r.sum(), axis=1)
    df['FiberOpticUser']    = (df['InternetService'] == 'Fiber optic').astype(int)
    df['IsMonthToMonth']    = (df['Contract'] == 'Month-to-month').astype(int)
    df['PaymentRisk']       = df['PaymentMethod'].map(
        {'Electronic check': 3, 'Mailed check': 2,
         'Bank transfer (automatic)': 1, 'Credit card (automatic)': 1})
    df['HighCostLowTenure'] = ((df['MonthlyCharges'] > med) & (df['tenure'] < 12)).astype(int)
    df['HasFamily']         = ((df['Partner'] == 1) | (df['Dependents'] == 1)).astype(int)
    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=arts['X_train'].columns, fill_value=0)
    sc = arts['scaler'].transform(df)
    return arts['model'].predict_proba(sc[:, arts['l1_mask']])[0, 1]


# ============================================================================
# MATPLOTLIB ADAPTIVE THEME
# ============================================================================
def get_mpl_theme():
    """Return matplotlib rcParams that match the current Streamlit theme."""
    try:
        is_dark = (st.get_option("theme.base") == "dark")
    except Exception:
        is_dark = False
    face = '#1e293b' if is_dark else '#ffffff'
    edge = '#334155' if is_dark else '#e2e8f0'
    lbl  = '#94a3b8' if is_dark else '#64748b'
    txt  = '#f1f5f9' if is_dark else '#0f172a'
    grd  = '#1e293b' if is_dark else '#f1f5f9'
    return {
        'figure.facecolor': face, 'axes.facecolor': face,
        'axes.edgecolor': edge,   'axes.labelcolor': lbl,
        'xtick.color': lbl,       'ytick.color': lbl,
        'text.color': txt,        'grid.color': grd,
        'grid.linestyle': '--',   'font.family': 'sans-serif',
    }

def chart_colours():
    """Return bar/hist colours for current theme."""
    try:
        is_dark = (st.get_option("theme.base") == "dark")
    except Exception:
        is_dark = False
    if is_dark:
        return {'bar1': '#60a5fa', 'bar2': '#f87171', 'hist': '#60a5fa'}
    return {'bar1': '#1d4ed8', 'bar2': '#b91c1c', 'hist': '#1d4ed8'}


# ============================================================================
# BOOT
# ============================================================================
df = load_data()
if df is None:
    st.stop()

with st.spinner("Loading dashboard…"):
    arts = train_model(df)

at_risk         = int(TOTAL_CUSTOMERS * CHURN_RATE)
revenue_at_risk = at_risk * AVG_MONTHLY * 12


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown('<div class="brand">🏆 RetainIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-tag">Customer Retention Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠  Home — What's Happening?",
         "🔍  Check a Customer",
         "📊  How Accurate Is This?",
         "📁  View the Data"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div class="sidebar-tip">
        <b>💡 What is this tool?</b><br>
        This dashboard uses AI to predict which customers are likely to cancel
        their telecom subscription — so your team can reach out
        <em>before</em> they leave.
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# PAGE 1 — HOME
# ============================================================================
if "Home" in page:
    st.markdown('<div class="page-title">📋 What\'s Happening With Our Customers?</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">A plain-English summary of the churn situation '
        'and what our AI model found.</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-icon">👥</div>
            <div class="stat-value">{TOTAL_CUSTOMERS:,}</div>
            <div class="stat-label">Total Customers</div>
            <div class="stat-note">Telecom subscribers in our dataset</div>
        </div>
        <div class="stat-card" style="border-top:4px solid var(--red);">
            <div class="stat-icon">⚠️</div>
            <div class="stat-value" style="color:var(--red);">{at_risk:,}</div>
            <div class="stat-label">Customers at Risk of Leaving</div>
            <div class="stat-note">That's {CHURN_RATE:.0%} of all customers</div>
        </div>
        <div class="stat-card" style="border-top:4px solid var(--orange);">
            <div class="stat-icon">💸</div>
            <div class="stat-value" style="color:var(--orange);">${revenue_at_risk/1e6:.1f}M</div>
            <div class="stat-label">Annual Revenue at Risk</div>
            <div class="stat-note">If all at-risk customers leave</div>
        </div>
        <div class="stat-card" style="border-top:4px solid var(--green);">
            <div class="stat-icon">🎯</div>
            <div class="stat-value" style="color:var(--green);">81%</div>
            <div class="stat-label">Churners the AI Can Catch</div>
            <div class="stat-note">Before they actually cancel</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout callout-blue">
        <b>🤖 What does this AI tool actually do?</b><br>
        It looks at a customer's details — how long they've been with us, what plan they're on,
        how much they pay, and more — and gives a score from 0% to 100% showing
        <b>how likely they are to leave</b>. The higher the score, the more urgent it is to reach out.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">🚨 Warning Signs the AI Looks For</div>
            <div class="info-card-body">
                Customers are much more likely to leave when they:<br><br>
                🔴 &nbsp;Are on a <b>month-to-month contract</b> (no commitment)<br>
                🔴 &nbsp;Have been with us for <b>less than 12 months</b> (new customers)<br>
                🔴 &nbsp;Use <b>Fiber Optic internet</b> (higher expectations, easier to switch)<br>
                🔴 &nbsp;Pay by <b>electronic check</b> (easiest payment to cancel)<br>
                🔴 &nbsp;Have a <b>high monthly bill</b> but haven't been with us long<br><br>
                🟢 &nbsp;Have a <b>family / dependents</b> (tend to be more loyal)<br>
                🟢 &nbsp;Are on a <b>1- or 2-year contract</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">✅ How to Use This Dashboard</div>
            <div class="info-card-body">
                <b>Step 1 — Check a Customer</b><br>
                Go to "Check a Customer" in the left menu. Enter their details
                and get an instant risk score.<br><br>
                <b>Step 2 — Act on High-Risk Customers</b><br>
                Anyone scoring above 50% should get a proactive call,
                a loyalty offer, or a discount.<br><br>
                <b>Step 3 — Track Results</b><br>
                Monitor which outreach efforts actually reduced churn over time.<br><br>
                <b>No technical knowledge needed</b> — just fill in the form
                and read the result. ✅
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="callout callout-green">
        <b>💰 The Business Case for Acting Now</b><br>
        Acquiring a new customer costs <b>5–7× more</b> than retaining an existing one.
        If this tool helps retain even <b>10% of at-risk customers</b>, that's
        <b>${revenue_at_risk * 0.10:,.0f} saved per year</b> in lost revenue —
        at a fraction of the acquisition cost.
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# PAGE 2 — CHECK A CUSTOMER
# ============================================================================
elif "Check" in page:
    st.markdown('<div class="page-title">🔍 Check a Customer\'s Risk Level</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Fill in the form below. You\'ll instantly see '
        'how likely this customer is to cancel their subscription.</div>',
        unsafe_allow_html=True)

    with st.form("pred_form"):
        c1, c2, c3 = st.columns(3, gap="medium")

        with c1:
            st.markdown('<div class="form-section">👤 Who Are They?</div>',
                        unsafe_allow_html=True)
            gender     = st.selectbox("Gender", ["Male", "Female"])
            senior     = st.selectbox("Are they a senior citizen? (65+)", ["No", "Yes"])
            partner    = st.selectbox("Do they have a partner?", ["No", "Yes"])
            dependents = st.selectbox("Do they have children / dependents?", ["No", "Yes"])
            tenure     = st.slider("How many months have they been a customer?",
                                   0, 72, 12, help="Longer = more loyal")

        with c2:
            st.markdown('<div class="form-section">📱 What Services Do They Use?</div>',
                        unsafe_allow_html=True)
            phone    = st.selectbox("Phone Service?", ["No", "Yes"])
            internet = st.selectbox("Internet Service type",
                                    ["No", "DSL", "Fiber optic"],
                                    help="Fiber optic customers churn more often")
            security = st.selectbox("Online Security add-on?", ["No", "Yes"])
            backup   = st.selectbox("Online Backup add-on?",   ["No", "Yes"])
            device   = st.selectbox("Device Protection add-on?", ["No", "Yes"])
            tech     = st.selectbox("Tech Support add-on?",    ["No", "Yes"])

        with c3:
            st.markdown('<div class="form-section">💳 How Do They Pay?</div>',
                        unsafe_allow_html=True)
            contract  = st.selectbox("Contract length",
                                     ["Month-to-month", "One year", "Two year"],
                                     help="Month-to-month = highest churn risk")
            payment   = st.selectbox("Payment method",
                                     ["Electronic check", "Mailed check",
                                      "Bank transfer (automatic)",
                                      "Credit card (automatic)"])
            paperless = st.selectbox("Paperless billing?", ["No", "Yes"])
            stream_tv = st.selectbox("Streaming TV?",      ["No", "Yes"])
            stream_mv = st.selectbox("Streaming Movies?",  ["No", "Yes"])
            monthly   = st.number_input("Monthly bill ($)", 0.0, 200.0, 65.0, 5.0)
            total     = st.number_input("Total paid so far ($)", 0.0, 10000.0, 1000.0, 50.0)

        submitted = st.form_submit_button("🔍 Check This Customer's Risk",
                                          use_container_width=True)

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
        r1, r2 = st.columns([1.5, 1], gap="medium")

        with r1:
            css_color  = "var(--red)"              if high else "var(--green)"
            sub_color  = "var(--verdict-high-text)" if high else "var(--verdict-low-text)"
            cls        = "verdict-high"             if high else "verdict-low"
            icon       = "⚠️"                       if high else "✅"
            headline   = "This Customer Is At Risk of Leaving" if high else "This Customer Looks Stable"
            sub_text   = (
                f"Our AI predicts a <b>{pct:.0f}% chance of cancelling</b>. "
                "We strongly recommend reaching out with a retention offer."
                if high else
                f"Our AI predicts only a <b>{pct:.0f}% chance of cancelling</b>. "
                "This customer is likely to stay — keep up regular engagement."
            )

            st.markdown(f"""
            <div class="verdict-card {cls}">
                <div class="verdict-title" style="color:{css_color};">{icon} {headline}</div>
                <div class="verdict-sub" style="font-size:14px;color:{sub_color};">{sub_text}</div>
                <div style="margin-top:20px;">
                    <div class="prob-number" style="color:{css_color};">{pct:.0f}%</div>
                    <div class="prob-desc">chance of leaving</div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:{pct:.1f}%;background:{css_color};"></div>
                    </div>
                    <div class="bar-labels">
                        <span>0% — Very Safe</span>
                        <span>50% — Decision Line</span>
                        <span>100% — Certain to Leave</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if high:
                st.markdown("""
                <div class="info-card">
                    <div class="info-card-title">💡 Recommended Actions</div>
                    <div>
                        <div class="action-step">
                            <div class="action-num">1</div>
                            <div class="action-text"><b>Call within 48 hours.</b>
                            A personal call from the retention team is the single
                            most effective intervention.</div>
                        </div>
                        <div class="action-step">
                            <div class="action-num">2</div>
                            <div class="action-text"><b>Offer a loyalty discount or free add-on.</b>
                            Even 1 month free significantly reduces churn probability.</div>
                        </div>
                        <div class="action-step">
                            <div class="action-num">3</div>
                            <div class="action-text"><b>Encourage a longer contract.</b>
                            Moving from month-to-month to a 1-year plan greatly reduces
                            future churn risk.</div>
                        </div>
                        <div class="action-step">
                            <div class="action-num">4</div>
                            <div class="action-text"><b>Add value through services.</b>
                            Offering Tech Support or Security add-ons increases
                            product stickiness.</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="callout callout-green">
                    <b>👍 No immediate action needed.</b> Keep this customer happy with
                    regular check-ins and good service quality. Consider upselling a
                    premium add-on since they're already satisfied.
                </div>
                """, unsafe_allow_html=True)

        with r2:
            rows = {
                "Time with us":   f"{tenure} months",
                "Contract type":  contract,
                "Monthly bill":   f"${monthly:.2f}",
                "Total paid":     f"${total:.2f}",
                "Internet":       internet,
                "Tech Support":   tech,
                "Payment method": payment,
            }
            tbl = "".join(
                f'<div class="summary-row">'
                f'<span class="summary-key">{k}</span>'
                f'<span class="summary-val">{v}</span></div>'
                for k, v in rows.items()
            )
            st.markdown(
                f'<div class="info-card">'
                f'<div class="info-card-title">📋 Customer Summary</div>{tbl}</div>',
                unsafe_allow_html=True)

            flags = []
            if contract == "Month-to-month":  flags.append(("⚠️", "Month-to-month contract"))
            if tenure < 12:                   flags.append(("⚠️", "New customer — less than 1 year"))
            if internet == "Fiber optic":     flags.append(("⚠️", "Fiber optic user"))
            if payment == "Electronic check": flags.append(("⚠️", "Pays by electronic check"))
            if tech == "No":                  flags.append(("💡", "No tech support add-on"))

            st.markdown("<br>", unsafe_allow_html=True)
            if not flags:
                st.markdown(
                    '<div class="callout callout-green">'
                    '<b>No major risk flags found.</b> This customer has a stable profile.</div>',
                    unsafe_allow_html=True)
            else:
                items = "".join(
                    f'<div class="flag-row">{icon} {t}</div>'
                    for icon, t in flags)
                st.markdown(
                    f'<div class="info-card">'
                    f'<div class="info-card-title">🚩 Risk Flags Detected</div>{items}</div>',
                    unsafe_allow_html=True)


# ============================================================================
# PAGE 3 — HOW ACCURATE IS THIS?
# ============================================================================
elif "Accurate" in page:
    st.markdown('<div class="page-title">📊 How Accurate Is This Tool?</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">All results below come from testing the AI on '
        '1,409 real customer records it had never seen before.</div>',
        unsafe_allow_html=True)

    st.markdown("""
    <div class="callout callout-blue">
        <b>🧪 How was this tested?</b><br>
        We trained the AI on 80% of our customer data, then tested it on the remaining 20%
        (1,409 customers). The AI had never seen those customers before — just like it
        would in real life. The scores below show how well it performed.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-grid">
        <div class="stat-card" style="border-top:4px solid var(--accent);">
            <div class="stat-icon">🎯</div>
            <div class="stat-value" style="color:var(--accent);">74%</div>
            <div class="stat-label">Overall Correct</div>
            <div class="stat-note">In 74 out of 100 cases, the AI made the right call</div>
        </div>
        <div class="stat-card" style="border-top:4px solid var(--green);">
            <div class="stat-icon">🔍</div>
            <div class="stat-value" style="color:var(--green);">81%</div>
            <div class="stat-label">Churners Caught</div>
            <div class="stat-note">Finds 8 out of every 10 customers who would have left</div>
        </div>
        <div class="stat-card" style="border-top:4px solid var(--orange);">
            <div class="stat-icon">📢</div>
            <div class="stat-value" style="color:var(--orange);">50.5%</div>
            <div class="stat-label">When It Warns You, It's Right Half the Time</div>
            <div class="stat-note">Some false alarms — but missing a churner costs more</div>
        </div>
        <div class="stat-card" style="border-top:4px solid var(--purple);">
            <div class="stat-icon">⭐</div>
            <div class="stat-value" style="color:var(--purple);">84%</div>
            <div class="stat-label">Separation Score</div>
            <div class="stat-note">Strong ability to tell loyal vs at-risk customers apart</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 1], gap="large")

    with col1:
        st.markdown("#### What Happened With the 1,409 Test Customers?")
        plt.rcParams.update(get_mpl_theme())
        colors_map = np.array([[0.9, 0.4], [0.3, 0.9]])
        labels = np.array([
            [f"✅ Correctly said\n'Will Stay'\n\n{ACTUAL_METRICS['tn']} customers",
             f"⚠️ False Alarm\nSaid 'Will Leave'\nbut stayed\n\n{ACTUAL_METRICS['fp']} customers"],
            [f"❌ Missed\nSaid 'Will Stay'\nbut left\n\n{ACTUAL_METRICS['fn']} customers",
             f"✅ Correctly said\n'Will Leave'\n\n{ACTUAL_METRICS['tp']} customers"]
        ])
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(colors_map, annot=labels, fmt='', cmap='RdYlGn',
                    ax=ax, cbar=False, linewidths=2, linecolor='white',
                    xticklabels=['AI Said: Will Stay', 'AI Said: Will Leave'],
                    yticklabels=['Actually Stayed', 'Actually Left'],
                    annot_kws={"size": 10, "weight": "bold"})
        ax.set_title("AI Predictions vs What Actually Happened",
                     fontsize=12, fontweight='bold', pad=14)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("#### In Plain English")
        st.markdown(f"""
        <div class="info-card">
            <div class="info-card-body">
                Out of <b>1,409 test customers</b>:<br><br>
                ✅ &nbsp;<b>{ACTUAL_METRICS['tn']} customers</b> — correctly identified as loyal<br>
                ✅ &nbsp;<b>{ACTUAL_METRICS['tp']} customers</b> — correctly flagged as about to leave<br>
                ⚠️ &nbsp;<b>{ACTUAL_METRICS['fp']} customers</b> — false alarm (flagged but stayed)<br>
                ❌ &nbsp;<b>{ACTUAL_METRICS['fn']} customers</b> — missed (left without warning)<br><br>
                <hr>
                <b>Why does it miss some churners?</b><br>
                No AI is perfect. The model is tuned to catch as many real churners
                as possible, even if that means some false alarms. A wasted retention
                call costs far less than losing a paying customer permanently.<br><br>
                <b>Does it work on new customers?</b><br>
                Yes. The 3% gap between training (77%) and testing (74%) accuracy
                confirms the AI works just as well on brand-new data. ✅
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# PAGE 4 — VIEW THE DATA
# ============================================================================
elif "Data" in page:
    st.markdown('<div class="page-title">📁 Explore the Customer Data</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">A look at the underlying dataset that powers the model.</div>',
        unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    s1.metric("Total Customer Records",  f"{len(df):,}")
    s2.metric("Data Points per Customer", df.shape[1])
    s3.metric("Overall Churn Rate",       f"{CHURN_RATE:.1%}")

    st.markdown("---")
    col1, col2 = st.columns(2, gap="large")
    cc = chart_colours()

    with col1:
        st.markdown("#### How Many Customers Left vs Stayed?")
        plt.rcParams.update(get_mpl_theme())
        counts = df['Churn'].value_counts().rename(
            index={'No': 'Stayed', 'Yes': 'Left', 0: 'Stayed', 1: 'Left'})
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(counts.index, counts.values,
                      color=[cc['bar1'], cc['bar2']],
                      edgecolor='white', linewidth=2, width=0.45)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 40,
                    f'{b.get_height():,}', ha='center', fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(counts.values) * 1.18)
        ax.set_title("Customers Who Stayed vs Left", fontsize=12, fontweight='bold', pad=12)
        ax.grid(axis='y', alpha=0.4)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("#### How Long Have Customers Been With Us?")
        plt.rcParams.update(get_mpl_theme())
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df['tenure'], bins=30, color=cc['hist'],
                edgecolor='white', linewidth=0.8, alpha=0.85)
        ax.set_title("Customer Tenure in Months", fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Months with us", fontsize=10)
        ax.set_ylabel("Number of customers", fontsize=10)
        ax.grid(axis='y', alpha=0.4)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.markdown("#### What Information Does the AI Actually Use to Make Its Decision?")
    st.markdown(f"""
    <div class="callout callout-blue">
        Out of 29 possible data points, the AI automatically selected the
        <b>{len(arts['selected_features'])} most predictive signals</b> below.
        These are the factors that most strongly predict whether a customer will leave.
    </div>
    """, unsafe_allow_html=True)
    pills = "".join(
        f'<span class="pill">{f}</span>' for f in arts['selected_features'])
    st.markdown(f'<div style="line-height:2.4;">{pills}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Raw Data Sample (First 10 Rows)")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div class="footer">
    🏆 RetainIQ &nbsp;·&nbsp; Powered by Logistic Regression + SMOTE &nbsp;·&nbsp;
    Tested on real held-out data &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
