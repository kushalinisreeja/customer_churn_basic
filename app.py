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
# DESIGN SYSTEM
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Lora:wght@600;700&display=swap');

    /* ══════════════════════════════════════
       LIGHT TOKENS
    ══════════════════════════════════════ */
    :root {
        --bg:                #f0f4fa;
        --surface:           #ffffff;
        --surface2:          #f8faff;
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
        --radius:            14px;
        --shadow:            0 2px 16px rgba(0,0,0,0.07);
        --cblue-text:        #1e3a8a;
        --cgreen-text:       #14532d;
        --cred-text:         #7f1d1d;
        --verdict-high-text: #7f1d1d;
        --verdict-low-text:  #14532d;
        --bar-track:         #cbd5e1;
        --pill-bg:           #dbeafe;
        --pill-text:         #1e3a8a;
        --nav-bg:            #ffffff;
        --nav-text:          #334155;
        --nav-hover-bg:      #eff6ff;
        --nav-hover-text:    #1d4ed8;
        --nav-active-bg:     #1d4ed8;
        --nav-active-text:   #ffffff;
        --nav-border:        #e2e8f0;
    }

    /* ══════════════════════════════════════
       DARK TOKENS  — OS preference
    ══════════════════════════════════════ */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg:                #0d1117;
            --surface:           #161b22;
            --surface2:          #1c2333;
            --border:            #30363d;
            --accent:            #58a6ff;
            --accent-light:      #1c2d4a;
            --green:             #3fb950;
            --green-light:       #0f2a1a;
            --red:               #f85149;
            --red-light:         #2d1b1b;
            --orange:            #f0883e;
            --purple:            #bc8cff;
            --text:              #e6edf3;
            --text-strong:       #ffffff;
            --muted:             #8b949e;
            --shadow:            0 2px 20px rgba(0,0,0,0.5);
            --cblue-text:        #a5c8ff;
            --cgreen-text:       #7ee8a2;
            --cred-text:         #ffa198;
            --verdict-high-text: #ffa198;
            --verdict-low-text:  #7ee8a2;
            --bar-track:         #30363d;
            --pill-bg:           #1c2d4a;
            --pill-text:         #79c0ff;
            --nav-bg:            #161b22;
            --nav-text:          #c9d1d9;
            --nav-hover-bg:      #1c2d4a;
            --nav-hover-text:    #58a6ff;
            --nav-active-bg:     #1c2d4a;
            --nav-active-text:   #58a6ff;
            --nav-border:        #30363d;
        }
    }

    /* ══════════════════════════════════════
       DARK TOKENS  — Streamlit toggle
    ══════════════════════════════════════ */
    [data-theme="dark"] {
        --bg:                #0d1117;
        --surface:           #161b22;
        --surface2:          #1c2333;
        --border:            #30363d;
        --accent:            #58a6ff;
        --accent-light:      #1c2d4a;
        --green:             #3fb950;
        --green-light:       #0f2a1a;
        --red:               #f85149;
        --red-light:         #2d1b1b;
        --orange:            #f0883e;
        --purple:            #bc8cff;
        --text:              #e6edf3;
        --text-strong:       #ffffff;
        --muted:             #8b949e;
        --shadow:            0 2px 20px rgba(0,0,0,0.5);
        --cblue-text:        #a5c8ff;
        --cgreen-text:       #7ee8a2;
        --cred-text:         #ffa198;
        --verdict-high-text: #ffa198;
        --verdict-low-text:  #7ee8a2;
        --bar-track:         #30363d;
        --pill-bg:           #1c2d4a;
        --pill-text:         #79c0ff;
        --nav-bg:            #161b22;
        --nav-text:          #c9d1d9;
        --nav-hover-bg:      #1c2d4a;
        --nav-hover-text:    #58a6ff;
        --nav-active-bg:     #1c2d4a;
        --nav-active-text:   #58a6ff;
        --nav-border:        #30363d;
    }

    /* ══════════════════════════════════════
       GLOBAL RESETS
    ══════════════════════════════════════ */
    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif !important;
        color: var(--text) !important;
    }
    .stApp { background: var(--bg) !important; }
    .block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1300px !important; }
    hr { border-color: var(--border) !important; margin: 1.5rem 0; }
    h1, h2, h3, h4, h5, h6 { color: var(--text-strong) !important; }
    .stMarkdown p, .stMarkdown li { color: var(--text) !important; }

    /* ══════════════════════════════════════
       SIDEBAR  — fully controlled, no Streamlit defaults
    ══════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: var(--nav-bg) !important;
        border-right: 1px solid var(--nav-border) !important;
        min-width: 240px !important;
    }
    [data-testid="stSidebarContent"] { padding: 1.5rem 1rem !important; }

    /* Hide Streamlit's default radio widget entirely */
    [data-testid="stSidebar"] .stRadio { display: none !important; }

    /* Brand */
    .brand {
        font-family: 'Lora', serif !important;
        font-size: 1.35rem; font-weight: 700;
        color: var(--text-strong) !important;
        margin-bottom: 2px;
    }
    .brand-sub {
        font-size: 11px; font-weight: 700; letter-spacing: 1px;
        text-transform: uppercase; color: var(--muted) !important;
        margin-bottom: 20px;
    }

    /* Custom nav links rendered via HTML */
    .nav-link {
        display: flex; align-items: center; gap: 10px;
        padding: 11px 14px; border-radius: 10px;
        font-size: 14px; font-weight: 700;
        color: var(--nav-text) !important;
        background: transparent;
        border: none; cursor: pointer; width: 100%;
        text-decoration: none; margin-bottom: 4px;
        transition: background .15s, color .15s;
    }
    .nav-link:hover {
        background: var(--nav-hover-bg) !important;
        color: var(--nav-hover-text) !important;
    }
    .nav-link.active {
        background: var(--nav-active-bg) !important;
        color: var(--nav-active-text) !important;
    }
    .nav-icon { font-size: 1rem; flex-shrink: 0; }

    /* Sidebar tip box */
    .sidebar-tip {
        background: var(--accent-light) !important;
        border: 1px solid var(--border);
        border-radius: 10px; padding: 13px 12px;
        font-size: 12.5px; color: var(--cblue-text) !important;
        line-height: 1.7; margin-top: 8px;
    }
    .sidebar-tip b, .sidebar-tip strong { color: var(--cblue-text) !important; }

    /* Sidebar divider label */
    .nav-section-label {
        font-size: 10px; font-weight: 800; letter-spacing: 1.2px;
        text-transform: uppercase; color: var(--muted) !important;
        padding: 0 14px; margin: 16px 0 6px;
    }

    /* ══════════════════════════════════════
       PAGE HEADER
    ══════════════════════════════════════ */
    .page-title {
        font-family: 'Lora', serif !important;
        font-size: 1.85rem; font-weight: 700;
        color: var(--text-strong) !important; margin-bottom: 4px;
    }
    .page-subtitle {
        font-size: 14.5px; color: var(--muted) !important; margin-bottom: 2rem;
    }

    /* ══════════════════════════════════════
       KPI STAT GRID
    ══════════════════════════════════════ */
    .stat-grid {
        display: grid; grid-template-columns: repeat(4, 1fr);
        gap: 14px; margin-bottom: 1.8rem;
    }
    .stat-card {
        background: var(--surface) !important;
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 20px 18px; box-shadow: var(--shadow);
        transition: transform .2s, box-shadow .2s;
    }
    .stat-card:hover { transform: translateY(-3px); box-shadow: 0 6px 24px rgba(0,0,0,0.12); }
    .stat-icon  { font-size: 1.6rem; margin-bottom: 8px; }
    .stat-value {
        font-family: 'Lora', serif !important;
        font-size: 1.9rem; font-weight: 700;
        color: var(--text-strong) !important; line-height: 1;
    }
    .stat-label {
        font-size: 11px; font-weight: 800; color: var(--muted) !important;
        text-transform: uppercase; letter-spacing: .8px; margin-top: 5px;
    }
    .stat-note { font-size: 12px; color: var(--muted) !important; margin-top: 5px; line-height: 1.4; }

    /* ══════════════════════════════════════
       INFO CARDS
    ══════════════════════════════════════ */
    .info-card {
        background: var(--surface) !important;
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 22px; box-shadow: var(--shadow); height: 100%;
    }
    .info-card-title {
        font-size: 15px; font-weight: 800;
        color: var(--text-strong) !important;
        margin-bottom: 14px; display: flex; align-items: center; gap: 8px;
    }
    .info-card-body { font-size: 13.5px; color: var(--muted) !important; line-height: 1.85; }
    .info-card-body b, .info-card-body strong { color: var(--text-strong) !important; }
    .info-card hr { border-color: var(--border) !important; }

    /* ══════════════════════════════════════
       CALLOUT BOXES
    ══════════════════════════════════════ */
    .callout {
        border-radius: var(--radius); padding: 16px 20px;
        font-size: 13.5px; line-height: 1.75; margin-bottom: 1rem;
    }
    .callout-blue  { background: var(--accent-light) !important; border-left: 4px solid var(--accent);  color: var(--cblue-text)  !important; }
    .callout-green { background: var(--green-light)  !important; border-left: 4px solid var(--green);   color: var(--cgreen-text) !important; }
    .callout-red   { background: var(--red-light)    !important; border-left: 4px solid var(--red);     color: var(--cred-text)   !important; }
    .callout-blue  b, .callout-blue  strong { color: var(--cblue-text)  !important; }
    .callout-green b, .callout-green strong { color: var(--cgreen-text) !important; }
    .callout-red   b, .callout-red   strong { color: var(--cred-text)   !important; }

    /* ══════════════════════════════════════
       VERDICT CARD
    ══════════════════════════════════════ */
    .verdict-card  { border-radius: var(--radius); padding: 28px 24px; box-shadow: var(--shadow); border: 2px solid transparent; }
    .verdict-high  { background: var(--red-light)   !important; border-color: var(--red); }
    .verdict-low   { background: var(--green-light) !important; border-color: var(--green); }
    .verdict-title { font-family: 'Lora', serif !important; font-size: 1.5rem; font-weight: 700; margin-bottom: 8px; }

    /* ══════════════════════════════════════
       PROBABILITY BAR
    ══════════════════════════════════════ */
    .prob-number { font-family: 'Lora', serif !important; font-size: 2.8rem; font-weight: 700; line-height: 1; }
    .prob-desc   { font-size: 12px; color: var(--muted) !important; margin-top: 4px; }
    .bar-track   { background: var(--bar-track) !important; border-radius: 100px; height: 12px; margin-top: 12px; overflow: hidden; }
    .bar-fill    { height: 100%; border-radius: 100px; }
    .bar-labels  { display: flex; justify-content: space-between; font-size: 10.5px; color: var(--muted) !important; margin-top: 5px; }

    /* ══════════════════════════════════════
       SUMMARY TABLE
    ══════════════════════════════════════ */
    .summary-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 9px 0; border-bottom: 1px solid var(--border); font-size: 13.5px;
    }
    .summary-key { color: var(--muted)       !important; font-weight: 600; }
    .summary-val { color: var(--text-strong) !important; font-weight: 700; }

    /* ══════════════════════════════════════
       ACTION STEPS
    ══════════════════════════════════════ */
    .action-step { display: flex; align-items: flex-start; gap: 13px; padding: 12px 0; border-bottom: 1px solid var(--border); }
    .action-num  {
        background: var(--accent); color: #ffffff !important;
        font-weight: 800; font-size: 12px; min-width: 26px; height: 26px;
        border-radius: 50%; display: flex; align-items: center;
        justify-content: center; flex-shrink: 0;
    }
    .action-text { font-size: 13.5px; color: var(--text) !important; line-height: 1.6; }
    .action-text b, .action-text strong { color: var(--text-strong) !important; }

    /* ══════════════════════════════════════
       RISK FLAGS
    ══════════════════════════════════════ */
    .flag-row {
        padding: 8px 0; font-size: 13px;
        color: var(--text) !important;
        border-bottom: 1px solid var(--border);
    }

    /* ══════════════════════════════════════
       FORM
    ══════════════════════════════════════ */
    [data-testid="stForm"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 24px !important; box-shadow: var(--shadow);
    }
    .form-section {
        font-size: 12px; font-weight: 800; text-transform: uppercase;
        letter-spacing: 1px; color: var(--accent) !important;
        margin-bottom: 10px; margin-top: 4px;
        padding-bottom: 6px; border-bottom: 2px solid var(--accent-light);
    }
    .stFormSubmitButton button {
        background: var(--accent) !important; color: #ffffff !important;
        border: none !important; border-radius: 10px !important;
        font-weight: 800 !important; font-size: 15px !important;
        padding: 13px 28px !important; width: 100% !important;
        transition: opacity .2s !important;
    }
    .stFormSubmitButton button:hover { opacity: 0.88 !important; }

    /* ══════════════════════════════════════
       STREAMLIT METRIC CARDS
    ══════════════════════════════════════ */
    [data-testid="metric-container"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 16px !important; box-shadow: var(--shadow);
    }
    [data-testid="stMetricLabel"] p { color: var(--muted) !important; font-weight: 700 !important; }
    [data-testid="stMetricValue"]   { color: var(--text-strong) !important; }
    [data-testid="stMetricDelta"]   { color: var(--muted) !important; }

    /* ══════════════════════════════════════
       FEATURE PILLS
    ══════════════════════════════════════ */
    .pill {
        display: inline-block;
        background: var(--pill-bg) !important; color: var(--pill-text) !important;
        font-size: 12px; font-weight: 700;
        padding: 4px 12px; border-radius: 100px; margin: 3px;
        border: 1px solid var(--border);
    }

    /* ══════════════════════════════════════
       FOOTER
    ══════════════════════════════════════ */
    .footer {
        text-align: center; font-size: 12px;
        color: var(--muted) !important; padding: 12px 0;
    }

    /* ══════════════════════════════════════
       STREAMLIT SELECTBOX / INPUTS
    ══════════════════════════════════════ */
    [data-baseweb="select"] > div,
    [data-baseweb="input"]  > div {
        background: var(--surface2) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
    }
    [data-baseweb="select"] span,
    [data-baseweb="input"] input {
        color: var(--text) !important;
    }

    /* ══════════════════════════════════════
       STREAMLIT SLIDER
    ══════════════════════════════════════ */
    [data-testid="stSlider"] label p { color: var(--text) !important; }

    /* ══════════════════════════════════════
       SPINNER / ALERTS
    ══════════════════════════════════════ */
    .stAlert { border-radius: var(--radius) !important; }

    /* ══════════════════════════════════════
       DATAFRAME
    ══════════════════════════════════════ */
    .stDataFrame { border: 1px solid var(--border) !important; border-radius: var(--radius); }

    /* ══════════════════════════════════════
       STAT BADGE  (small inline badge)
    ══════════════════════════════════════ */
    .badge {
        display: inline-block; border-radius: 6px;
        padding: 2px 8px; font-size: 11px; font-weight: 800;
        background: var(--accent-light) !important;
        color: var(--cblue-text) !important;
    }
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
    ("🏠", "Home",        "What's Happening?"),
    ("🔍", "Check",       "Check a Customer"),
    ("📊", "Accurate",    "How Accurate Is This?"),
    ("📁", "Data",        "Explore the Data"),
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

    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(X_train)
    Xte_sc = scaler.transform(X_test)
    Xr, yr = SMOTE(random_state=42, k_neighbors=5).fit_resample(Xtr_sc, y_train)

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
    try:
        is_dark = (st.get_option("theme.base") == "dark")
    except Exception:
        is_dark = False
    face = '#161b22' if is_dark else '#ffffff'
    edge = '#30363d' if is_dark else '#e2e8f0'
    lbl  = '#8b949e' if is_dark else '#64748b'
    txt  = '#e6edf3' if is_dark else '#0f172a'
    grd  = '#1c2333' if is_dark else '#f1f5f9'
    return {
        'figure.facecolor': face, 'axes.facecolor': face,
        'axes.edgecolor': edge,   'axes.labelcolor': lbl,
        'xtick.color': lbl,       'ytick.color': lbl,
        'text.color': txt,        'grid.color': grd,
        'grid.linestyle': '--',   'font.family': 'sans-serif',
    }

def chart_colours():
    try:
        is_dark = (st.get_option("theme.base") == "dark")
    except Exception:
        is_dark = False
    if is_dark:
        return {'bar1': '#58a6ff', 'bar2': '#f85149', 'hist': '#58a6ff'}
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
# SIDEBAR  — fully custom HTML navigation (bypasses Streamlit radio styling)
# ============================================================================
with st.sidebar:
    st.markdown("""
    <div class="brand">🏆 RetainIQ</div>
    <div class="brand-sub">Customer Retention AI</div>
    """, unsafe_allow_html=True)

    # Use Streamlit radio hidden by CSS — state stored via session_state
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    # Render custom nav buttons
    st.markdown('<div class="nav-section-label">Menu</div>', unsafe_allow_html=True)

    for key, page_key, label in PAGES:
        is_active = (st.session_state.page == page_key)
        active_cls = "active" if is_active else ""
        if st.button(
            f"{key}  {label}",
            key=f"nav_{page_key}",
            use_container_width=True,
        ):
            st.session_state.page = page_key
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div class="sidebar-tip">
        <b>💡 What is this?</b><br>
        AI-powered tool that predicts which customers are likely to cancel —
        so your team can act <em>before</em> they leave.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:11px;color:var(--muted);line-height:1.9;padding:0 2px;">
        <b style="color:var(--text-strong);">Model</b> &nbsp; Logistic Regression<br>
        <b style="color:var(--text-strong);">Balancing</b> &nbsp; SMOTE<br>
        <b style="color:var(--text-strong);">Tuning</b> &nbsp; RandomizedSearchCV<br>
        <b style="color:var(--text-strong);">Dataset</b> &nbsp; {TOTAL_CUSTOMERS:,} customers
    </div>
    """, unsafe_allow_html=True)

# ── Style the sidebar buttons to look like nav links ──────────────────────
st.markdown("""
<style>
    /* Turn every sidebar button into a nav-link style */
    [data-testid="stSidebar"] .stButton button {
        background: transparent !important;
        border: none !important;
        color: var(--nav-text) !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        text-align: left !important;
        padding: 11px 14px !important;
        border-radius: 10px !important;
        width: 100% !important;
        transition: background .15s, color .15s !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: var(--nav-hover-bg) !important;
        color: var(--nav-hover-text) !important;
    }
    [data-testid="stSidebar"] .stButton button:focus {
        outline: none !important;
        box-shadow: none !important;
    }
</style>
""", unsafe_allow_html=True)

page = st.session_state.page


# ============================================================================
# PAGE 1 — HOME
# ============================================================================
if page == "Home":
    st.markdown('<div class="page-title">📋 What\'s Happening With Our Customers?</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">A plain-English summary of the churn situation '
        'and what our AI model found.</div>', unsafe_allow_html=True)

    # KPI cards
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
            <div class="stat-label">Customers at Risk</div>
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
            <div class="stat-label">Churners AI Can Catch</div>
            <div class="stat-note">Before they actually cancel</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout callout-blue">
        <b>🤖 What does this AI tool actually do?</b><br>
        It analyses a customer's profile — tenure, plan, billing, services — and gives a
        score from 0% to 100% showing <b>how likely they are to leave</b>.
        The higher the score, the more urgent it is to reach out.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-title">🚨 Warning Signs the AI Looks For</div>
            <div class="info-card-body">
                Customers are more likely to leave when they:<br><br>
                🔴 &nbsp;Are on a <b>month-to-month contract</b> (no commitment)<br>
                🔴 &nbsp;Have been with us for <b>less than 12 months</b><br>
                🔴 &nbsp;Use <b>Fiber Optic internet</b> (easier to switch)<br>
                🔴 &nbsp;Pay by <b>electronic check</b> (simplest to cancel)<br>
                🔴 &nbsp;Have a <b>high bill</b> but short tenure<br><br>
                🟢 &nbsp;Have a <b>family / dependents</b> (more loyal)<br>
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
                Use the left menu → "Check a Customer". Enter their details
                and get an instant risk score.<br><br>
                <b>Step 2 — Act on High-Risk Customers</b><br>
                Anyone scoring above 50% should get a proactive call,
                a loyalty offer, or a contract upgrade.<br><br>
                <b>Step 3 — Track What Works</b><br>
                Monitor which outreach efforts actually reduced churn.<br><br>
                <b>No technical knowledge needed</b> — read the result and act. ✅
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="callout callout-green">
        <b>💰 The Business Case</b><br>
        Acquiring a new customer costs <b>5–7× more</b> than retaining one.
        Retaining just <b>10% of at-risk customers</b> saves
        <b>${revenue_at_risk * 0.10:,.0f} per year</b> in lost revenue.
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# PAGE 2 — CHECK A CUSTOMER
# ============================================================================
elif page == "Check":
    st.markdown('<div class="page-title">🔍 Check a Customer\'s Risk Level</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Fill in the form below to get an instant '
        'churn risk score for any customer.</div>', unsafe_allow_html=True)

    with st.form("pred_form"):
        c1, c2, c3 = st.columns(3, gap="medium")

        with c1:
            st.markdown('<div class="form-section">👤 Who Are They?</div>',
                        unsafe_allow_html=True)
            gender     = st.selectbox("Gender", ["Male", "Female"])
            senior     = st.selectbox("Senior citizen? (65+)", ["No", "Yes"])
            partner    = st.selectbox("Do they have a partner?", ["No", "Yes"])
            dependents = st.selectbox("Do they have children / dependents?", ["No", "Yes"])
            tenure     = st.slider("Months as a customer", 0, 72, 12,
                                   help="Longer tenure = more loyal")

        with c2:
            st.markdown('<div class="form-section">📱 Services They Use</div>',
                        unsafe_allow_html=True)
            phone    = st.selectbox("Phone Service?", ["No", "Yes"])
            internet = st.selectbox("Internet Service",
                                    ["No", "DSL", "Fiber optic"],
                                    help="Fiber optic customers churn more")
            security = st.selectbox("Online Security add-on?", ["No", "Yes"])
            backup   = st.selectbox("Online Backup add-on?",   ["No", "Yes"])
            device   = st.selectbox("Device Protection?",      ["No", "Yes"])
            tech     = st.selectbox("Tech Support add-on?",    ["No", "Yes"])

        with c3:
            st.markdown('<div class="form-section">💳 Billing & Contract</div>',
                        unsafe_allow_html=True)
            contract  = st.selectbox("Contract length",
                                     ["Month-to-month", "One year", "Two year"],
                                     help="Month-to-month = highest risk")
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
            css_color = "var(--red)"               if high else "var(--green)"
            sub_color = "var(--verdict-high-text)"  if high else "var(--verdict-low-text)"
            cls       = "verdict-high"              if high else "verdict-low"
            icon      = "⚠️"                        if high else "✅"
            headline  = "This Customer Is At Risk of Leaving" if high else "This Customer Looks Stable"
            sub_text  = (
                f"Our AI predicts a <b>{pct:.0f}% chance of cancelling</b>. "
                "We strongly recommend reaching out with a retention offer."
                if high else
                f"Our AI predicts only a <b>{pct:.0f}% chance of cancelling</b>. "
                "This customer is likely to stay — keep up regular engagement."
            )

            st.markdown(f"""
            <div class="verdict-card {cls}">
                <div class="verdict-title" style="color:{css_color};">{icon} {headline}</div>
                <div style="font-size:13.5px;color:{sub_color};margin-top:6px;">{sub_text}</div>
                <div style="margin-top:18px;">
                    <div class="prob-number" style="color:{css_color};">{pct:.0f}%</div>
                    <div class="prob-desc">estimated chance of leaving</div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:{pct:.1f}%;background:{css_color};"></div>
                    </div>
                    <div class="bar-labels">
                        <span>0% — Very Safe</span>
                        <span>50% — Threshold</span>
                        <span>100% — Certain</span>
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
                            A personal call from the retention team is the most effective intervention.</div>
                        </div>
                        <div class="action-step">
                            <div class="action-num">2</div>
                            <div class="action-text"><b>Offer a loyalty discount or free add-on.</b>
                            Even 1 month free significantly reduces churn probability.</div>
                        </div>
                        <div class="action-step">
                            <div class="action-num">3</div>
                            <div class="action-text"><b>Encourage a longer contract.</b>
                            Upgrading from month-to-month to a 1-year plan greatly reduces future risk.</div>
                        </div>
                        <div class="action-step">
                            <div class="action-num">4</div>
                            <div class="action-text"><b>Add value through services.</b>
                            Tech Support and Security add-ons increase stickiness.</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="callout callout-green">
                    <b>👍 No immediate action needed.</b> Keep this customer happy with
                    regular check-ins and great service. Consider upselling a premium
                    add-on — satisfied customers are more receptive.
                </div>
                """, unsafe_allow_html=True)

        with r2:
            rows = {
                "Time with us":   f"{tenure} months",
                "Contract":       contract,
                "Monthly bill":   f"${monthly:.2f}",
                "Total paid":     f"${total:.2f}",
                "Internet":       internet,
                "Tech Support":   tech,
                "Payment":        payment,
            }
            tbl = "".join(
                f'<div class="summary-row">'
                f'<span class="summary-key">{k}</span>'
                f'<span class="summary-val">{v}</span></div>'
                for k, v in rows.items()
            )
            st.markdown(
                f'<div class="info-card">'
                f'<div class="info-card-title">📋 Customer Snapshot</div>{tbl}</div>',
                unsafe_allow_html=True)

            flags = []
            if contract == "Month-to-month":  flags.append(("⚠️", "Month-to-month contract"))
            if tenure < 12:                   flags.append(("⚠️", "New customer — under 1 year"))
            if internet == "Fiber optic":     flags.append(("⚠️", "Fiber optic user"))
            if payment == "Electronic check": flags.append(("⚠️", "Electronic check payer"))
            if tech == "No":                  flags.append(("💡", "No tech support add-on"))

            st.markdown("<br>", unsafe_allow_html=True)
            if not flags:
                st.markdown(
                    '<div class="callout callout-green">'
                    '<b>No major risk flags found.</b> Stable customer profile.</div>',
                    unsafe_allow_html=True)
            else:
                items = "".join(
                    f'<div class="flag-row">{ic} {t}</div>'
                    for ic, t in flags)
                st.markdown(
                    f'<div class="info-card">'
                    f'<div class="info-card-title">🚩 Risk Flags</div>{items}</div>',
                    unsafe_allow_html=True)


# ============================================================================
# PAGE 3 — HOW ACCURATE IS THIS?
# ============================================================================
elif page == "Accurate":
    st.markdown('<div class="page-title">📊 How Accurate Is This Tool?</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Tested on 1,409 real customers the AI had '
        'never seen before — here\'s how it performed.</div>',
        unsafe_allow_html=True)

    st.markdown("""
    <div class="callout callout-blue">
        <b>🧪 How was this tested?</b><br>
        We trained the AI on 80% of the data and tested it on the remaining 20%
        (1,409 customers) it had never seen. The scores below reflect real-world performance.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-grid">
        <div class="stat-card" style="border-top:4px solid var(--accent);">
            <div class="stat-icon">🎯</div>
            <div class="stat-value" style="color:var(--accent);">74%</div>
            <div class="stat-label">Overall Correct</div>
            <div class="stat-note">Right in 74 out of 100 cases</div>
        </div>
        <div class="stat-card" style="border-top:4px solid var(--green);">
            <div class="stat-icon">🔍</div>
            <div class="stat-value" style="color:var(--green);">81%</div>
            <div class="stat-label">Churners Caught</div>
            <div class="stat-note">Finds 8 out of 10 customers who would have left</div>
        </div>
        <div class="stat-card" style="border-top:4px solid var(--orange);">
            <div class="stat-icon">📢</div>
            <div class="stat-value" style="color:var(--orange);">50.5%</div>
            <div class="stat-label">Alert Accuracy</div>
            <div class="stat-note">When it warns you, it's right half the time</div>
        </div>
        <div class="stat-card" style="border-top:4px solid var(--purple);">
            <div class="stat-icon">⭐</div>
            <div class="stat-value" style="color:var(--purple);">84%</div>
            <div class="stat-label">Separation Score</div>
            <div class="stat-note">Strong ability to tell loyal vs at-risk apart</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 1], gap="large")

    with col1:
        st.markdown("#### AI Predictions vs Reality")
        plt.rcParams.update(get_mpl_theme())
        colors_map = np.array([[0.9, 0.4], [0.3, 0.9]])
        labels = np.array([
            [f"✅ Correctly said\n'Will Stay'\n\n{ACTUAL_METRICS['tn']} customers",
             f"⚠️ False Alarm\n'Said Will Leave'\nbut stayed\n\n{ACTUAL_METRICS['fp']} customers"],
            [f"❌ Missed\n'Said Will Stay'\nbut left\n\n{ACTUAL_METRICS['fn']} customers",
             f"✅ Correctly said\n'Will Leave'\n\n{ACTUAL_METRICS['tp']} customers"]
        ])
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(colors_map, annot=labels, fmt='', cmap='RdYlGn',
                    ax=ax, cbar=False, linewidths=2, linecolor='white',
                    xticklabels=['AI Said: Will Stay', 'AI Said: Will Leave'],
                    yticklabels=['Actually Stayed', 'Actually Left'],
                    annot_kws={"size": 9.5, "weight": "bold"})
        ax.set_title("Out of 1,409 Test Customers",
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
                ✅ &nbsp;<b>{ACTUAL_METRICS['tn']}</b> — correctly identified as loyal<br>
                ✅ &nbsp;<b>{ACTUAL_METRICS['tp']}</b> — correctly flagged as about to leave<br>
                ⚠️ &nbsp;<b>{ACTUAL_METRICS['fp']}</b> — false alarm (flagged but stayed)<br>
                ❌ &nbsp;<b>{ACTUAL_METRICS['fn']}</b> — missed (left without warning)<br><br>
                <hr>
                <b>Why does it miss some churners?</b><br>
                The model prioritises catching real churners over being perfectly precise.
                A missed churner costs far more than a wasted retention call.<br><br>
                <b>Does it work on completely new customers?</b><br>
                Yes — the 3% train-to-test gap confirms no overfitting. It
                generalises well to data it has never seen. ✅<br><br>
                <b>Train accuracy:</b> 76.9% &nbsp;|&nbsp;
                <b>Test accuracy:</b> 73.9%
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# PAGE 4 — EXPLORE THE DATA
# ============================================================================
elif page == "Data":
    st.markdown('<div class="page-title">📁 Explore the Customer Data</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Charts and signals that power the model.</div>',
        unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    s1.metric("Total Records",          f"{len(df):,}")
    s2.metric("Features per Customer",  df.shape[1])
    s3.metric("Overall Churn Rate",     f"{CHURN_RATE:.1%}")

    st.markdown("---")
    col1, col2 = st.columns(2, gap="large")
    cc = chart_colours()

    with col1:
        st.markdown("#### Customers Who Stayed vs Left")
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
        ax.set_ylim(0, max(counts.values) * 1.2)
        ax.set_title("Retention vs Churn Count", fontsize=12, fontweight='bold', pad=12)
        ax.grid(axis='y', alpha=0.4)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("#### How Long Customers Stay With Us")
        plt.rcParams.update(get_mpl_theme())
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df['tenure'], bins=30, color=cc['hist'],
                edgecolor='white', linewidth=0.8, alpha=0.9)
        ax.set_title("Customer Tenure Distribution", fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Months with us", fontsize=10)
        ax.set_ylabel("Number of customers", fontsize=10)
        ax.grid(axis='y', alpha=0.4)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.markdown("#### Monthly Charges: Churned vs Retained Customers")
    plt.rcParams.update(get_mpl_theme())
    churn_map = {0: 'Retained', 1: 'Churned', 'No': 'Retained', 'Yes': 'Churned'}
    df_plot = df.copy()
    df_plot['Churn_label'] = df_plot['Churn'].map(churn_map)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for label, colour in [('Retained', cc['bar1']), ('Churned', cc['bar2'])]:
        subset = df_plot[df_plot['Churn_label'] == label]['MonthlyCharges']
        ax.hist(subset, bins=40, alpha=0.65, label=label, color=colour, edgecolor='white', linewidth=0.5)
    ax.set_title("Monthly Charges Distribution by Churn Status",
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Monthly Charges ($)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.4)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")
    st.markdown("#### Key Predictive Signals Used by the AI")
    st.markdown(f"""
    <div class="callout callout-blue">
        The AI automatically selected <b>{len(arts['selected_features'])} out of 29</b>
        available data points as the strongest predictors of churn.
    </div>
    """, unsafe_allow_html=True)
    pills = "".join(
        f'<span class="pill">{f}</span>' for f in arts['selected_features'])
    st.markdown(f'<div style="line-height:2.6;">{pills}</div>', unsafe_allow_html=True)


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div class="footer">
    🏆 RetainIQ &nbsp;·&nbsp; Logistic Regression + SMOTE &nbsp;·&nbsp;
    Tested on held-out data — no data leakage &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
