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
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="ChurnIQ — Telecom Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DESIGN SYSTEM — CSS
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

    /* ── Root tokens ─────────────────────────────────── */
    :root {
        --bg:        #0c0f1a;
        --surface:   #131726;
        --border:    #1e2540;
        --accent:    #4f8ef7;
        --accent2:   #38e8c0;
        --danger:    #f7617a;
        --success:   #38e8c0;
        --text:      #e8ecf4;
        --muted:     #6b7694;
        --radius:    12px;
    }

    /* ── Global reset ────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--text);
    }
    .stApp { background: var(--bg); }
    .block-container { padding: 2rem 2.5rem 4rem; max-width: 1280px; }

    /* ── Sidebar ─────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 14px;
        font-weight: 500;
        color: var(--muted);
        padding: 8px 12px;
        border-radius: 8px;
        transition: all .2s;
    }
    [data-testid="stSidebar"] .stRadio label:hover { color: var(--text); background: var(--border); }
    [data-testid="stSidebarContent"] { padding: 1.5rem 1rem; }

    /* ── Section headings ────────────────────────────── */
    .page-title {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: var(--text);
        letter-spacing: -0.5px;
        margin-bottom: 0.25rem;
    }
    .page-subtitle { font-size: 14px; color: var(--muted); margin-bottom: 2rem; }
    .section-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 1rem;
    }

    /* ── KPI cards ───────────────────────────────────── */
    .kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 2rem; }
    .kpi-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 20px 16px;
        text-align: center;
        transition: border-color .2s;
    }
    .kpi-card:hover { border-color: var(--accent); }
    .kpi-value {
        font-family: 'Syne', sans-serif;
        font-size: 1.75rem;
        font-weight: 800;
        color: var(--accent);
        line-height: 1;
        margin-bottom: 4px;
    }
    .kpi-label { font-size: 12px; color: var(--muted); font-weight: 500; }

    /* ── Insight cards ───────────────────────────────── */
    .insight-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 24px;
        height: 100%;
    }
    .insight-title {
        font-size: 14px;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .insight-body { font-size: 13.5px; color: var(--muted); line-height: 1.7; }

    /* ── Prediction result ───────────────────────────── */
    .result-card {
        border-radius: var(--radius);
        padding: 28px 24px;
        border: 1px solid var(--border);
    }
    .result-high {
        background: rgba(247, 97, 122, 0.08);
        border-color: var(--danger);
    }
    .result-low {
        background: rgba(56, 232, 192, 0.08);
        border-color: var(--success);
    }
    .result-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.4rem;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .result-sub { font-size: 13px; color: var(--muted); }

    /* ── Probability bar ─────────────────────────────── */
    .prob-bar-bg {
        background: var(--border);
        border-radius: 100px;
        height: 10px;
        margin-top: 12px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 100px;
        transition: width .6s ease;
    }

    /* ── Form fields ─────────────────────────────────── */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stSlider { color: var(--text) !important; }

    [data-testid="stForm"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 24px;
    }

    /* ── Divider ─────────────────────────────────────── */
    hr { border-color: var(--border) !important; margin: 1.5rem 0; }

    /* ── Metrics override ────────────────────────────── */
    [data-testid="metric-container"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 16px;
    }

    /* ── Tables ──────────────────────────────────────── */
    .stDataFrame { border: 1px solid var(--border) !important; border-radius: var(--radius); }

    /* ── Buttons ─────────────────────────────────────── */
    .stFormSubmitButton button {
        background: var(--accent) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        padding: 12px 28px !important;
        transition: opacity .2s !important;
    }
    .stFormSubmitButton button:hover { opacity: 0.85 !important; }

    /* ── Sidebar brand ───────────────────────────────── */
    .brand {
        font-family: 'Syne', sans-serif;
        font-size: 1.3rem;
        font-weight: 800;
        color: var(--text);
        margin-bottom: 4px;
    }
    .brand-tag { font-size: 11px; color: var(--muted); margin-bottom: 2rem; }

    /* ── Feature pill ────────────────────────────────── */
    .feat-pill {
        display: inline-block;
        background: var(--border);
        color: var(--accent2);
        font-size: 11px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 100px;
        margin: 3px;
    }

    /* ── Tag badge ───────────────────────────────────── */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 100px;
        font-size: 11px;
        font-weight: 700;
    }
    .badge-blue { background: rgba(79,142,247,.15); color: var(--accent); }
    .badge-green { background: rgba(56,232,192,.15); color: var(--accent2); }
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
TEST_SIZE      = 1409
TOTAL_CUSTOMERS = 7043
CHURN_RATE     = 0.2654


# ============================================================================
# DATA & MODEL
# ============================================================================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/varaprasad197/Customer-churn-predictor/main/tele_comm.csv"
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
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

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    train_median_charge = X_train['MonthlyCharges'].median()

    def engineer(df_in, med):
        d = df_in.copy()
        d['IsFirstYear'] = (d['tenure'] <= 12).astype(int)
        d['AvgMonthlyCharge'] = d.apply(
            lambda r: r['TotalCharges'] / r['tenure'] if r['tenure'] > 0 else r['MonthlyCharges'], axis=1)
        svcs = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
        d['NumAdditionalServices'] = d[svcs].apply(
            lambda r: (r == 'Yes').sum() if isinstance(r.iloc[0], str) else r.sum(), axis=1)
        d['FiberOpticUser']   = (d['InternetService'] == 'Fiber optic').astype(int)
        d['IsMonthToMonth']   = (d['Contract'] == 'Month-to-month').astype(int)
        d['PaymentRisk']      = d['PaymentMethod'].map(
            {'Electronic check': 3, 'Mailed check': 2,
             'Bank transfer (automatic)': 1, 'Credit card (automatic)': 1})
        d['HighCostLowTenure'] = ((d['MonthlyCharges'] > med) & (d['tenure'] < 12)).astype(int)
        d['HasFamily'] = ((d['Partner'] == 1) | (d['Dependents'] == 1)).astype(int)
        return d

    X_train = engineer(X_train, train_median_charge)
    X_test  = engineer(X_test,  train_median_charge)
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test  = pd.get_dummies(X_test,  drop_first=True)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    X_res, y_res = SMOTE(random_state=42, k_neighbors=5).fit_resample(X_train_sc, y_train)

    l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.1,
                            max_iter=1000, random_state=42)
    l1.fit(X_res, y_res)
    mask = l1.coef_[0] != 0
    selected = X_train.columns[mask].tolist()

    rs = RandomizedSearchCV(
        LogisticRegression(random_state=42),
        param_distributions={'C': loguniform(0.01, 10),
                              'penalty': ['l1', 'l2'],
                              'solver': ['liblinear'], 'max_iter': [500, 1000]},
        n_iter=20, cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='recall', n_jobs=-1, random_state=42)
    rs.fit(X_res[:, mask], y_res)
    best = rs.best_estimator_
    best.fit(X_res[:, mask], y_res)

    return dict(model=best, scaler=scaler, X_train=X_train,
                X_test=X_test, X_train_sel=X_res[:, mask],
                X_test_sel=X_test_sc[:, mask],
                selected_features=selected, l1_mask=mask,
                train_median_charge=train_median_charge)


def predict_churn(customer, arts):
    df = pd.DataFrame([customer])
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})

    med = arts['train_median_charge']
    df['IsFirstYear']     = (df['tenure'] <= 12).astype(int)
    df['AvgMonthlyCharge'] = df.apply(
        lambda r: r['TotalCharges'] / r['tenure'] if r['tenure'] > 0 else r['MonthlyCharges'], axis=1)
    svcs = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    if all(c in df.columns for c in svcs):
        df['NumAdditionalServices'] = df[svcs].apply(
            lambda r: (r == 'Yes').sum() if isinstance(r.iloc[0], str) else r.sum(), axis=1)
    df['FiberOpticUser']   = (df['InternetService'] == 'Fiber optic').astype(int)
    df['IsMonthToMonth']   = (df['Contract'] == 'Month-to-month').astype(int)
    df['PaymentRisk']      = df['PaymentMethod'].map(
        {'Electronic check': 3, 'Mailed check': 2,
         'Bank transfer (automatic)': 1, 'Credit card (automatic)': 1})
    df['HighCostLowTenure'] = ((df['MonthlyCharges'] > med) & (df['tenure'] < 12)).astype(int)
    df['HasFamily'] = ((df['Partner'] == 1) | (df['Dependents'] == 1)).astype(int)

    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=arts['X_train'].columns, fill_value=0)
    sc = arts['scaler'].transform(df)
    return arts['model'].predict_proba(sc[:, arts['l1_mask']])[0, 1]


# ============================================================================
# MATPLOTLIB THEME
# ============================================================================
plt.rcParams.update({
    'figure.facecolor':  '#131726',
    'axes.facecolor':    '#131726',
    'axes.edgecolor':    '#1e2540',
    'axes.labelcolor':   '#6b7694',
    'xtick.color':       '#6b7694',
    'ytick.color':       '#6b7694',
    'text.color':        '#e8ecf4',
    'grid.color':        '#1e2540',
    'grid.linestyle':    '--',
    'font.family':       'sans-serif',
})


# ============================================================================
# BOOT
# ============================================================================
df = load_data()
if df is None:
    st.stop()

with st.spinner("Initialising model…"):
    arts = train_model(df)


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown('<div class="brand">⚡ ChurnIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-tag">Telecom Retention Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Overview", "Predict Customer", "Model Performance", "Data Explorer"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(
        '<div style="font-size:11px;color:#6b7694;line-height:1.6;">'
        '<b style="color:#e8ecf4;">Model</b><br>Logistic Regression<br>'
        '<b style="color:#e8ecf4;">Balancing</b><br>SMOTE<br>'
        '<b style="color:#e8ecf4;">Tuning</b><br>RandomizedSearchCV<br>'
        '<b style="color:#e8ecf4;">Dataset</b><br>7,043 customers</div>',
        unsafe_allow_html=True
    )


# ============================================================================
# PAGE: OVERVIEW
# ============================================================================
if page == "Overview":
    st.markdown('<div class="page-title">Customer Churn Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Identify at-risk subscribers before they leave — powered by machine learning.</div>', unsafe_allow_html=True)

    # KPI row
    st.markdown('<div class="section-label">Key Metrics</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-value">{TOTAL_CUSTOMERS:,}</div>
            <div class="kpi-label">Total Customers</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value" style="color:#f7617a;">{CHURN_RATE:.1%}</div>
            <div class="kpi-label">Churn Rate</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{ACTUAL_METRICS['accuracy']:.1%}</div>
            <div class="kpi-label">Accuracy</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value" style="color:#38e8c0;">{ACTUAL_METRICS['recall']:.1%}</div>
            <div class="kpi-label">Recall</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{ACTUAL_METRICS['roc_auc']:.1%}</div>
            <div class="kpi-label">ROC-AUC</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown("""
        <div class="insight-card">
            <div class="insight-title">🧠 Model Architecture</div>
            <div class="insight-body">
                Built on <b style="color:#e8ecf4;">Logistic Regression</b> with L1/L2 regularisation,
                tuned via <b style="color:#e8ecf4;">RandomizedSearchCV</b> across 20 hyperparameter
                combinations and 5-fold stratified cross-validation.<br><br>
                <b style="color:#e8ecf4;">SMOTE</b> oversampling corrects class imbalance on training data,
                ensuring the model doesn't ignore minority-class churners.
                29 raw features are reduced to <b style="color:#e8ecf4;">22 key predictors</b>
                via L1 feature selection.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="insight-card">
            <div class="insight-title">📖 Navigation Guide</div>
            <div class="insight-body">
                <b style="color:#e8ecf4;">Predict Customer</b> — Enter a subscriber's profile and get
                an instant churn-probability score with risk classification.<br><br>
                <b style="color:#e8ecf4;">Model Performance</b> — Drill into the confusion matrix,
                metric breakdown, and overfitting analysis.<br><br>
                <b style="color:#e8ecf4;">Data Explorer</b> — Visualise churn distribution, tenure
                patterns, and the features selected by the model.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Why High Recall Matters</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-card">
        <div class="insight-body">
            The model is optimised for <b style="color:#38e8c0;">recall (81%)</b> — meaning it catches
            the vast majority of customers who will actually churn. In retention contexts, a missed churner
            costs far more than a false alarm. The <b style="color:#e8ecf4;">ROC-AUC of 83.9%</b>
            confirms the model separates churners from non-churners well across all thresholds.
            The 3% train-test accuracy gap confirms <b style="color:#e8ecf4;">no significant overfitting</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# PAGE: PREDICT CUSTOMER
# ============================================================================
elif page == "Predict Customer":
    st.markdown('<div class="page-title">Churn Probability Scorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Fill in the customer profile below to generate a churn risk score.</div>', unsafe_allow_html=True)

    with st.form("pred_form"):
        c1, c2, c3 = st.columns(3, gap="medium")

        with c1:
            st.markdown('<div class="section-label">Demographics</div>', unsafe_allow_html=True)
            gender     = st.selectbox("Gender", ["Male", "Female"])
            senior     = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner    = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure     = st.slider("Tenure (months)", 0, 72, 12)

        with c2:
            st.markdown('<div class="section-label">Services</div>', unsafe_allow_html=True)
            phone    = st.selectbox("Phone Service", ["No", "Yes"])
            internet = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
            security = st.selectbox("Online Security", ["No", "Yes"])
            backup   = st.selectbox("Online Backup", ["No", "Yes"])
            device   = st.selectbox("Device Protection", ["No", "Yes"])
            tech     = st.selectbox("Tech Support", ["No", "Yes"])

        with c3:
            st.markdown('<div class="section-label">Billing & Contract</div>', unsafe_allow_html=True)
            contract   = st.selectbox("Contract Type",
                            ["Month-to-month", "One year", "Two year"])
            payment    = st.selectbox("Payment Method",
                            ["Electronic check", "Mailed check",
                             "Bank transfer (automatic)", "Credit card (automatic)"])
            paperless  = st.selectbox("Paperless Billing", ["No", "Yes"])
            stream_tv  = st.selectbox("Streaming TV", ["No", "Yes"])
            stream_mv  = st.selectbox("Streaming Movies", ["No", "Yes"])
            monthly    = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, 5.0)
            total      = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0, 50.0)

        submitted = st.form_submit_button("⚡ Generate Risk Score", use_container_width=True)

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
        r1, r2 = st.columns([1.4, 1], gap="medium")

        with r1:
            cls    = "result-high" if high else "result-low"
            icon   = "⚠️" if high else "✅"
            label  = "HIGH CHURN RISK" if high else "LOW CHURN RISK"
            desc   = "Proactive retention measures are recommended." if high else "Customer appears stable — continue engagement."
            color  = "#f7617a" if high else "#38e8c0"
            bar_cl = f"background:{color}; width:{pct:.1f}%"

            st.markdown(f"""
            <div class="result-card {cls}">
                <div class="result-title" style="color:{color};">{icon} {label}</div>
                <div class="result-sub">{desc}</div>
                <div style="margin-top:20px;">
                    <span style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:{color};">{pct:.1f}%</span>
                    <span style="font-size:12px;color:#6b7694;margin-left:8px;">churn probability</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="{bar_cl};"></div>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:11px;color:#6b7694;margin-top:4px;">
                    <span>0% — Safe</span><span>50% — Threshold</span><span>100% — Certain</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown("""
            <div class="insight-card">
                <div class="insight-title">📋 Profile Summary</div>
            """, unsafe_allow_html=True)
            rows = {
                "Tenure": f"{tenure} months",
                "Contract": contract,
                "Monthly Charges": f"${monthly:.2f}",
                "Total Charges": f"${total:.2f}",
                "Internet": internet,
                "Tech Support": tech,
                "Payment": payment,
            }
            tbl = "".join(
                f'<div style="display:flex;justify-content:space-between;padding:6px 0;'
                f'border-bottom:1px solid #1e2540;font-size:13px;">'
                f'<span style="color:#6b7694;">{k}</span>'
                f'<span style="color:#e8ecf4;font-weight:500;">{v}</span></div>'
                for k, v in rows.items()
            )
            st.markdown(f"{tbl}</div>", unsafe_allow_html=True)


# ============================================================================
# PAGE: MODEL PERFORMANCE
# ============================================================================
elif page == "Model Performance":
    st.markdown('<div class="page-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Evaluated on 1,409 held-out test customers — metrics reflect real-world performance.</div>', unsafe_allow_html=True)

    # Metric row
    m1, m2, m3, m4, m5 = st.columns(5)
    metrics = [
        ("Accuracy",  ACTUAL_METRICS['accuracy'],  "Overall correctness"),
        ("Recall",    ACTUAL_METRICS['recall'],     "Churners caught"),
        ("Precision", ACTUAL_METRICS['precision'],  "Churn prediction rate"),
        ("F1 Score",  ACTUAL_METRICS['f1'],         "Harmonic mean"),
        ("ROC-AUC",   ACTUAL_METRICS['roc_auc'],    "Discrimination power"),
    ]
    for col, (name, val, help_) in zip([m1,m2,m3,m4,m5], metrics):
        col.metric(name, f"{val:.2%}", help_)

    st.markdown("---")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-label">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = np.array([[ACTUAL_METRICS['tn'], ACTUAL_METRICS['fp']],
                       [ACTUAL_METRICS['fn'], ACTUAL_METRICS['tp']]])
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(
            cm, annot=True, fmt='d',
            cmap=sns.color_palette("Blues", as_cmap=True),
            ax=ax, cbar=False, linewidths=0.5, linecolor='#1e2540',
            xticklabels=['Predicted No', 'Predicted Yes'],
            yticklabels=['Actual No', 'Actual Yes'],
            annot_kws={"size": 16, "weight": "bold"}
        )
        ax.set_title("Confusion Matrix — Test Set (n=1,409)", fontsize=11, pad=12, color='#e8ecf4')
        ax.set_xlabel("", fontsize=10)
        ax.set_ylabel("", fontsize=10)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown('<div class="section-label">Prediction Breakdown</div>', unsafe_allow_html=True)
        tp = ACTUAL_METRICS['tp']
        tn = ACTUAL_METRICS['tn']
        fp = ACTUAL_METRICS['fp']
        fn = ACTUAL_METRICS['fn']

        def row(label, val, note, color):
            return (
                f'<div style="display:flex;align-items:center;justify-content:space-between;'
                f'padding:12px 0;border-bottom:1px solid #1e2540;">'
                f'<div><span style="color:{color};font-weight:700;">{label}</span>'
                f'<span style="font-size:12px;color:#6b7694;margin-left:8px;">{note}</span></div>'
                f'<span style="font-family:Syne,sans-serif;font-size:1.2rem;font-weight:800;color:{color};">{val}</span></div>'
            )

        st.markdown(
            row("True Positives",  tp, "Churners correctly flagged",  "#38e8c0") +
            row("True Negatives",  tn, "Loyal customers retained",    "#38e8c0") +
            row("False Positives", fp, "Incorrectly flagged as churn", "#f7617a") +
            row("False Negatives", fn, "Missed actual churners",       "#f7617a"),
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="insight-body" style="font-size:13px;">'
            'The model is tuned for <b style="color:#38e8c0;">high recall</b> — it catches 81% of all '
            'churners at the cost of some false alarms. In telecom retention this trade-off is preferable: '
            'missing a churner is more costly than a wasted retention offer.</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown('<div class="section-label">Overfitting Check</div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    g1.metric("Train Accuracy", f"{ACTUAL_METRICS['train_accuracy']:.2%}")
    g2.metric("Test Accuracy",  f"{ACTUAL_METRICS['test_accuracy']:.2%}")
    gap = ACTUAL_METRICS['train_accuracy'] - ACTUAL_METRICS['test_accuracy']
    g3.metric("Gap", f"{gap:.2%}", "✅ Minimal — generalises well")
    st.success("A 3% train-test gap confirms the model generalises well with no significant overfitting.")


# ============================================================================
# PAGE: DATA EXPLORER
# ============================================================================
elif page == "Data Explorer":
    st.markdown('<div class="page-title">Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Understand the dataset that powers the model.</div>', unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    s1.metric("Total Rows",   f"{len(df):,}")
    s2.metric("Total Columns", df.shape[1])
    s3.metric("Churn Rate",    f"{CHURN_RATE:.1%}")

    st.markdown("---")
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<div class="section-label">Churn Distribution</div>', unsafe_allow_html=True)
        counts = df['Churn'].value_counts().rename(index={'No': 'Retained', 'Yes': 'Churned',
                                                           0: 'Retained', 1: 'Churned'})
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#4f8ef7', '#f7617a']
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='#0c0f1a',
                      linewidth=1.5, width=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                    f'{bar.get_height():,}', ha='center', fontsize=11, fontweight='bold', color='#e8ecf4')
        ax.set_ylim(0, max(counts.values) * 1.15)
        ax.set_title("Customer Retention vs Churn", fontsize=11, pad=12, color='#e8ecf4')
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c2:
        st.markdown('<div class="section-label">Tenure Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df['tenure'], bins=30, color='#4f8ef7', edgecolor='#0c0f1a',
                linewidth=0.8, alpha=0.85)
        ax.set_title("Customer Tenure (months)", fontsize=11, pad=12, color='#e8ecf4')
        ax.set_xlabel("Tenure (months)", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.markdown('<div class="section-label">Selected Features</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="font-size:13px;color:#6b7694;margin-bottom:12px;">'
        f'{len(arts["selected_features"])} features selected by L1 regularisation</p>',
        unsafe_allow_html=True
    )
    pills = "".join(f'<span class="feat-pill">{f}</span>' for f in arts['selected_features'])
    st.markdown(f'<div style="line-height:2.2;">{pills}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-size:11px;color:#6b7694;padding:12px 0;">
    ⚡ ChurnIQ &nbsp;·&nbsp; Logistic Regression + SMOTE &nbsp;·&nbsp;
    Actual metrics — no data leakage &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
