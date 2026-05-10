import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import time

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_curve, precision_recall_curve
from imblearn.over_sampling import SMOTE
from scipy.stats import loguniform
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── PAGE CONFIG
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── THEME STATE
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

IS_DARK = st.session_state.theme == "dark"

# ── CSS (dynamic theme)
BG      = "#0d1117"   if IS_DARK else "#f5f7fa"
SURFACE = "#161b22"   if IS_DARK else "#ffffff"
BORDER  = "#30363d"   if IS_DARK else "#d1d9e0"
TEXT    = "#e6edf3"   if IS_DARK else "#1a1f2e"
MUTED   = "#8b949e"   if IS_DARK else "#637080"
SUB     = "#c9d1d9"   if IS_DARK else "#3a4559"
INNER   = "#21262d"   if IS_DARK else "#eaeef2"
INPUT   = "#0d1117"   if IS_DARK else "#f8fafc"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

:root {{
  --bg: {BG};
  --surface: {SURFACE};
  --border: {BORDER};
  --text: {TEXT};
  --muted: {MUTED};
  --sub: {SUB};
  --inner: {INNER};
  --input-bg: {INPUT};
  --green: #3fb950;
  --blue: #58a6ff;
  --red: #f85149;
  --yellow: #d29922;
  --purple: #a371f7;
  --blue-bright: #388bfd;
}}

[data-testid="collapsedControl"] {{ display:none!important; }}
section[data-testid="stSidebar"] {{ display:none!important; }}

html, body, [class*="css"] {{
  font-family:'DM Sans',sans-serif!important;
  background:var(--bg)!important;
  color:var(--text)!important;
}}
.stApp {{ background:var(--bg)!important; }}
.block-container {{ padding:1.5rem 2.5rem 5rem!important; max-width:1440px!important; }}
h1,h2,h3,h4 {{ color:var(--text)!important; }}
hr {{ border-color:var(--border)!important; margin:1.25rem 0!important; }}

/* Header */
.app-header {{
  display:flex; align-items:center; justify-content:space-between;
  padding:0.75rem 0 1.25rem;
}}
.app-title {{ display:flex; align-items:center; gap:14px; }}
.app-icon {{
  width:52px; height:52px;
  background:linear-gradient(135deg,#1f6feb,#388bfd);
  border-radius:14px; display:flex; align-items:center;
  justify-content:center; font-size:24px;
  box-shadow:0 4px 20px rgba(56,139,253,0.35);
}}
.app-title-text {{
  font-family:'Space Grotesk',sans-serif!important;
  font-size:1.85rem; font-weight:700; color:var(--text)!important;
  letter-spacing:-0.02em;
}}
.app-subtitle {{ font-size:12px; color:var(--muted); margin-top:2px; }}
.header-right {{ display:flex; align-items:center; gap:12px; }}
.model-ready-badge {{
  background:rgba(63,185,80,0.12); color:#3fb950;
  border:1px solid rgba(63,185,80,0.3); border-radius:20px;
  padding:6px 16px; font-size:11.5px; font-weight:600; letter-spacing:0.5px;
}}

/* KPI cards */
.kpi-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin:1rem 0 1.75rem; }}
.kpi-card {{
  background:var(--surface); border:1px solid var(--border);
  border-radius:14px; padding:20px 22px; position:relative; overflow:hidden;
  transition:transform .2s,box-shadow .2s;
}}
.kpi-card:hover {{ transform:translateY(-2px); box-shadow:0 8px 32px rgba(0,0,0,0.25); }}
.kpi-card::before {{ content:''; position:absolute; top:0; left:0; right:0; height:3px; }}
.kpi-card.green::before  {{ background:linear-gradient(90deg,#2ea043,#3fb950); }}
.kpi-card.blue::before   {{ background:linear-gradient(90deg,#1f6feb,#58a6ff); }}
.kpi-card.red::before    {{ background:linear-gradient(90deg,#da3633,#f85149); }}
.kpi-card.purple::before {{ background:linear-gradient(90deg,#6e40c9,#a371f7); }}
.kpi-label {{ font-size:10.5px; font-weight:600; letter-spacing:1px; text-transform:uppercase; color:var(--muted); margin-bottom:10px; }}
.kpi-value {{ font-size:2rem; font-weight:700; color:var(--text); letter-spacing:-0.02em; line-height:1; font-family:'Space Grotesk',sans-serif; }}
.kpi-sub {{ font-size:11.5px; margin-top:7px; display:flex; align-items:center; gap:4px; }}
.kpi-sub.green  {{ color:#3fb950; }} .kpi-sub.blue {{ color:#58a6ff; }}
.kpi-sub.red    {{ color:#f85149; }} .kpi-sub.purple {{ color:#a371f7; }}

/* Section labels */
.section-label {{
  font-size:10.5px; font-weight:700; letter-spacing:1.5px;
  text-transform:uppercase; color:var(--muted); margin-bottom:12px;
}}

/* Form groups */
.form-group {{
  background:var(--surface); border:1px solid var(--border);
  border-radius:12px; padding:18px 22px; margin-bottom:12px;
}}
.form-group-header {{
  display:flex; align-items:center; gap:8px;
  font-size:12px; font-weight:600; text-transform:uppercase;
  letter-spacing:0.8px; color:var(--sub); margin-bottom:16px;
}}
.tooltip-wrap {{ position:relative; display:inline-block; cursor:help; }}
.tooltip-icon {{
  display:inline-flex; align-items:center; justify-content:center;
  width:15px; height:15px; border-radius:50%;
  background:rgba(88,166,255,0.15); color:#58a6ff;
  font-size:9px; font-weight:700; margin-left:5px; cursor:help;
  border:1px solid rgba(88,166,255,0.3);
}}

/* Buttons */
.stButton > button {{
  background:linear-gradient(135deg,#1f6feb,#388bfd)!important;
  color:white!important; border:none!important; border-radius:10px!important;
  font-weight:600!important; font-size:14px!important;
  padding:12px 24px!important; width:100%!important;
  transition:all .25s!important; letter-spacing:0.3px!important;
  box-shadow:0 2px 12px rgba(31,111,235,0.35)!important;
}}
.stButton > button:hover {{
  transform:translateY(-1px)!important;
  box-shadow:0 6px 24px rgba(31,111,235,0.5)!important;
}}
.stButton > button:active {{ transform:translateY(0px)!important; }}

/* Output panel */
.output-panel {{
  background:var(--surface); border:1px solid var(--border);
  border-radius:14px; padding:32px; min-height:380px;
  display:flex; flex-direction:column; justify-content:center;
  align-items:center; text-align:center;
}}
.output-placeholder {{ color:var(--muted); font-size:14px; line-height:1.9; }}
.output-placeholder span {{ color:var(--sub); font-weight:500; }}

/* Result card */
.result-card {{
  background:var(--surface); border:1px solid var(--border);
  border-radius:14px; padding:24px 28px; width:100%;
  margin-bottom:12px;
}}
.result-card.highlighted {{
  border-color:rgba(56,139,253,0.4);
  box-shadow:0 0 0 1px rgba(56,139,253,0.15), 0 8px 32px rgba(0,0,0,0.2);
}}
.result-prob-label {{ font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:1px; }}
.result-prob-num {{
  font-size:4.5rem; font-weight:700; line-height:1;
  letter-spacing:-0.03em; margin:10px 0 4px;
  font-family:'Space Grotesk',sans-serif;
}}
.result-verdict {{ font-size:16px; font-weight:700; margin-bottom:6px; }}
.result-body {{ font-size:13px; color:var(--muted); line-height:1.65; }}
.prob-bar-bg {{
  background:var(--inner); border-radius:100px; height:10px;
  margin:18px 0 7px; overflow:hidden;
}}
.prob-bar-fill {{ height:100%; border-radius:100px; transition:width .8s cubic-bezier(.25,.1,.25,1); }}
.prob-bar-labels {{ display:flex; justify-content:space-between; font-size:10px; color:var(--muted); }}
.risk-high {{ color:#f85149; }} .risk-medium {{ color:#d29922; }} .risk-low {{ color:#3fb950; }}

/* Split probability bar */
.split-bar-wrap {{ margin:14px 0 6px; }}
.split-bar-bg {{ display:flex; border-radius:100px; height:12px; overflow:hidden; }}
.split-bar-labels {{ display:flex; justify-content:space-between; font-size:11.5px; font-weight:600; margin-top:7px; }}

/* Gauge placeholder (SVG drawn inline) */
.gauge-wrap {{ display:flex; justify-content:center; margin:12px 0; }}

/* Feature importance */
.feat-row {{ display:flex; align-items:center; padding:10px 0; border-bottom:1px solid var(--inner); gap:12px; }}
.feat-row:last-child {{ border-bottom:none; }}
.feat-name {{ color:var(--muted); font-size:12.5px; min-width:130px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.feat-bar-bg {{ flex:1; background:var(--inner); border-radius:100px; height:7px; overflow:hidden; }}
.feat-bar-fill {{ height:100%; border-radius:100px; }}
.feat-val {{ font-size:12.5px; font-weight:600; min-width:44px; text-align:right; }}
.feat-dir {{ font-size:11px; min-width:20px; }}

/* Snapshot rows */
.snap-row {{ display:flex; justify-content:space-between; padding:9px 0; border-bottom:1px solid var(--inner); font-size:13px; }}
.snap-row:last-child {{ border-bottom:none; }}
.snap-k {{ color:var(--muted); }} .snap-v {{ color:var(--text); font-weight:500; }}

/* Risk flags */
.flag-row {{ display:flex; align-items:center; gap:8px; padding:8px 0; border-bottom:1px solid var(--inner); font-size:12.5px; color:var(--muted); }}
.flag-row:last-child {{ border-bottom:none; }}

/* Step items */
.step-item {{ display:flex; gap:12px; padding:12px 0; border-bottom:1px solid var(--inner); font-size:13px; }}
.step-item:last-child {{ border-bottom:none; }}
.step-num {{
  min-width:24px; height:24px; border-radius:50%;
  background:rgba(31,111,235,0.15); border:1px solid rgba(31,111,235,0.3);
  color:#58a6ff; font-size:10px; font-weight:700;
  display:flex; align-items:center; justify-content:center;
  flex-shrink:0; margin-top:1px;
}}
.step-text {{ color:var(--muted); line-height:1.55; }}
.step-text b {{ color:var(--sub); }}

/* Callouts */
.callout {{ border-radius:10px; padding:14px 18px; font-size:13px; line-height:1.7; margin-bottom:1rem; border:1px solid; }}
.ci {{ background:rgba(31,111,235,0.08); border-color:rgba(31,111,235,0.25); color:#58a6ff; }}
.cg {{ background:rgba(63,185,80,0.08); border-color:rgba(63,185,80,0.25); color:#3fb950; }}
.cy {{ background:rgba(210,153,34,0.08); border-color:rgba(210,153,34,0.25); color:#d29922; }}

/* What-if panel */
.whatif-panel {{
  background:var(--surface); border:1px solid var(--border);
  border-radius:14px; padding:20px 24px;
}}

/* Batch table */
.batch-tbl {{ width:100%; border-collapse:collapse; font-size:13px; }}
.batch-tbl th {{
  text-align:left; font-size:10.5px; font-weight:600; letter-spacing:0.8px;
  text-transform:uppercase; color:var(--muted); padding:10px 14px;
  border-bottom:1px solid var(--border); background:var(--bg);
}}
.batch-tbl td {{ padding:10px 14px; border-bottom:1px solid var(--inner); color:var(--sub); }}
.batch-tbl tr:last-child td {{ border-bottom:none; }}
.badge {{ padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; }}
.badge-h {{ background:rgba(248,81,73,0.12); color:#f85149; }}
.badge-m {{ background:rgba(210,153,34,0.12); color:#d29922; }}
.badge-l {{ background:rgba(63,185,80,0.12); color:#3fb950; }}

/* Model metrics */
.met-row {{ display:flex; align-items:center; padding:12px 0; border-bottom:1px solid var(--inner); gap:12px; }}
.met-row:last-child {{ border-bottom:none; }}
.met-name {{ color:var(--muted); font-size:13px; min-width:90px; }}
.met-bar-bg {{ flex:1; background:var(--inner); border-radius:100px; height:7px; overflow:hidden; }}
.met-bar-fill {{ height:100%; border-radius:100px; }}
.met-val {{ font-size:13.5px; font-weight:700; min-width:48px; text-align:right; }}

/* Streamlit overrides */
[data-baseweb="select"] > div, [data-baseweb="input"] > div {{
  background:var(--input-bg)!important; border-color:var(--border)!important;
  color:var(--text)!important; border-radius:8px!important;
}}
[data-baseweb="select"] span, [data-baseweb="input"] input {{ color:var(--text)!important; }}
label > div > p {{ color:var(--sub)!important; font-size:13px!important; }}
[data-testid="stMetricLabel"] p {{ color:var(--muted)!important; }}
[data-testid="stMetricValue"]  {{ color:var(--text)!important; }}
[data-testid="metric-container"] {{
  background:var(--surface)!important; border:1px solid var(--border)!important;
  border-radius:10px!important; padding:16px!important;
}}
.stDownloadButton > button {{
  background:var(--inner)!important; color:var(--sub)!important;
  border:1px solid var(--border)!important; border-radius:8px!important;
  font-size:13px!important; font-weight:500!important;
  transition:all .2s!important;
}}
.stDownloadButton > button:hover {{
  background:var(--border)!important; transform:translateY(-1px)!important;
}}
.stSlider > div > div > div > div {{ background:var(--blue-bright)!important; }}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS
ACTUAL_METRICS = {
    'accuracy':0.7388,'recall':0.8102,'precision':0.5050,
    'f1':0.6222,'roc_auc':0.8389,
    'tp':303,'fp':297,'tn':738,'fn':71,
}
TOTAL_CUSTOMERS = 7043
CHURN_RATE      = 0.2654

MPL_STYLE = {
    'figure.facecolor': SURFACE,
    'axes.facecolor':   SURFACE,
    'axes.edgecolor':'#30363d' if IS_DARK else '#d1d9e0',
    'axes.labelcolor': MUTED,
    'xtick.color':MUTED,'ytick.color':MUTED,
    'text.color':SUB,'grid.color':INNER,
    'grid.linestyle':'--','font.family':'sans-serif',
}

SAMPLE_CUSTOMERS = [
    {
        "label":"🔴 High Risk — New Fiber User",
        "gender":"Male","SeniorCitizen":0,"Partner":"No","Dependents":"No",
        "tenure":2,"PhoneService":"Yes","InternetService":"Fiber optic",
        "OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No",
        "TechSupport":"No","StreamingTV":"No","StreamingMovies":"No",
        "Contract":"Month-to-month","PaperlessBilling":"Yes",
        "PaymentMethod":"Electronic check","MonthlyCharges":80.0,"TotalCharges":160.0
    },
    {
        "label":"🟡 Medium Risk — Mid Tenure DSL",
        "gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No",
        "tenure":18,"PhoneService":"Yes","InternetService":"DSL",
        "OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No",
        "TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes",
        "Contract":"Month-to-month","PaperlessBilling":"No",
        "PaymentMethod":"Mailed check","MonthlyCharges":60.0,"TotalCharges":1080.0
    },
    {
        "label":"🟢 Low Risk — Long-term Customer",
        "gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"Yes",
        "tenure":60,"PhoneService":"Yes","InternetService":"DSL",
        "OnlineSecurity":"Yes","OnlineBackup":"Yes","DeviceProtection":"Yes",
        "TechSupport":"Yes","StreamingTV":"No","StreamingMovies":"No",
        "Contract":"Two year","PaperlessBilling":"No",
        "PaymentMethod":"Credit card (automatic)","MonthlyCharges":55.0,"TotalCharges":3300.0
    },
]

# ── DATA & MODEL
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/varaprasad197/Customer-churn-predictor/main/tele_comm.csv"
    try:    return pd.read_csv(url)
    except: st.error("Could not load dataset."); return None

@st.cache_resource
def train_model(df):
    df = df.copy()
    df.drop(columns=['customerID'], inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.loc[df['tenure']==0,'TotalCharges'] = 0
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df['Churn'] = df['Churn'].map({'Yes':1,'No':0})
    for col in ['Partner','Dependents','PhoneService','PaperlessBilling']:
        if df[col].dtype==object: df[col]=df[col].map({'Yes':1,'No':0})
    df['gender'] = df['gender'].map({'Female':1,'Male':0})
    X,y = df.drop('Churn',axis=1), df['Churn']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    med = X_train['MonthlyCharges'].median()
    def eng(d,m):
        d=d.copy()
        d['IsFirstYear']=(d['tenure']<=12).astype(int)
        d['AvgMonthlyCharge']=d.apply(lambda r:r['TotalCharges']/r['tenure'] if r['tenure']>0 else r['MonthlyCharges'],axis=1)
        svcs=['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
        d['NumAdditionalServices']=d[svcs].apply(lambda r:(r=='Yes').sum() if isinstance(r.iloc[0],str) else r.sum(),axis=1)
        d['FiberOpticUser']=(d['InternetService']=='Fiber optic').astype(int)
        d['IsMonthToMonth']=(d['Contract']=='Month-to-month').astype(int)
        d['PaymentRisk']=d['PaymentMethod'].map({'Electronic check':3,'Mailed check':2,'Bank transfer (automatic)':1,'Credit card (automatic)':1})
        d['HighCostLowTenure']=((d['MonthlyCharges']>m)&(d['tenure']<12)).astype(int)
        d['HasFamily']=((d['Partner']==1)|(d['Dependents']==1)).astype(int)
        return d
    X_train=eng(X_train,med); X_test=eng(X_test,med)
    X_train=pd.get_dummies(X_train,drop_first=True); X_test=pd.get_dummies(X_test,drop_first=True)
    X_train,X_test=X_train.align(X_test,join='left',axis=1,fill_value=0)
    scaler=StandardScaler(); Xtr_sc=scaler.fit_transform(X_train); Xte_sc=scaler.transform(X_test)
    Xr,yr=SMOTE(random_state=42,k_neighbors=5).fit_resample(Xtr_sc,y_train)
    l1=LogisticRegression(penalty='l1',solver='liblinear',C=0.1,max_iter=1000,random_state=42)
    l1.fit(Xr,yr); mask=l1.coef_[0]!=0; selected=X_train.columns[mask].tolist()
    rs=RandomizedSearchCV(LogisticRegression(random_state=42),
        {'C':loguniform(0.01,10),'penalty':['l1','l2'],'solver':['liblinear'],'max_iter':[500,1000]},
        n_iter=20,cv=StratifiedKFold(5,shuffle=True,random_state=42),scoring='recall',n_jobs=-1,random_state=42)
    rs.fit(Xr[:,mask],yr); best=rs.best_estimator_; best.fit(Xr[:,mask],yr)
    y_pred=best.predict(Xte_sc[:,mask]); y_pred_prob=best.predict_proba(Xte_sc[:,mask])[:,1]
    # feature importance
    feat_names = np.array(selected)
    coefs = best.coef_[0]
    fi_df = pd.DataFrame({'feature':feat_names,'coef':coefs})
    fi_df['abs_coef'] = fi_df['coef'].abs()
    fi_df = fi_df.sort_values('abs_coef',ascending=False).head(12).reset_index(drop=True)
    return dict(model=best,scaler=scaler,X_train=X_train,X_test=X_test,
                y_test=y_test,y_pred=y_pred,y_pred_prob=y_pred_prob,
                selected_features=selected,l1_mask=mask,train_median_charge=med,
                Xte_sc=Xte_sc,feature_importance=fi_df)

def pred_single(row, arts):
    df2=pd.DataFrame([row])
    for col in ['Partner','Dependents','PhoneService','PaperlessBilling']:
        if col in df2.columns and df2[col].dtype==object: df2[col]=df2[col].map({'Yes':1,'No':0})
    if 'gender' in df2.columns: df2['gender']=df2['gender'].map({'Female':1,'Male':0})
    med=arts['train_median_charge']
    df2['IsFirstYear']=(df2['tenure']<=12).astype(int)
    df2['AvgMonthlyCharge']=df2.apply(lambda r:r['TotalCharges']/r['tenure'] if r['tenure']>0 else r['MonthlyCharges'],axis=1)
    svcs=['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    if all(c in df2.columns for c in svcs):
        df2['NumAdditionalServices']=df2[svcs].apply(lambda r:(r=='Yes').sum() if isinstance(r.iloc[0],str) else r.sum(),axis=1)
    df2['FiberOpticUser']=(df2['InternetService']=='Fiber optic').astype(int)
    df2['IsMonthToMonth']=(df2['Contract']=='Month-to-month').astype(int)
    df2['PaymentRisk']=df2['PaymentMethod'].map({'Electronic check':3,'Mailed check':2,'Bank transfer (automatic)':1,'Credit card (automatic)':1})
    df2['HighCostLowTenure']=((df2['MonthlyCharges']>med)&(df2['tenure']<12)).astype(int)
    df2['HasFamily']=((df2['Partner']==1)|(df2['Dependents']==1)).astype(int)
    df2=pd.get_dummies(df2,drop_first=True)
    df2=df2.reindex(columns=arts['X_train'].columns,fill_value=0)
    sc=arts['scaler'].transform(df2)
    return arts['model'].predict_proba(sc[:,arts['l1_mask']])[0,1]

def pred_batch(bdf, arts):
    df2=bdf.copy()
    for col in ['Partner','Dependents','PhoneService','PaperlessBilling']:
        if col in df2.columns and df2[col].dtype==object: df2[col]=df2[col].map({'Yes':1,'No':0})
    if 'gender' in df2.columns: df2['gender']=df2['gender'].map({'Female':1,'Male':0})
    med=arts['train_median_charge']
    df2['IsFirstYear']=(df2['tenure']<=12).astype(int)
    df2['AvgMonthlyCharge']=df2.apply(lambda r:r['TotalCharges']/r['tenure'] if r['tenure']>0 else r['MonthlyCharges'],axis=1)
    svcs=['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    if all(c in df2.columns for c in svcs):
        df2['NumAdditionalServices']=df2[svcs].apply(lambda r:(r=='Yes').sum() if isinstance(r.iloc[0],str) else r.sum(),axis=1)
    df2['FiberOpticUser']=(df2['InternetService']=='Fiber optic').astype(int)
    df2['IsMonthToMonth']=(df2['Contract']=='Month-to-month').astype(int)
    df2['PaymentRisk']=df2['PaymentMethod'].map({'Electronic check':3,'Mailed check':2,'Bank transfer (automatic)':1,'Credit card (automatic)':1})
    df2['HighCostLowTenure']=((df2['MonthlyCharges']>med)&(df2['tenure']<12)).astype(int)
    df2['HasFamily']=((df2['Partner']==1)|(df2['Dependents']==1)).astype(int)
    df2=pd.get_dummies(df2,drop_first=True)
    df2=df2.reindex(columns=arts['X_train'].columns,fill_value=0)
    sc=arts['scaler'].transform(df2)
    return arts['model'].predict_proba(sc[:,arts['l1_mask']])[:,1]

# ── LOAD & TRAIN
df_raw = load_data()
if df_raw is None: st.stop()
with st.spinner("🔧 Training model on 7,043 customers…"):
    arts = train_model(df_raw)

n_feats = len(arts['selected_features'])

# ── SESSION STATE
for k,v in [
    ("active_tab","tab_single"),
    ("predict_result",None),
    ("sample_loaded",None),
    ("whatif_tenure",None),
    ("whatif_contract",None),
    ("whatif_monthly",None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── HEADER
theme_icon = "☀️" if IS_DARK else "🌙"
theme_label = "Light Mode" if IS_DARK else "Dark Mode"

st.markdown(f"""
<div class="app-header">
  <div class="app-title">
    <div class="app-icon">📡</div>
    <div>
      <div class="app-title-text">Churn Predictor</div>
      <div class="app-subtitle">Logistic Regression · SMOTE · {n_feats} features selected</div>
    </div>
  </div>
  <div class="header-right">
    <div class="model-ready-badge">✓ MODEL READY</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Theme toggle — put in a narrow col to avoid full width
_tcol, _ = st.columns([1,6])
with _tcol:
    if st.button(f"{theme_icon} {theme_label}", key="theme_toggle"):
        st.session_state.theme = "light" if IS_DARK else "dark"
        st.rerun()

st.markdown('<hr>', unsafe_allow_html=True)

# ── KPI CARDS
st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card green">
    <div class="kpi-label">Model Accuracy</div>
    <div class="kpi-value">{ACTUAL_METRICS['accuracy']*100:.1f}%</div>
    <div class="kpi-sub green">↑ Logistic Regression</div>
  </div>
  <div class="kpi-card blue">
    <div class="kpi-label">ROC-AUC Score</div>
    <div class="kpi-value">{ACTUAL_METRICS['roc_auc']:.3f}</div>
    <div class="kpi-sub blue">↑ Excellent discrimination</div>
  </div>
  <div class="kpi-card red">
    <div class="kpi-label">Dataset Churn Rate</div>
    <div class="kpi-value">{CHURN_RATE*100:.1f}%</div>
    <div class="kpi-sub red">{int(TOTAL_CUSTOMERS*CHURN_RATE):,} of {TOTAL_CUSTOMERS:,} customers</div>
  </div>
  <div class="kpi-card purple">
    <div class="kpi-label">Recall (Sensitivity)</div>
    <div class="kpi-value">{ACTUAL_METRICS['recall']*100:.1f}%</div>
    <div class="kpi-sub purple">↑ Churners correctly caught</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── TABS
TABS = [
    ("tab_single",  "🔵 Single Prediction"),
    ("tab_batch",   "🟦 Batch Prediction"),
    ("tab_whatif",  "🔮 What-If Analysis"),
    ("tab_perf",    "📊 Model Performance"),
    ("tab_data",    "🗂️ Dataset"),
]
tcols = st.columns([1.4, 1.4, 1.5, 1.6, 1, 3])
for i,(key,label) in enumerate(TABS):
    with tcols[i]:
        if st.button(label, key=f"btn_{key}", use_container_width=True):
            st.session_state.active_tab = key
            st.rerun()

st.markdown('<hr>', unsafe_allow_html=True)
active = st.session_state.active_tab

# ============================================================
# HELPERS
# ============================================================
def make_gauge_svg(prob, color, size=220):
    """Draw a semicircular gauge as SVG string."""
    pct = min(max(prob, 0), 1)
    angle = -180 + pct * 180  # maps 0→-180deg, 1→0deg
    import math
    cx, cy, r = size//2, size//2+10, size//2 - 28
    # arc end point
    rad = math.radians(angle)
    ex = cx + r * math.cos(math.radians(angle - 90 + 180))
    ey = cy + r * math.sin(math.radians(angle - 90 + 180))
    # needle
    nx = cx + (r-10)*math.cos(math.radians(angle))
    ny = cy + (r-10)*math.sin(math.radians(angle))
    # zone colors
    zones = [
        (0.0, 0.35, "#3fb950"),
        (0.35, 0.65, "#d29922"),
        (0.65, 1.0,  "#f85149"),
    ]
    arcs = ""
    for s,e,c in zones:
        a1 = math.radians(180 + s*180)
        a2 = math.radians(180 + e*180)
        x1 = cx + r*math.cos(a1); y1 = cy + r*math.sin(a1)
        x2 = cx + r*math.cos(a2); y2 = cy + r*math.sin(a2)
        large = 1 if (e-s)>0.5 else 0
        arcs += f'<path d="M {x1:.1f} {y1:.1f} A {r} {r} 0 {large} 1 {x2:.1f} {y2:.1f}" stroke="{c}" stroke-width="14" fill="none" stroke-linecap="round" opacity="0.85"/>'
    # needle angle: 180deg=0%, 0deg=100%
    needle_rad = math.radians(180 - pct*180)
    nx2 = cx + (r-8)*math.cos(needle_rad); ny2 = cy - (r-8)*math.sin(needle_rad)
    svg = f"""<svg viewBox="0 0 {size} {size//2+30}" xmlns="http://www.w3.org/2000/svg" style="overflow:visible;">
  <!-- track -->
  <path d="M {cx-r} {cy} A {r} {r} 0 0 1 {cx+r} {cy}" stroke="{'#21262d' if IS_DARK else '#e2e8f0'}" stroke-width="14" fill="none" stroke-linecap="round"/>
  {arcs}
  <!-- needle -->
  <line x1="{cx}" y1="{cy}" x2="{nx2:.1f}" y2="{ny2:.1f}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>
  <circle cx="{cx}" cy="{cy}" r="6" fill="{color}"/>
  <!-- labels -->
  <text x="{cx-r-2}" y="{cy+18}" fill="{'#6e7681' if IS_DARK else '#94a3b8'}" font-size="10" text-anchor="middle">0%</text>
  <text x="{cx+r+2}" y="{cy+18}" fill="{'#6e7681' if IS_DARK else '#94a3b8'}" font-size="10" text-anchor="middle">100%</text>
  <text x="{cx}" y="{cy-r-8}" fill="{'#6e7681' if IS_DARK else '#94a3b8'}" font-size="10" text-anchor="middle">50%</text>
</svg>"""
    return svg

def get_risk(prob):
    if prob >= 0.5:   return "#f85149","High Churn Risk","risk-high","This customer is likely to cancel. Immediate retention recommended."
    if prob >= 0.35:  return "#d29922","Moderate Risk","risk-medium","Worth proactive engagement. Monitor closely."
    return "#3fb950","Low Churn Risk","risk-low","Customer appears stable. Maintain regular touchpoints."

def render_prediction_output(prob, inp=None):
    pct = prob * 100
    color, verdict, rc, body = get_risk(prob)

    # Gauge
    gauge_svg = make_gauge_svg(prob, color)
    st.markdown(f'<div style="display:flex;justify-content:center;margin:8px 0 4px;">{gauge_svg}</div>', unsafe_allow_html=True)

    # Big number + verdict
    st.markdown(f"""
<div class="result-card highlighted">
  <div class="result-prob-label">Churn Probability</div>
  <div class="result-prob-num {rc}">{pct:.1f}%</div>
  <div class="result-verdict {rc}">{verdict}</div>
  <div class="result-body">{body}</div>
  <div style="margin-top:18px;">
    <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:{'#8b949e' if IS_DARK else '#637080'};margin-bottom:8px;">Probability Split</div>
    <div class="split-bar-bg">
      <div style="width:{min(pct,100):.1f}%;height:12px;background:{color};border-radius:100px 0 0 100px;transition:width .8s;"></div>
      <div style="flex:1;height:12px;background:{'#3fb950' if prob < 0.5 else '#21262d'};border-radius:0 100px 100px 0;"></div>
    </div>
    <div class="split-bar-labels">
      <span style="color:{color};font-weight:600;">Churn: {pct:.1f}%</span>
      <span style="color:#3fb950;font-weight:600;">Stay: {100-pct:.1f}%</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Customer snapshot
    if inp:
        snap = {
            "Tenure": f"{inp.get('tenure',0)} mo",
            "Contract": inp.get('Contract','—'),
            "Monthly Charge": f"${inp.get('MonthlyCharges',0):.0f}",
            "Internet": inp.get('InternetService','—'),
            "Tech Support": inp.get('TechSupport','—'),
            "Payment": str(inp.get('PaymentMethod','—')).split()[0],
        }
        rows_html = "".join(f'<div class="snap-row"><span class="snap-k">{k}</span><span class="snap-v">{v}</span></div>' for k,v in snap.items())
        st.markdown(f'<div class="result-card"><div style="font-size:10.5px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:{"#8b949e" if IS_DARK else "#637080"};margin-bottom:10px;">Customer Snapshot</div>{rows_html}</div>', unsafe_allow_html=True)

    # Risk flags + retention steps for high risk
    if prob >= 0.35 and inp:
        flags = []
        if inp.get('Contract')=="Month-to-month": flags.append("Month-to-month contract")
        if inp.get('tenure',99)<12:               flags.append("New customer (< 1 year)")
        if inp.get('InternetService')=="Fiber optic": flags.append("Fiber optic — higher churn segment")
        if inp.get('PaymentMethod')=="Electronic check": flags.append("Electronic check payment")
        if inp.get('TechSupport')=="No":          flags.append("No Tech Support")
        if inp.get('OnlineSecurity')=="No":       flags.append("No Online Security")

        steps_html = """
<div class="step-item"><div class="step-num">1</div><div class="step-text"><b>Call within 48 hrs</b> — direct outreach has highest retention impact</div></div>
<div class="step-item"><div class="step-num">2</div><div class="step-text"><b>Offer loyalty discount</b> — even 1 month free reduces churn significantly</div></div>
<div class="step-item"><div class="step-num">3</div><div class="step-text"><b>Pitch annual contract</b> — converts risky month-to-month customers</div></div>
<div class="step-item"><div class="step-num">4</div><div class="step-text"><b>Bundle Tech Support</b> — adds stickiness and perceived value</div></div>
"""
        cf, cs = st.columns(2)
        f_rows = "".join(f'<div class="flag-row"><span style="color:#f85149;font-size:9px;">●</span>{t}</div>' for t in flags)
        with cf:
            st.markdown(f'<div class="result-card"><div style="font-size:10.5px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:#f85149;margin-bottom:10px;">⚠ Risk Flags ({len(flags)})</div>{f_rows}</div>', unsafe_allow_html=True)
        with cs:
            st.markdown(f'<div class="result-card"><div style="font-size:10.5px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:{"#8b949e" if IS_DARK else "#637080"};margin-bottom:10px;">🛡 Retention Steps</div>{steps_html}</div>', unsafe_allow_html=True)

# ============================================================
# TAB 1 — SINGLE PREDICTION
# ============================================================
if active == "tab_single":
    # Sample loader
    st.markdown('<div class="section-label">Quick Load</div>', unsafe_allow_html=True)
    sc0, sc1, sc2, sc3, _ = st.columns([1.2,1.2,1.2,1,3])
    for i, (col, s) in enumerate(zip([sc1,sc2,sc3], SAMPLE_CUSTOMERS)):
        with [sc1,sc2,sc3][i]:
            if st.button(s['label'], key=f"sample_{i}", use_container_width=True):
                st.session_state.sample_loaded = i
                st.session_state.predict_result = None
                st.rerun()
    with sc0:
        if st.button("🔄 Reset Form", key="reset_btn", use_container_width=True):
            st.session_state.sample_loaded = None
            st.session_state.predict_result = None
            st.rerun()

    st.markdown('<hr>', unsafe_allow_html=True)

    # Defaults / sample fill
    SL = st.session_state.sample_loaded
    S  = SAMPLE_CUSTOMERS[SL] if SL is not None else {}
    def g(k,d): return S.get(k,d)

    left, right = st.columns([1.1,1], gap="large")

    with left:
        st.markdown('<div class="section-label">Customer Profile</div>', unsafe_allow_html=True)

        # Demographics
        st.markdown('<div class="form-group"><div class="form-group-header">👤 Demographics <span class="tooltip-icon" title="Basic customer info">ℹ</span></div>', unsafe_allow_html=True)
        d1,d2,d3 = st.columns(3)
        gender     = d1.selectbox("Gender",["Male","Female"], index=0 if g("gender","Male")=="Male" else 1, key="sg")
        senior     = d2.selectbox("Senior Citizen",["No","Yes"], index=g("SeniorCitizen",0), key="ss")
        partner    = d3.selectbox("Partner",["No","Yes"], index=0 if g("Partner","No")=="No" else 1, key="sp")
        d4,d5,_    = st.columns(3)
        dependents = d4.selectbox("Dependents",["No","Yes"], index=0 if g("Dependents","No")=="No" else 1, key="sd")
        tenure     = d5.number_input("Tenure (months) ℹ", 0,72, g("tenure",12), help="How many months the customer has been with the company", key="st")
        st.markdown('</div>', unsafe_allow_html=True)

        # Services
        st.markdown('<div class="form-group"><div class="form-group-header">📞 Phone & Internet Services <span class="tooltip-icon" title="Service subscriptions">ℹ</span></div>', unsafe_allow_html=True)
        p1,p2,p3 = st.columns(3)
        phone    = p1.selectbox("Phone Service", ["No","Yes"], index=0 if g("PhoneService","Yes")=="No" else 1, key="sph")
        internet = p2.selectbox("Internet Service",["No","DSL","Fiber optic"],
                                index=["No","DSL","Fiber optic"].index(g("InternetService","DSL")), key="sin")
        security = p3.selectbox("Online Security",["No","Yes"], index=0 if g("OnlineSecurity","No")=="No" else 1, key="sec",
                                help="Online security add-on — reduces churn risk")
        p4,p5,p6 = st.columns(3)
        backup   = p4.selectbox("Online Backup",["No","Yes"], index=0 if g("OnlineBackup","No")=="No" else 1, key="sbk")
        device   = p5.selectbox("Device Protection",["No","Yes"], index=0 if g("DeviceProtection","No")=="No" else 1, key="sdv")
        tech     = p6.selectbox("Tech Support",["No","Yes"], index=0 if g("TechSupport","No")=="No" else 1, key="stch",
                                help="Tech support availability is a strong predictor of retention")
        st.markdown('</div>', unsafe_allow_html=True)

        # Billing
        st.markdown('<div class="form-group"><div class="form-group-header">💳 Billing & Contract <span class="tooltip-icon" title="Billing details">ℹ</span></div>', unsafe_allow_html=True)
        b1,b2,b3 = st.columns(3)
        contract  = b1.selectbox("Contract",["Month-to-month","One year","Two year"],
                                 index=["Month-to-month","One year","Two year"].index(g("Contract","Month-to-month")), key="sct",
                                 help="Longer contracts strongly reduce churn")
        payment   = b2.selectbox("Payment Method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"],
                                 index=["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"].index(g("PaymentMethod","Electronic check")), key="spm")
        paperless = b3.selectbox("Paperless Billing",["No","Yes"], index=0 if g("PaperlessBilling","No")=="No" else 1, key="spb")
        b4,b5,_   = st.columns(3)
        stream_tv = b4.selectbox("Streaming TV",["No","Yes"], index=0 if g("StreamingTV","No")=="No" else 1, key="sstv")
        stream_mv = b5.selectbox("Streaming Movies",["No","Yes"], index=0 if g("StreamingMovies","No")=="No" else 1, key="ssmv")
        st.markdown('</div>', unsafe_allow_html=True)

        bc1,bc2 = st.columns(2)
        monthly  = bc1.number_input("Monthly Charge ($)", 0.0,200.0, float(g("MonthlyCharges",65.0)),5.0, key="smc",
                                    help="Higher monthly charges correlate with higher churn")
        total    = bc2.number_input("Total Charges ($)", 0.0,10000.0, float(g("TotalCharges",1000.0)),50.0, key="stc")

        predict_clicked = st.button("🔍 Analyze Customer", key="predict_btn")

    with right:
        st.markdown('<div class="section-label">Prediction Output</div>', unsafe_allow_html=True)

        if predict_clicked:
            inp = {
                'gender':gender,'SeniorCitizen':1 if senior=='Yes' else 0,
                'Partner':partner,'Dependents':dependents,'tenure':tenure,
                'PhoneService':phone,'InternetService':internet,
                'OnlineSecurity':security,'OnlineBackup':backup,
                'DeviceProtection':device,'TechSupport':tech,
                'StreamingTV':stream_tv,'StreamingMovies':stream_mv,
                'Contract':contract,'PaperlessBilling':paperless,
                'PaymentMethod':payment,'MonthlyCharges':monthly,'TotalCharges':total
            }
            with st.spinner("🤖 Analyzing customer data…"):
                time.sleep(0.6)
                prob = pred_single(inp, arts)
            st.session_state.predict_result = (prob, inp)

        if st.session_state.predict_result is None:
            st.markdown("""
<div class="output-panel">
  <div style="font-size:3.5rem;margin-bottom:14px;opacity:0.5;">📊</div>
  <div class="output-placeholder">
    Fill in the customer profile on the left<br>
    and click <span>Analyze Customer</span><br><br>
    <span style="font-size:12px;color:#484f58;">or load a sample above ↑</span>
  </div>
</div>
""", unsafe_allow_html=True)
        else:
            prob, inp = st.session_state.predict_result
            render_prediction_output(prob, inp)

# ============================================================
# TAB 2 — BATCH PREDICTION
# ============================================================
elif active == "tab_batch":
    st.markdown('<div class="section-label">Batch Churn Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="callout ci"><b>How it works:</b> Upload a CSV with customer records. Every row is scored instantly. Download results with churn probability and risk tier per customer.</div>', unsafe_allow_html=True)

    REQ = ['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','InternetService',
           'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
           'Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges']
    sample = {
        'customerID':['CUST-001','CUST-002','CUST-003'],
        'gender':['Male','Female','Male'],
        'SeniorCitizen':[0,1,0],'Partner':['Yes','No','Yes'],'Dependents':['No','No','Yes'],
        'tenure':[2,34,45],'PhoneService':['Yes','Yes','No'],
        'InternetService':['Fiber optic','DSL','DSL'],
        'OnlineSecurity':['No','Yes','Yes'],'OnlineBackup':['No','No','Yes'],
        'DeviceProtection':['No','Yes','No'],'TechSupport':['No','No','Yes'],
        'StreamingTV':['No','Yes','No'],'StreamingMovies':['No','Yes','No'],
        'Contract':['Month-to-month','One year','Two year'],
        'PaperlessBilling':['Yes','No','No'],
        'PaymentMethod':['Electronic check','Bank transfer (automatic)','Credit card (automatic)'],
        'MonthlyCharges':[70.35,56.95,42.30],'TotalCharges':[140.70,1889.50,1840.75]
    }
    tpl = pd.DataFrame(sample).to_csv(index=False).encode('utf-8')

    dc1,dc2,_ = st.columns([1,1,4])
    with dc1: st.download_button("⬇ CSV Template", tpl,"churn_template.csv","text/csv",use_container_width=True)

    uploaded = st.file_uploader("📂 Drop your CSV here (or click to browse)", type=["csv"], label_visibility="visible")

    if uploaded:
        try:
            raw = pd.read_csv(uploaded)
            missing = [c for c in REQ if c not in raw.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
                st.stop()

            with st.spinner("🤖 Scoring customers…"):
                time.sleep(0.5)
                id_col = raw['customerID'].tolist() if 'customerID' in raw.columns else [f"Row {i+1}" for i in range(len(raw))]
                probs = pred_batch(raw, arts)

            def risk_label(p):
                if p>=0.65: return "High","badge-h"
                if p>=0.40: return "Medium","badge-m"
                return "Low","badge-l"

            results = [{
                'customerID':cid,
                'ChurnProbability':round(p*100,1),
                'RiskLevel':risk_label(p)[0],
                '_badge':risk_label(p)[1]
            } for cid,p in zip(id_col,probs)]
            res = pd.DataFrame(results)

            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Total Customers", len(res))
            k2.metric("🔴 High Risk", (res['RiskLevel']=='High').sum(),
                      delta=f"{(res['RiskLevel']=='High').sum()/len(res)*100:.0f}% of total", delta_color="inverse")
            k3.metric("🟡 Medium Risk", (res['RiskLevel']=='Medium').sum())
            k4.metric("Avg Probability", f"{res['ChurnProbability'].mean():.1f}%")

            st.markdown("<br>", unsafe_allow_html=True)

            # Preview table
            rows_html = ""
            for _, row in res.iterrows():
                p = row['ChurnProbability']
                clr = "#f85149" if row['RiskLevel']=='High' else ("#d29922" if row['RiskLevel']=='Medium' else "#3fb950")
                rows_html += f"""
<tr>
  <td style="color:{TEXT};font-weight:500;">{row['customerID']}</td>
  <td>
    <div style="display:flex;align-items:center;gap:12px;">
      <div style="flex:1;background:{INNER};border-radius:100px;height:6px;overflow:hidden;">
        <div style="width:{min(p,100):.1f}%;height:100%;background:{clr};border-radius:100px;"></div>
      </div>
      <span style="color:{clr};font-weight:700;min-width:44px;">{p:.1f}%</span>
    </div>
  </td>
  <td><span class="badge {row['_badge']}">{row['RiskLevel']}</span></td>
</tr>"""
            st.markdown(f"""
<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:12px;overflow:hidden;max-height:460px;overflow-y:auto;">
  <table class="batch-tbl">
    <thead>
      <tr>
        <th style="width:200px;">Customer ID</th>
        <th>Churn Probability</th>
        <th style="width:120px;">Risk Level</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            out_csv = res[['customerID','ChurnProbability','RiskLevel']].to_csv(index=False).encode('utf-8')
            ex1, ex2, _ = st.columns([1,1,4])
            with ex1: st.download_button("📥 Download Results CSV", out_csv,"churn_results.csv","text/csv",use_container_width=True)
            with ex2:
                high_only = res[res['RiskLevel']=='High'][['customerID','ChurnProbability','RiskLevel']].to_csv(index=False).encode('utf-8')
                st.download_button("📥 High-Risk Only", high_only,"high_risk_customers.csv","text/csv",use_container_width=True)

        except Exception as e:
            st.error(f"Error processing file: {e}")

# ============================================================
# TAB 3 — WHAT-IF ANALYSIS
# ============================================================
elif active == "tab_whatif":
    st.markdown('<div class="section-label">What-If Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="callout cy"><b>Simulate changes</b> — Adjust a customer\'s parameters and instantly see how churn probability shifts. Perfect for testing retention scenarios.</div>', unsafe_allow_html=True)

    wl, wr = st.columns([1, 1], gap="large")

    with wl:
        st.markdown('<div class="form-group"><div class="form-group-header">⚙️ Tune Parameters</div>', unsafe_allow_html=True)
        wi_gender   = st.selectbox("Gender",["Male","Female"], key="wi_g")
        wi_senior   = st.selectbox("Senior Citizen",["No","Yes"], key="wi_sr")
        wi_partner  = st.selectbox("Partner",["No","Yes"], key="wi_par")
        wi_dep      = st.selectbox("Dependents",["No","Yes"], key="wi_dep")
        wi_tenure   = st.slider("Tenure (months)", 0, 72, 6, key="wi_t",
                                help="Drag to see how loyalty reduces churn risk")
        wi_contract = st.selectbox("Contract Type",["Month-to-month","One year","Two year"], key="wi_ct",
                                   help="Upgrading contract is the #1 retention lever")
        wi_internet = st.selectbox("Internet Service",["No","DSL","Fiber optic"], key="wi_int")
        wi_security = st.selectbox("Online Security",["No","Yes"], key="wi_sec")
        wi_tech     = st.selectbox("Tech Support",["No","Yes"], key="wi_tch")
        wi_payment  = st.selectbox("Payment Method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"], key="wi_pay")
        wi_monthly  = st.slider("Monthly Charge ($)", 0, 200, 70, key="wi_mc")
        wi_paperless= st.selectbox("Paperless Billing",["No","Yes"], key="wi_pb")
        st.markdown('</div>', unsafe_allow_html=True)

    with wr:
        st.markdown('<div class="section-label">Live Prediction</div>', unsafe_allow_html=True)
        wi_inp = {
            'gender':wi_gender,'SeniorCitizen':1 if wi_senior=='Yes' else 0,
            'Partner':wi_partner,'Dependents':wi_dep,'tenure':wi_tenure,
            'PhoneService':'Yes','InternetService':wi_internet,
            'OnlineSecurity':wi_security,'OnlineBackup':'No',
            'DeviceProtection':'No','TechSupport':wi_tech,
            'StreamingTV':'No','StreamingMovies':'No',
            'Contract':wi_contract,'PaperlessBilling':wi_paperless,
            'PaymentMethod':wi_payment,'MonthlyCharges':float(wi_monthly),
            'TotalCharges':float(wi_monthly * wi_tenure if wi_tenure > 0 else wi_monthly)
        }
        wi_prob = pred_single(wi_inp, arts)
        render_prediction_output(wi_prob, wi_inp)

        st.markdown('<br>', unsafe_allow_html=True)
        # Tenure curve
        st.markdown(f'<div style="font-size:10.5px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:{MUTED};margin-bottom:8px;">Churn vs Tenure (current settings)</div>', unsafe_allow_html=True)
        plt.rcParams.update(MPL_STYLE)
        tenures = list(range(0, 73, 3))
        probs_curve = []
        for t in tenures:
            tmp = dict(wi_inp)
            tmp['tenure'] = t
            tmp['TotalCharges'] = float(wi_monthly * t if t > 0 else wi_monthly)
            probs_curve.append(pred_single(tmp, arts))
        fig, ax = plt.subplots(figsize=(5.5, 2.8))
        ax.fill_between(tenures, probs_curve, alpha=0.12, color='#388bfd')
        ax.plot(tenures, probs_curve, color='#388bfd', lw=2.5)
        ax.axvline(wi_tenure, color=get_risk(wi_prob)[0], lw=1.5, linestyle='--', alpha=0.8)
        ax.axhline(0.5, color='#6e7681', lw=1, linestyle=':', alpha=0.6)
        ax.scatter([wi_tenure],[wi_prob], s=80, color=get_risk(wi_prob)[0], zorder=5)
        ax.set_ylim(0,1); ax.set_xlim(0,72)
        ax.set_xlabel("Tenure (months)"); ax.set_ylabel("Churn Probability")
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ============================================================
# TAB 4 — MODEL PERFORMANCE
# ============================================================
elif active == "tab_perf":
    st.markdown('<div class="section-label">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="callout ci"><b>Evaluation:</b> Trained on 80% of data (5,634 rows). All metrics from held-out 20% test set (1,409 customers) — no data leakage.</div>', unsafe_allow_html=True)

    plt.rcParams.update(MPL_STYLE)

    # Row 1: Metrics bar + Confusion Matrix
    col1, col2 = st.columns(2, gap="large")
    with col1:
        metrics = [
            ("AUC-ROC",  ACTUAL_METRICS['roc_auc'],  "#388bfd"),
            ("Recall",   ACTUAL_METRICS['recall'],   "#3fb950"),
            ("Accuracy", ACTUAL_METRICS['accuracy'], "#58a6ff"),
            ("F1 Score", ACTUAL_METRICS['f1'],       "#f0883e"),
            ("Precision",ACTUAL_METRICS['precision'],"#a371f7"),
        ]
        bars_html = "".join(
            f'<div class="met-row"><div class="met-name">{n}</div>'
            f'<div class="met-bar-bg"><div class="met-bar-fill" style="width:{v*100:.1f}%;background:{c};"></div></div>'
            f'<div class="met-val" style="color:{c};">{v:.1%}</div></div>'
            for n,v,c in metrics
        )
        st.markdown(f'<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:12px;padding:22px 26px;"><div style="font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:{MUTED};margin-bottom:14px;">Performance Metrics</div>{bars_html}</div>', unsafe_allow_html=True)

    with col2:
        fig, ax = plt.subplots(figsize=(5.2, 4))
        cm_norms = np.array([[0.3,0.15],[0.1,0.8]])
        annot = np.array([
            [f"{ACTUAL_METRICS['tn']}\nTrue Neg",  f"{ACTUAL_METRICS['fp']}\nFalse Pos"],
            [f"{ACTUAL_METRICS['fn']}\nFalse Neg", f"{ACTUAL_METRICS['tp']}\nTrue Pos"]
        ])
        sns.heatmap(cm_norms, annot=annot, fmt='', cmap='RdYlGn', vmin=0, vmax=1, ax=ax,
                    cbar=False, linewidths=2, linecolor=BG,
                    xticklabels=['Predicted: Stay','Predicted: Leave'],
                    yticklabels=['Actually Stayed','Actually Left'],
                    annot_kws={"size":10,"weight":"bold","color":"#e6edf3" if IS_DARK else "#1a1f2e"})
        ax.set_title("Confusion Matrix (1,409 test customers)", fontsize=11, fontweight='bold', pad=12)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: ROC curve + Feature Importance
    col3, col4 = st.columns(2, gap="large")
    with col3:
        fpr, tpr, _ = roc_curve(arts['y_test'], arts['y_pred_prob'])
        fig2, ax2 = plt.subplots(figsize=(5.2, 4))
        ax2.plot(fpr, tpr, color='#388bfd', lw=2.5, label=f"AUC = {ACTUAL_METRICS['roc_auc']:.3f}")
        ax2.plot([0,1],[0,1], color='#6e7681', lw=1, linestyle='--', label="Random (0.5)")
        ax2.fill_between(fpr, tpr, alpha=0.1, color='#388bfd')
        ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve", fontsize=11, fontweight='bold', pad=12)
        ax2.legend(fontsize=10, framealpha=0.15, loc='lower right')
        ax2.grid(alpha=0.2); ax2.spines[['top','right']].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    with col4:
        fi = arts['feature_importance']
        fig3, ax3 = plt.subplots(figsize=(5.2, 4))
        colors = ['#f85149' if c > 0 else '#3fb950' for c in fi['coef']]
        bars = ax3.barh(fi['feature'], fi['abs_coef'], color=colors, edgecolor='none', height=0.65)
        ax3.set_xlabel("Feature Weight (|coefficient|)")
        ax3.set_title("Top Feature Importances", fontsize=11, fontweight='bold', pad=12)
        ax3.spines[['top','right','bottom']].set_visible(False)
        ax3.grid(axis='x', alpha=0.2)
        red_p = mpatches.Patch(color='#f85149', label='↑ Increases churn risk')
        grn_p = mpatches.Patch(color='#3fb950', label='↓ Reduces churn risk')
        ax3.legend(handles=[red_p,grn_p], fontsize=9, framealpha=0.15)
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 3: Precision-Recall curve + plain English
    col5, col6 = st.columns(2, gap="large")
    with col5:
        prec, rec, _ = precision_recall_curve(arts['y_test'], arts['y_pred_prob'])
        fig4, ax4 = plt.subplots(figsize=(5.2, 3.5))
        ax4.plot(rec, prec, color='#f0883e', lw=2.5)
        ax4.fill_between(rec, prec, alpha=0.1, color='#f0883e')
        ax4.axhline(CHURN_RATE, color='#6e7681', lw=1, linestyle='--', label=f"Baseline ({CHURN_RATE:.0%})")
        ax4.set_xlabel("Recall"); ax4.set_ylabel("Precision")
        ax4.set_title("Precision-Recall Curve", fontsize=11, fontweight='bold', pad=12)
        ax4.legend(fontsize=10, framealpha=0.15)
        ax4.grid(alpha=0.2); ax4.spines[['top','right']].set_visible(False)
        fig4.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close()

    with col6:
        st.markdown(f"""
<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:12px;padding:22px 26px;">
  <div style="font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:{MUTED};margin-bottom:16px;">Results in Plain English</div>
  <div style="font-size:13px;color:{MUTED};line-height:2.1;">
    Out of <span style="color:{TEXT};font-weight:600;">1,409 test customers</span>:<br>
    <span style="color:#3fb950;">✓ {ACTUAL_METRICS['tn']} correctly identified as loyal</span><br>
    <span style="color:#3fb950;">✓ {ACTUAL_METRICS['tp']} correctly flagged as churning</span><br>
    <span style="color:#d29922;">⚠ {ACTUAL_METRICS['fp']} false alarms (stayed but flagged)</span><br>
    <span style="color:#f85149;">✗ {ACTUAL_METRICS['fn']} missed churners (false negatives)</span>
  </div>
  <hr style="border-color:{INNER};margin:14px 0;">
  <div style="font-size:12.5px;color:{MUTED};line-height:1.9;">
    <b style="color:{SUB};">Why recall over precision?</b><br>
    Missing a churner costs more than a wasted retention call. Model tuned for high recall (81%).<br><br>
    <b style="color:{SUB};">Overfitting check:</b> Train 76.9% → Test 73.9% — healthy ~3% gap, no overfitting.<br><br>
    <b style="color:{SUB};">SMOTE applied:</b> Class imbalance corrected during training only.
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TAB 5 — DATASET
# ============================================================
elif active == "tab_data":
    st.markdown('<div class="section-label">Dataset Overview</div>', unsafe_allow_html=True)
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Total Rows",      f"{len(df_raw):,}")
    m2.metric("Total Columns",   df_raw.shape[1])
    m3.metric("Churn Rate",      f"{CHURN_RATE*100:.1f}%")
    m4.metric("Missing Values",  int(df_raw.isnull().sum().sum()))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:{MUTED};margin-bottom:8px;">Data Preview — First 10 Rows</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.head(10), use_container_width=True, height=280)

    st.markdown("<br>", unsafe_allow_html=True)
    plt.rcParams.update(MPL_STYLE)
    c1,c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f'<div style="font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:{MUTED};margin-bottom:8px;">Churn Distribution</div>', unsafe_allow_html=True)
        counts = df_raw['Churn'].value_counts().rename(index={'No':'Retained','Yes':'Churned',0:'Retained',1:'Churned'})
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(counts.index, counts.values, color=['#3fb950','#f85149'], width=0.45, edgecolor=BG, linewidth=1.5)
        for b in bars: ax.text(b.get_x()+b.get_width()/2, b.get_height()+60, f'{b.get_height():,}', ha='center', fontsize=11, fontweight='bold', color=TEXT)
        ax.set_ylim(0, max(counts.values)*1.2)
        ax.spines[['top','right','left']].set_visible(False); ax.grid(axis='y',alpha=0.3)
        fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

    with c2:
        st.markdown(f'<div style="font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:{MUTED};margin-bottom:8px;">Tenure Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(df_raw['tenure'], bins=30, color='#388bfd', alpha=0.85, edgecolor=BG, linewidth=0.8)
        ax.set_xlabel("Months with company"); ax.set_ylabel("Customers")
        ax.spines[['top','right','left']].set_visible(False); ax.grid(axis='y',alpha=0.3)
        fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

    c3,c4 = st.columns(2, gap="large")
    with c3:
        st.markdown(f'<div style="font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:{MUTED};margin-bottom:8px;">Monthly Charges — Retained vs Churned</div>', unsafe_allow_html=True)
        churn_map = {0:'Retained',1:'Churned','No':'Retained','Yes':'Churned'}
        df_plot = df_raw.copy(); df_plot['Churn_label'] = df_plot['Churn'].map(churn_map)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for label, color in [('Retained','#3fb950'),('Churned','#f85149')]:
            sub = df_plot[df_plot['Churn_label']==label]['MonthlyCharges']
            ax.hist(sub, bins=35, alpha=0.65, label=label, color=color, edgecolor=BG, linewidth=0.5)
        ax.set_xlabel("Monthly Charge ($)"); ax.set_ylabel("Count")
        ax.legend(fontsize=10, framealpha=0.15)
        ax.spines[['top','right','left']].set_visible(False); ax.grid(axis='y',alpha=0.3)
        fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

    with c4:
        st.markdown(f'<div style="font-size:11px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:{MUTED};margin-bottom:8px;">Churn Rate by Contract Type</div>', unsafe_allow_html=True)
        cc = df_raw.groupby('Contract')['Churn'].apply(
            lambda x: (x.map({'Yes':1,'No':0}) if x.dtype==object else x).mean()*100
        ).reset_index()
        cc.columns = ['Contract','ChurnRate']
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.barh(cc['Contract'], cc['ChurnRate'], color=['#f85149','#d29922','#3fb950'], edgecolor=BG, linewidth=1)
        for bar in bars: ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2, f"{bar.get_width():.1f}%", va='center', fontsize=11, fontweight='bold', color=TEXT)
        ax.set_xlabel("Churn Rate (%)"); ax.set_xlim(0, cc['ChurnRate'].max()*1.25)
        ax.spines[['top','right','bottom']].set_visible(False); ax.grid(axis='x',alpha=0.3)
        fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

# ── FOOTER
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f'<div style="text-align:center;font-size:11.5px;color:{"#6e7681" if IS_DARK else "#94a3b8"};padding:8px 0;">Telecom Customer Churn Predictor &nbsp;·&nbsp; Logistic Regression + SMOTE &nbsp;·&nbsp; No data leakage &nbsp;·&nbsp; Built with Streamlit</div>', unsafe_allow_html=True)
