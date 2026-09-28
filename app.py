import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS for Dark UI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Modeling",
    page_icon="💳",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #0c0f16;
        color: #e0e0e0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e222d 0%, #151922 100%);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        border: 1px solid #2a2e38;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-val {
        font-size: 30px;
        font-weight: 800;
    }
    .pos-val { color: #2ecc71; }
    .neg-val { color: #e74c3c; }
    .tot-val { color: #3498db; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Synthetic Dataset Generator
# -----------------------------------------------------------------------------
@st.cache_data
def load_credit_data():
    np.random.seed(42)
    n_samples = 1000

    income = np.random.normal(60000, 20000, n_samples).clip(20000, 150000)
    loan_amount = np.random.normal(15000, 8000, n_samples).clip(2000, 40000)
    credit_score = np.random.normal(680, 50, n_samples).clip(500, 850)
    dti = np.random.uniform(5, 40, n_samples)
    delinquencies = np.random.choice([0, 1, 2], size=n_samples, p=[0.8, 0.15, 0.05])

    default_prob = (
        (800 - credit_score) / 1000 + 
        (dti / 100) + 
        (delinquencies * 0.2) - 
        (income / 300000)
    )
    default_prob = np.clip(default_prob, 0.05, 0.8)
    loan_status = np.random.binomial(1, default_prob)

    df = pd.DataFrame({
        'Annual_Income': np.round(income, 2),
        'Loan_Amount': np.round(loan_amount, 2),
        'Credit_Score': np.round(credit_score),
        'Debt_to_Income_Ratio': np.round(dti, 2),
        'Past_Delinquencies': delinquencies,
        'Loan_Default': loan_status
    })
    return df

# -----------------------------------------------------------------------------
# 3. Model Training Pipeline (Corrected train_test_split and random_state)
# -----------------------------------------------------------------------------
@st.cache_resource
def train_credit_model(df, apply_smote=True):
    X = df.drop(columns=['Loan_Default'])
    y = df['Loan_Default']

    # Correction made here: train_test_split and random_state
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)

    if apply_smote:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    auc_score = roc_auc_score(y_test, y_prob)
    return model, X_test, y_test, y_pred, auc_score, X.columns

# -----------------------------------------------------------------------------
# 4. Header & Dashboard UI
# -----------------------------------------------------------------------------
st.title("💳 Credit Risk Modeling & Assessment")
st.caption("Machine Learning model to assess loan applicant creditworthiness and predict default risk.")
st.markdown("---")

df = load_credit_data()

st.sidebar.title("⚙️ Model Controls")
mode = st.sidebar.radio("Navigation:", ["📊 Dataset & Model Evaluation", "🎯 Individual Applicant Risk Predictor"])

use_smote = st.sidebar.checkbox("Apply SMOTE (Imbalanced Data Handling)", value=True)
model, X_test, y_test, y_pred, auc_score, feature_names = train_credit_model(df, apply_smote=use_smote)

# -----------------------------------------------------------------------------
# Page 1: Dataset & Model Performance
# -----------------------------------------------------------------------------
if mode == "📊 Dataset & Model Evaluation":
    st.subheader("📌 Model Performance & Imbalance Metrics")

    total_apps = len(df)
    defaults = df['Loan_Default'].sum()
    default_rate = (defaults / total_apps) * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div>Total Applicants</div><div class="metric-val tot-val">{total_apps}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div>Default Cases</div><div class="metric-val neg-val">{defaults}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div>Default Rate</div><div class="metric-val neg-val">{default_rate:.1f}%</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div>ROC-AUC Score</div><div class="metric-val pos-val">{auc_score:.2f}</div></div>', unsafe_allow_html=True)

    st.write("")
    st.write("")

    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        st.subheader("📋 Credit Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

    with right_col:
        st.subheader("🔍 Feature Importance (Explainability)")
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=True)

        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor('#0c0f16')
        ax.set_facecolor('#0c0f16')
        ax.barh(feat_df['Feature'], feat_df['Importance'], color='#3498db')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#2a2e38')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#2a2e38')
        ax.set_xlabel("Importance Score", color="white")
        st.pyplot(fig)

# -----------------------------------------------------------------------------
# Page 2: Individual Applicant Predictor
# -----------------------------------------------------------------------------
else:
    st.subheader("🎯 Test Creditworthiness of a New Applicant")
    
    col1, col2 = st.columns(2)
    with col1:
        inc = st.number_input("Annual Income ($)", min_value=10000, max_value=200000, value=55000, step=2000)
        loan = st.number_input("Requested Loan Amount ($)", min_value=1000, max_value=50000, value=15000, step=1000)
        score = st.slider("Credit Score (CIBIL/FICO)", 300, 850, 680)
    
    with col2:
        dti = st.slider("Debt-to-Income (DTI) Ratio (%)", 0.0, 50.0, 18.5)
        delinq = st.selectbox("Past Delinquencies (Past 2 Years)", [0, 1, 2, 3])

    if st.button("🚨 Assess Credit Risk"):
        input_data = pd.DataFrame([[inc, loan, score, dti, delinq]], columns=feature_names)
        risk_prob = model.predict_proba(input_data)[0][1] * 100
        prediction = model.predict(input_data)[0]

        st.write("")
        st.subheader("📋 Assessment Results")

        if prediction == 0:
            st.success(f"✅ **LOW RISK (APPROVED)** | Risk Probability: `{risk_prob:.1f}%`")
            st.info("The applicant meets credit criteria with a strong score and healthy DTI ratio.")
        else:
            st.error(f"🚨 **HIGH RISK (DEFAULT PROBABILITY HIGH)** | Risk Probability: `{risk_prob:.1f}%`")
            st.warning("Higher default risk detected. Key risk drivers: Credit Score, Debt Ratio, or Past Delinquencies.")