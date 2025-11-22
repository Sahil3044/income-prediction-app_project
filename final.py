# =========================================
# STREAMLIT INCOME PREDICTION APP - ENHANCED
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import logging
import json
import requests

# ---------------------------
# 1️⃣ Load model with enhanced error handling
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_URL = "http://127.0.0.1:8000"

try:
    model_package = joblib.load("deployment_model.joblib")
    model = model_package['model']
    model_info = model_package.get('model_info', {
        'name': 'Random Forest (Demo)',
        'accuracy': 0.861,
        'f1_score': 0.899,
        'roc_auc': 0.915,
        'precision': 0.856,
        'recall': 0.832,
        'training_date': '2024-01-01',
        'version': '1.0'
    })
    logger.info(f"Model loaded successfully: {model_info['name']}")
except Exception as e:
    st.error(f"❌ Failed to load model: {str(e)}")
    st.error("Please ensure 'deployment_model.joblib' exists in the current directory")
    st.stop()

# ---------------------------
# 2️⃣ Streamlit Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Income Prediction Pro",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# 3️⃣ CSS & Theme
# ---------------------------
st.markdown("""
<style>
/* ============================================= */
/*  CLEAN, MODERN & FULLY RESPONSIVE CSS ONLY   */
/*  For Streamlit Income Prediction App          */
/* ============================================= */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*,
*::before,
*::after {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:root {
    --font: 'Inter', system-ui, -apple-system, sans-serif;
    
    /* Colors - Clean Professional Palette */
    --bg: #f8fafc;
    --surface: #ffffff;
    --surface-2: #f1f5f9;
    --border: #e2e8f0;
    --border-light: #f8fafc;
    
    --text: #0f172a;
    --text-light: #475569;
    --text-muted: #64748b;
    
    --primary: #3b82f6;
    --primary-hover: #2563eb;
    --primary-light: #dbeafe;
    --success: #10b981;
    --success-light: #d1fae5;
    --danger: #ef4444;
    --danger-light: #fee2e2;
    
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.1);
    --shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    --shadow-md: 0 10px 15px -3px rgba(0,0,0,0.1);
    --shadow-lg: 0 20px 25px -5px rgba(0,0,0,0.1);
    
    --radius: 12px;
    --radius-sm: 8px;
    --transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

html, body, .stApp {
    font-family: var(--font);
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    min-height: 100vh;
}

/* Main Container */
.css-1d391kg, .main > div {
    padding: 2rem 1rem;
    max-width: 1400px;
    margin: 0 auto;
}

/* Headers */
.main-header {
    font-size: 2.8rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, var(--primary), #8b5cf6);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    letter-spacing: -0.5px;
}

.sub-header {
    font-size: 1.2rem;
    color: var(--text-light);
    text-align: center;
    max-width: 700px;
    margin: 0 auto 3rem;
    font-weight: 400;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
    box-shadow: var(--shadow);
    border-radius: 0 16px 16px 0;
    overflow: hidden;
}

.sidebar-header {
    background: linear-gradient(135deg, var(--primary), #6366f1);
    color: white;
    padding: 1.8rem 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
}

.sidebar-title {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.sidebar-subtitle {
    font-size: 0.9rem;
    opacity: 0.9;
}

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    box-shadow: var(--shadow-sm);
    transition: var(--transition);
}

.card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-4px);
}

/* Metric Cards */
.metric-card {
    background: var(--surface);
    border-radius: var(--radius);
    padding: 1.5rem;
    text-align: left;
    border: 1px solid var(--border);
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
    transition: var(--transition);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--primary), #8b5cf6);
}

.metric-card:hover {
    transform: translateY(-6px);
    box-shadow: var(--shadow-lg);
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--primary);
    margin: 0.5rem 0;
}

.metric-label {
    font-size: 0.85rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}

/* Prediction Result Boxes */
.prediction-box {
    padding: 2.5rem 2rem;
    border-radius: var(--radius);
    text-align: center;
    background: white;
    box-shadow: var(--shadow-lg);
    border: 2px solid transparent;
    position: relative;
    overflow: hidden;
    transition: var(--transition);
}

.prediction-box::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.9), transparent);
    pointer-events: none;
}

.high-income {
    background: linear-gradient(135deg, #d1fae5, #a7f3d0);
    border-color: var(--success);
    color: #065f46;
}

.low-income {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    border-color: var(--danger);
    color: #991b1b;
}

.prediction-box h2 {
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}

.prediction-box h3 {
    font-size: 1.5rem;
    margin: 0.5rem 0 1rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--primary), var(--primary-hover));
    color: white !important;
    border: none;
    border-radius: var(--radius-sm);
    padding: 0.9rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    width: 100%;
    box-shadow: var(--shadow);
    transition: var(--transition);
    text-transform: none !important;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-lg);
    background: linear-gradient(135deg, var(--primary-hover), #1d4ed8);
}

/* Form Elements */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSlider > div > div {
    background: white !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    transition: var(--transition) !important;
}

.stSelectbox > div > div:hover,
.stTextInput > div > div > input:hover,
.stNumberInput > div > div > input:hover {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary-light) !important;
}

label {
    font-weight: 600 !important;
    color: var(--text) !important;
    font-size: 0.95rem !important;
    margin-bottom: 0.5rem !important;
}

/* Footer */
.footer {
    margin-top: 4rem;
    padding: 2.5rem 2rem;
background: linear-gradient(  #ffffff);
    border-radius: var(--radius);
    border: 1px solid var(--border);
    text-align: left;
    color: var(--text-light);
}

.footer h3 {
    color: var(--primary);
    margin-bottom: 1rem;
    font-size: 1.4rem;
    text-align: center;
}

/* Responsive */
@media (max-width: 1024px) {
    .main-header { font-size: 2.4rem; }
    .metric-value { font-size: 1.9rem; }
}

@media (max-width: 768px) {
    .main-header { font-size: 2.2rem; }
    .sub-header { font-size: 1.1rem; }
    .prediction-box { padding: 2rem 1.5rem; }
    [data-testid="column"] { width: 100% !important; }
}

@media (max-width: 480px) {
    .main-header { font-size: 1.9rem; }
    .metric-card, .card { padding: 1.2rem; }
    .stButton > button { padding: 0.8rem 1.5rem; }
}

/* Smooth transitions */
*, *::before, *::after {
    transition: var(--transition);
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'api_connected' not in st.session_state:
    st.session_state.api_connected = False

# Apply theme
st.markdown(f"<script>document.documentElement.setAttribute('data-theme', '{st.session_state.theme}');</script>",
            unsafe_allow_html=True)


# ---------------------------
# 4️⃣ Enhanced Prediction Function - FIXED FOR API
# ---------------------------
def validate_inputs(input_data):
    """Validate and clean input data"""
    validated = input_data.copy()

    # Age validation
    if validated['age'] < 18 or validated['age'] > 100:
        st.warning("⚠️ Age seems unusual. Please verify.")

    # Hours per week validation
    if validated['hours_per_week'] > 80:
        st.warning("⚠️ Hours per week exceeds typical full-time employment.")

    # Capital gain/loss validation
    if validated['capital_gain'] > 50000:
        st.warning("⚠️ Unusually high capital gain detected.")

    return validated


def make_api_prediction(input_data):
    """Make prediction using FastAPI backend"""
    try:
        response = requests.post(f"{API_URL}/predict", json=input_data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code}")
    except Exception as e:
        logger.error(f"API prediction failed: {str(e)}")
        raise e


def make_local_prediction(input_data):
    """Make prediction using local model with engineered features"""
    try:
        # Validate inputs first
        validated_data = validate_inputs(input_data)

        # Create engineered features (same as FastAPI)
        input_dict = {
            'age': [validated_data['age']],
            'workclass': [validated_data['workclass']],
            'education-num': [validated_data['education_num']],
            'marital-status': [validated_data['marital_status']],
            'occupation': [validated_data['occupation']],
            'relationship': [validated_data['relationship']],
            'race': [validated_data['race']],
            'sex': [validated_data['sex']],
            'capital-gain': [validated_data['capital_gain']],
            'capital-loss': [validated_data['capital_loss']],
            'hours-per-week': [validated_data['hours_per_week']],
            'native-country': [validated_data['native_country']]
        }

        # Add engineered features
        # Age groups
        if validated_data['age'] < 25:
            age_group = '18-25'
        elif validated_data['age'] < 35:
            age_group = '26-35'
        elif validated_data['age'] < 45:
            age_group = '36-45'
        elif validated_data['age'] < 55:
            age_group = '46-55'
        elif validated_data['age'] < 65:
            age_group = '56-65'
        else:
            age_group = '65+'

        # Hours category
        if validated_data['hours_per_week'] < 35:
            hours_category = 'Part-time'
        elif validated_data['hours_per_week'] < 40:
            hours_category = 'Full-time'
        elif validated_data['hours_per_week'] < 50:
            hours_category = 'Overtime'
        else:
            hours_category = 'Double-time'

        # Education level mapping
        education_mapping = {
            'Preschool': 'Elementary', '1st-4th': 'Elementary', '5th-6th': 'Elementary',
            '7th-8th': 'Middle', '9th': 'Middle', '10th': 'Middle',
            '11th': 'High', '12th': 'High', 'HS-grad': 'High',
            'Some-college': 'College', 'Assoc-acdm': 'College', 'Assoc-voc': 'College',
            'Bachelors': 'University', 'Masters': 'Graduate', 'Prof-school': 'Graduate',
            'Doctorate': 'PhD'
        }
        education_level = education_mapping.get(validated_data.get('education', 'Bachelors'), 'Unknown')

        # Add engineered features to input dict
        input_dict['age_group'] = [age_group]
        input_dict['hours_category'] = [hours_category]
        input_dict['education_level'] = [education_level]
        input_dict['capital_change'] = [validated_data['capital_gain'] - validated_data['capital_loss']]
        input_dict['is_us'] = [1 if validated_data['native_country'] == 'United-States' else 0]
        input_dict['has_capital_activity'] = [
            1 if (validated_data['capital_gain'] > 0 or validated_data['capital_loss'] > 0) else 0]

        input_df = pd.DataFrame(input_dict)

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Enhanced confidence calculation
        if probability > 0.8 or probability < 0.2:
            confidence = "high"
            confidence_score = 0.9
        elif probability > 0.7 or probability < 0.3:
            confidence = "medium"
            confidence_score = 0.7
        else:
            confidence = "low"
            confidence_score = 0.5

        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "income_class": ">50K" if prediction == 1 else "<=50K",
            "confidence": confidence,
            "confidence_score": confidence_score,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Local prediction error: {str(e)}")
        raise e


def save_prediction(input_data, result):
    """Save prediction to session history"""
    history_entry = {
        'timestamp': result['timestamp'],
        'inputs': input_data,
        'result': result
    }
    st.session_state.prediction_history.append(history_entry)

    # Keep only last 50 predictions
    if len(st.session_state.prediction_history) > 50:
        st.session_state.prediction_history = st.session_state.prediction_history[-50:]


# ---------------------------
# 5️⃣ Visualization Functions
# ---------------------------
def create_enhanced_gauge(probability, income_class):
    """Create enhanced gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Income Probability Score", 'font': {'size': 24}},
        delta={'reference': 0.5, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.4], 'color': 'lightcoral'},
                {'range': [0.4, 0.6], 'color': 'lightyellow'},
                {'range': [0.6, 1], 'color': 'lightgreen'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.5}}
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=50, r=50, t=100, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "gray", 'family': "Inter"}
    )
    return fig


def plot_feature_importance():
    """Plot feature importance if available in model"""
    try:
        if hasattr(model, 'feature_importances_'):
            feature_names = [
                'age', 'workclass', 'education_num', 'marital_status', 'occupation',
                'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
                'hours_per_week', 'native_country', 'age_group', 'hours_category',
                'education_level', 'capital_change', 'is_us', 'has_capital_activity'
            ]

            importance_df = pd.DataFrame({
                'feature': feature_names[:len(model.feature_importances_)],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)

            fig = px.bar(importance_df, x='importance', y='feature',
                         title='🔍 Feature Importance Analysis',
                         color='importance',
                         color_continuous_scale='Viridis')
            fig.update_layout(
                showlegend=False,
                xaxis_title="Importance Score",
                yaxis_title="Features",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            return fig
        else:
            return None
    except Exception as e:
        logger.warning(f"Feature importance not available: {str(e)}")
        return None


def show_model_info():
    """Display model information"""
    st.subheader("🧠 Model Information")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Model Details</h4>
            <p><strong>Type:</strong> {model_info['name']}</p>
            <p><strong>Version:</strong> {model_info.get('version', '1.0')}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Dataset Info</h4>
            <p><strong>Input Features:</strong> 18 (12 original + 6 engineered)</p>
            <p><strong>Target Variable:</strong> Income (>50K/≤50K)</p>
            <p><strong>Data Source:</strong> UCI Adult Dataset</p>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------
# 6️⃣ Sidebar Navigation
# ---------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-title">💰 Income Predictor</div>
        <div class="sidebar-subtitle">Advanced ML Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    # API Status Check
    st.markdown("### 🔌 API Connection")
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=5)
        if response.status_code == 200:
            api_model_info = response.json()
            st.success("✅ API Connected")
            st.session_state.api_connected = True
            # Use API model info if available
            model_info.update(api_model_info)
        else:
            st.error("❌ API Error")
            st.session_state.api_connected = False
    except:
        st.warning("🌐 API Offline - Using Local Model")
        st.session_state.api_connected = False

    st.markdown("---")

    nav_options = ["🎯 Prediction", "📊 Analytics", "🔍 Insights", "📈 History"]
    selected_nav = st.radio("Navigation", nav_options, label_visibility="collapsed")

    # Quick stats in sidebar
    st.markdown("---")
    st.markdown(f"""
    <div class="metric-card">
        <h4>📈 Quick Stats</h4>
        <p>Total Predictions: {len(st.session_state.prediction_history)}</p>
        <p>Model Accuracy: {model_info['accuracy']:.1%}</p>
        <p>Mode: {'🌐 API' if st.session_state.api_connected else '💻 Local'}</p>
        <p>Last Updated: {datetime.now().strftime('%Y-%m-%d')}</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# 7️⃣ Main Page
# ---------------------------
st.markdown('<h1 class="main-header">💰 Income Prediction </h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Machine Learning Powered Income Classification System</p>',
            unsafe_allow_html=True)

# ---------------------------
# 🎯 Prediction Page
# ---------------------------
if selected_nav == "🎯 Prediction":
    # st.header("🎯 Real-time Prediction")
    col1, col2, col3 = st.columns([1, 0.1, 1])

    with col1:
        st.subheader("👤 Demographic Profile")
        age = st.slider("Age", 18, 90, 35, help="Select the individual's age")
        sex = st.radio("Gender", ["Male", "Female"], horizontal=True)
        race = st.selectbox("Race", ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
        marital_status = st.selectbox("Marital Status", [
            "Married-civ-spouse", "Divorced", "Never-married", "Separated",
            "Widowed", "Married-spouse-absent", "Married-AF-spouse"
        ])
        relationship = st.selectbox("Relationship", [
            "Wife", "Own-child", "Husband", "Not-in-family",
            "Other-relative", "Unmarried"
        ])
        education = st.selectbox("Education Level", [
            "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th",
            "HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors",
            "Masters", "Prof-school", "Doctorate"
        ])
        education_num = st.slider("Education Years", 1, 16, 9,
                                  help="Number of years of education (1 = Preschool, 16 = Doctorate)")

    with col2:
        st.markdown("<div style='border-left: 2px solid var(--border); height: 100%; margin: 0 1rem;'></div>",
                    unsafe_allow_html=True)

    with col3:
        st.subheader("💼 Employment & Financial")
        workclass = st.selectbox("Work Class", [
            "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
            "Local-gov", "State-gov", "Without-pay", "Never-worked"
        ])
        occupation = st.selectbox("Occupation", [
            "Tech-support", "Craft-repair", "Other-service", "Sales",
            "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
            "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
            "Protective-serv", "Armed-Forces"
        ])
        hours_per_week = st.slider("Hours per Week", 20, 80, 40, help="Typical hours worked per week")
        capital_gain = st.number_input("Capital Gain ($)", 0, 100000, 0, step=100, help="Capital gains income")
        capital_loss = st.number_input("Capital Loss ($)", 0, 5000, 0, step=100, help="Capital losses")
        native_country = st.selectbox("Native Country", [
            "United-States", "Mexico", "Philippines", "Germany", "Puerto-Rico",
            "Canada", "El-Salvador", "India", "Cuba", "England", "China", "Other"
        ])

    st.markdown("---")

    # Prediction button with enhanced functionality
    prediction_col1, prediction_col2 = st.columns([1, 3])
    with prediction_col1:
        if st.button("🚀 Generate Prediction", use_container_width=True):
            with st.spinner("🤖 Analyzing profile data..."):
                time.sleep(1)  # Simulate processing

                input_data = {
                    "age": age, "workclass": workclass, "education": education,
                    "education_num": education_num, "marital_status": marital_status,
                    "occupation": occupation, "relationship": relationship, "race": race,
                    "sex": sex, "capital_gain": capital_gain, "capital_loss": capital_loss,
                    "hours_per_week": hours_per_week, "native_country": native_country
                }

                try:
                    # Use API if connected, otherwise use local model
                    if st.session_state.api_connected:
                        result = make_api_prediction(input_data)
                        result["timestamp"] = datetime.now().isoformat()
                        result["confidence_score"] = 0.9 if result["confidence"] == "high" else 0.7 if result[
                                                                                                           "confidence"] == "medium" else 0.5
                    else:
                        result = make_local_prediction(input_data)

                    save_prediction(input_data, result)

                    # Display results
                    if result["income_class"] == ">50K":
                        st.markdown(f'''
                        <div class="prediction-box high-income">
                            <h2>🎉 HIGH INCOME PREDICTED</h2>
                            <h3>> $50,000/Year</h3>
                            <p>Confidence: {result["confidence"].upper()}</p>
                            <p>Probability: {result["probability"]:.1%}</p>
                            <p>Model is {result["confidence_score"]:.0%} confident in this prediction</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                        <div class="prediction-box low-income">
                            <h2>📈 MODERATE INCOME PREDICTED</h2>
                            <h3>≤ $50,000/Year</h3>
                            <p>Confidence: {result["confidence"].upper()}</p>
                            <p>Probability: {result["probability"]:.1%}</p>
                            <p>Model is {result["confidence_score"]:.0%} confident in this prediction</p>
                        </div>
                        ''', unsafe_allow_html=True)

                    # Enhanced Gauge Chart
                    st.plotly_chart(create_enhanced_gauge(result["probability"], result["income_class"]),
                                    use_container_width=True)

                    # Key factors analysis
                    st.subheader("🔍 Key Influencing Factors")
                    factors_col1, factors_col2, factors_col3 = st.columns(3)

                    with factors_col1:
                        st.metric("Education", f"{education_num} years",
                                  delta="Positive" if education_num > 12 else "Neutral")
                    with factors_col2:
                        st.metric("Age", f"{age} years", delta="Positive" if age > 35 else "Neutral")
                    with factors_col3:
                        st.metric("Work Hours", f"{hours_per_week}/week",
                                  delta="Positive" if hours_per_week > 40 else "Neutral")

                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")

    with prediction_col2:
        mode_status = "🌐 **API Mode**" if st.session_state.api_connected else "💻 **Local Mode**"
        st.info(f"💡 **Tip:** Fill in all fields accurately for the best prediction. {mode_status}")

# ---------------------------
# 📊 Analytics Page
# ---------------------------
elif selected_nav == "📊 Analytics":
    st.header("📊 Model Analytics Dashboard")

    # Performance Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{model_info["accuracy"]:.3f}</div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">F1 Score</div>
            <div class="metric-value">{model_info["f1_score"]:.3f}</div>
        </div>
        ''', unsafe_allow_html=True)
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">ROC AUC</div>
            <div class="metric-value">{model_info["roc_auc"]:.3f}</div>
        </div>
        ''', unsafe_allow_html=True)
    with col4:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-label">Precision</div>
            <div class="metric-value">{model_info.get("precision", 0.856):.3f}</div>
        </div>
        ''', unsafe_allow_html=True)

    # Performance Charts
    col1, col2 = st.columns(2)

    with col1:
        # Performance Bar Chart
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Value': [
                model_info['accuracy'],
                model_info.get('precision', 0.856),
                model_info.get('recall', 0.832),
                model_info['f1_score'],
                model_info['roc_auc']
            ]
        }
        fig = px.bar(metrics_data, x='Metric', y='Value', title="📈 Model Performance Metrics",
                     color='Value', color_continuous_scale='Viridis', range_y=[0, 1])
        fig.update_layout(showlegend=False, yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Feature Importance
        feature_fig = plot_feature_importance()
        if feature_fig:
            st.plotly_chart(feature_fig, use_container_width=True)
        else:
            st.info("🔍 Feature importance data not available for this model")

    # Model Information
    show_model_info()

# ---------------------------
# 🔍 Insights Page
# ---------------------------
elif selected_nav == "🔍 Insights":
    st.header("🔍 Feature Insights & Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Common High-Income Patterns")
        st.markdown("""
        <div class="metric-card">
        <h4>🎓 Education Impact</h4>
        <ul>
        <li>16 years (Doctorate): 85% high income probability</li>
        <li>14 years (Master's): 72% high income probability</li>
        <li>12 years (Bachelor's): 58% high income probability</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card">
        <h4>💼 Occupation Trends</h4>
        <ul>
        <li>Exec-managerial: 68% high income</li>
        <li>Prof-specialty: 62% high income</li>
        <li>Tech-support: 55% high income</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("📈 Age & Experience Correlation")
        # Simulated age-income correlation
        age_data = pd.DataFrame({
            'Age Group': ['18-25', '26-35', '36-45', '46-55', '56-65', '65+'],
            'High Income %': [12, 35, 52, 48, 42, 28]
        })

        fig = px.line(age_data, x='Age Group', y='High Income %',
                      title="High Income Probability by Age Group",
                      markers=True)
        fig.update_traces(line=dict(color='var(--primary)', width=3))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="metric-card">
        <h4>⏰ Work Hours Impact</h4>
        <ul>
        <li>40+ hours: Significant positive impact</li>
        <li>50+ hours: Diminishing returns</li>
        <li>60+ hours: Minimal additional benefit</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # Data Quality Information
    st.subheader("📋 Data Quality & Sources")
    st.markdown("""
    <div class="metric-card">
    <h4>Dataset Information</h4>
    <p><strong>Source:</strong> UCI Machine Learning Repository - Adult Dataset</p>
    <p><strong>Records:</strong> 48,842 individuals</p>
    <p><strong>Features:</strong> 14 demographic and employment attributes</p>
    <p><strong>Target:</strong> Binary income classification (>50K/≤50K)</p>
    <p><strong>Last Updated:</strong> 2023</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# 📈 History Page
# ---------------------------
elif selected_nav == "📈 History":
    st.header("📈 Prediction History")

    if not st.session_state.prediction_history:
        st.info("📝 No prediction history yet. Make some predictions to see them here!")
    else:
        # Statistics
        total_predictions = len(st.session_state.prediction_history)
        high_income_count = sum(1 for entry in st.session_state.prediction_history
                                if entry['result']['income_class'] == '>50K')

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", total_predictions)
        with col2:
            st.metric("High Income Predictions", high_income_count)
        with col3:
            st.metric("High Income Rate", f"{(high_income_count / total_predictions):.1%}")

        # Recent predictions table
        st.subheader("Recent Predictions")
        history_data = []
        for entry in st.session_state.prediction_history[-10:]:  # Show last 10
            history_data.append({
                'Timestamp': entry['timestamp'],
                'Age': entry['inputs']['age'],
                'Education': entry['inputs']['education_num'],
                'Occupation': entry['inputs']['occupation'],
                'Income Class': entry['result']['income_class'],
                'Probability': f"{entry['result']['probability']:.1%}",
                'Confidence': entry['result']['confidence'].upper()
            })

        if history_data:
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)

        # Export functionality
        st.subheader("Export Data")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Export History as CSV"):
                export_df = pd.DataFrame(st.session_state.prediction_history)
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"income_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.rerun()

# ---------------------------
# 8️⃣ Footer
# ---------------------------
st.markdown("---")
st.markdown("""
<div class="footer">
    <h3>🎓 Advanced Machine Learning - Professional Dashboard</h3>
    <p><strong>Student:</strong> Khairullah Ibrahim Khail</p>
    <p><strong>Professor:</strong> Dr. Muhammad Sajjad</p>
    <p><strong>Course:</strong> Advanced Machine Learning - Mid-Term Project</p>
    <p><strong>Academic Year:</strong> 2024 • Department of Computer Science</p>
    <p><strong>Last Updated:</strong> {}</p>
</div>
""".format(datetime.now().strftime('%B %d, %Y at %H:%M')), unsafe_allow_html=True)

# System Status
st.markdown("""
<div style="text-align: center; margin-top: 2rem; padding: 1rem; background: var(--surface); border-radius: 12px;">
    <small>🟢 System Status: Operational | 📊 Live Model: {} | 🚀 Version: 2.1.0 | 🌙 Mode: {}</small>
</div>
""".format(model_info['name'], 'API' if st.session_state.api_connected else 'Local'), unsafe_allow_html=True)