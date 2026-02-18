"""
Streamlit Frontend for Cardekho V5.0 Car Price Prediction
=========================================================

Professional web interface with 93% R² accuracy!
Polished version with error-free animations and dark theme.

Author: Omar Elsaber
Date: Feb 2026
Version: 5.0 (Production Ready)
"""

import os
import streamlit as st
from streamlit_lottie import st_lottie
import requests
from datetime import datetime
from typing import Optional, Dict, Any
import time
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Lottie Animation URLs (with fallbacks)
LOTTIE_CAR_URL = "https://assets5.lottiefiles.com/packages/lf20_svy4ivvd.json"
LOTTIE_LOADING_URL = "https://assets4.lottiefiles.com/packages/lf20_jcikwtux.json"
LOTTIE_SUCCESS_URL = "https://assets3.lottiefiles.com/packages/lf20_uu0x8lqv.json"

# Page configuration with DARK THEME forced
st.set_page_config(
    page_title="Car Price AI | Cardekho Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Car Price Prediction AI - Cardekho V5.0"
    }
)

# ============================================================================
# CUSTOM CSS STYLING (DARK THEME + FIXES)
# ============================================================================

def inject_custom_css():
    """Inject production-grade custom CSS with dark theme enforcement."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* ===== FORCE DARK THEME ===== */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #FFFFFF !important;
    }
    
    /* Force dark background on main container */
    .main {
        padding-top: 1rem;
        background-color: transparent !important;
    }
    
    /* Gradient background */
    .stApp {
        background: linear-gradient(135deg, #1a1f3a 0%, #2d1b4e 100%);
        background-attachment: fixed;
    }
    
    /* Main content container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(30, 30, 50, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
    }
    
    /* ===== BUTTON STYLING ===== */
    .stButton > button {
        width: 100%;
        height: 3.5em;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        font-size: 1.2em;
        font-weight: 700;
        border: none;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6); }
        50% { box-shadow: 0 15px 60px rgba(102, 126, 234, 0.9); }
    }
    
    /* ===== METRIC CARD ===== */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        text-align: center;
        color: white !important;
        margin: 1rem 0;
        animation: slideUp 0.5s ease-out;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .metric-card h1 {
        font-size: 3.5em;
        font-weight: 800;
        margin: 0.2em 0;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        color: white !important;
    }
    
    .metric-card h3 {
        font-size: 1.3em;
        font-weight: 600;
        margin-bottom: 0;
        opacity: 0.95;
        color: white !important;
    }
    
    .metric-card p {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* ===== INPUT SECTION ===== */
    .input-section {
        background: rgba(40, 40, 60, 0.8);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* ===== RESULT SECTION ===== */
    .result-section {
        background: rgba(40, 40, 60, 0.8);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
        min-height: 400px;
    }
    
    /* ===== HEADERS ===== */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        font-size: 2.5em;
    }
    
    h2, h3 {
        color: #FFFFFF !important;
        font-weight: 700;
    }
    
    /* ===== SIDEBAR STYLING ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f3a 0%, #2d1b4e 100%);
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* ===== INFO BOXES ===== */
    .info-box {
        background: rgba(102, 126, 234, 0.2);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        color: white !important;
    }
    
    .info-box strong {
        color: #FFFFFF !important;
    }
    
    .success-box {
        background: rgba(0, 200, 83, 0.2);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #00c853;
        margin: 1rem 0;
        color: white !important;
    }
    
    .success-box strong {
        color: #FFFFFF !important;
    }
    
    /* ===== STAT BADGES ===== */
    .stat-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* ===== PRICE RANGE CARD (FIXED CONTRAST) ===== */
    .price-range {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid rgba(102, 126, 234, 0.5);
    }
    
    .price-range h4 {
        color: #FFFFFF !important;
        font-weight: 700;
    }
    
    /* ===== STREAMLIT WIDGET OVERRIDES ===== */
    .stSelectbox label, .stNumberInput label, .stSlider label, .stRadio label {
        color: #FFFFFF !important;
        font-weight: 600;
    }
    
    .stTextInput label {
        color: #FFFFFF !important;
        font-weight: 600;
    }
    
    /* Metric widget text */
    [data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #667eea !important;
        font-weight: 700;
    }
    
    /* ===== LOADING CONTAINER ===== */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 400px;
    }
    
    .loading-container h3 {
        color: #667eea !important;
    }
    
    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background-color: rgba(102, 126, 234, 0.2);
        color: white !important;
        border-radius: 10px;
    }
    
    /* ===== MARKDOWN TEXT ===== */
    .stMarkdown {
        color: #FFFFFF !important;
    }
    
    /* ===== INFO/WARNING/ERROR MESSAGES (HIDDEN) ===== */
    .stAlert {
        display: none !important;
    }
    
    /* ===== DOWNLOAD BUTTON ===== */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA & CONSTANTS
# ============================================================================

CAR_MODELS = [
    "Maruti Swift Dzire",
    "Maruti Alto",
    "Hyundai i20",
    "Honda City",
    "Maruti WagonR",
    "Hyundai Creta",
    "Maruti Baleno",
    "Honda Civic",
    "Toyota Innova",
    "Mahindra XUV500",
    "Tata Nexon",
    "Ford EcoSport",
    "Renault Kwid",
    "Maruti Ertiga",
    "Hyundai Verna",
    "Other (Type below)"
]

FUEL_TYPES = ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
SELLER_TYPES = ["Individual", "Dealer", "Trustmark Dealer"]
TRANSMISSION_TYPES = ["Manual", "Automatic"]
OWNER_TYPES = [
    "First Owner",
    "Second Owner",
    "Third Owner",
    "Fourth & Above Owner",
    "Test Drive Car"
]

# ============================================================================
# LOTTIE ANIMATION HELPERS (SILENT ERROR HANDLING)
# ============================================================================

def load_lottieurl(url: str) -> Optional[Dict]:
    """
    Fetch and load Lottie animation from URL.
    
    PRODUCTION VERSION: Fails silently without showing errors to users.
    
    Args:
        url (str): URL to Lottie JSON file
        
    Returns:
        dict: Lottie animation JSON or None if failed
    """
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        # Fail silently - don't show errors to users
        return None


def get_placeholder_animation() -> Dict:
    """
    Return a simple placeholder animation if external animations fail.
    
    Returns:
        dict: Simple spinning circle animation
    """
    return {
        "v": "5.5.7",
        "fr": 30,
        "ip": 0,
        "op": 60,
        "w": 200,
        "h": 200,
        "nm": "Loading",
        "ddd": 0,
        "assets": [],
        "layers": []
    }

# ============================================================================
# API FUNCTIONS
# ============================================================================

def check_api_health() -> bool:
    """Check if API is healthy."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200 and response.json().get("status") == "healthy"
    except:
        return False

def get_prediction(car_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get price prediction from API."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=car_data,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"❌ API Error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API service")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_animated_header():
    """Render animated header."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>🚗 Car Price AI</h1>", 
                   unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #FFFFFF; font-size: 1.2em;'>Cardekho Dataset • 93% Accuracy • Instant Predictions</p>", 
                   unsafe_allow_html=True)

def render_professional_sidebar():
    """Render professional sidebar."""
    with st.sidebar:
        # Try to load animation, use emoji fallback if fails
        lottie_car = load_lottieurl(LOTTIE_CAR_URL)
        if lottie_car:
            st_lottie(lottie_car, height=200, key="sidebar_car")
        else:
            # Fallback: Display emoji
            st.markdown("<div style='text-align: center; font-size: 100px;'>🚗</div>", 
                       unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🎯 How It Works")
        st.markdown("""
        <div class="info-box">
        <strong>1. Enter Details</strong><br>
        Fill in your car's specifications<br><br>
        <strong>2. AI Analysis</strong><br>
        XGBoost processes features<br><br>
        <strong>3. Get Price</strong><br>
        Instant market value estimate
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Model Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="stat-badge">R² 0.93</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="stat-badge">1000 Trees</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🔌 System Status")
        is_healthy = check_api_health()
        
        if is_healthy:
            st.markdown('<div class="success-box">✅ <strong>API Online</strong><br>Ready for predictions</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div style="background: rgba(244, 67, 54, 0.2); padding: 1rem; border-radius: 10px; border-left: 4px solid #f44336; color: white;">❌ <strong>API Offline</strong><br>Please start the API service</div>', 
                       unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption("© 2024 MLOps Team • Cardekho V5.0")

def render_input_form():
    """Render the input form with all Cardekho V5.0 fields."""
    st.subheader("🚗 Car Details")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        car_model_selection = st.selectbox(
            "Car Model",
            options=CAR_MODELS,
            help="Select the car model"
        )
        
        if car_model_selection == "Other (Type below)":
            car_name = st.text_input("Enter Car Model", placeholder="e.g., Tata Nexon")
        else:
            car_name = car_model_selection
        
        current_year = datetime.now().year
        year = st.slider(
            "📅 Manufacturing Year",
            min_value=2000,
            max_value=current_year,
            value=2015,
            help="Select the year your car was manufactured"
        )
        
        km_driven = st.number_input(
            "🛣️ Kilometers Driven",
            min_value=0,
            max_value=500000,
            value=45000,
            step=1000,
            help="Total kilometers driven"
        )
        
        mileage = st.number_input(
            "⛽ Mileage (kmpl)",
            min_value=5.0,
            max_value=50.0,
            value=20.0,
            step=0.1,
            help="Fuel efficiency in kilometers per liter"
        )
        
        engine = st.number_input(
            "🔧 Engine (CC)",
            min_value=500,
            max_value=5000,
            value=1200,
            step=100,
            help="Engine capacity in cubic centimeters"
        )
        
        max_power = st.number_input(
            "⚡ Max Power (bhp)",
            min_value=20.0,
            max_value=500.0,
            value=80.0,
            step=5.0,
            help="Maximum power output in brake horsepower"
        )
    
    with col2:
        fuel = st.selectbox(
            "⛽ Fuel Type",
            options=FUEL_TYPES,
            help="Select the fuel type"
        )
        
        st.markdown("🔄 **Transmission**")
        transmission = st.radio(
            "Transmission Type",
            options=TRANSMISSION_TYPES,
            horizontal=True,
            label_visibility="collapsed"
        )
        
        seller_type = st.selectbox(
            "👤 Seller Type",
            options=SELLER_TYPES,
            help="Select the seller type"
        )
        
        owner = st.selectbox(
            "👥 Owner",
            options=OWNER_TYPES,
            help="Select the ownership history"
        )
        
        seats = st.slider(
            "💺 Number of Seats",
            min_value=2,
            max_value=10,
            value=5,
            help="Total seating capacity"
        )
        
        car_age = current_year - year
        st.info(f"🕐 Car Age: **{car_age}** years")
    
    st.markdown("---")
    
    return {
        "name": car_name,
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats
    }

def render_prediction_result(prediction: Dict[str, Any], input_data: Dict[str, Any]):
    """Render the prediction result."""
    st.markdown("---")
    st.subheader("📊 Price Prediction")
    
    predicted_price = prediction.get("predicted_price", 0)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Estimated Market Value</h3>
            <h1>₹{predicted_price:,.0f}</h1>
            <p>INR • V5.0 (93% R²)</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Car Age", f"{datetime.now().year - input_data['year']} years")
    
    with col2:
        st.metric("KM Driven", f"{input_data['km_driven']:,} km")
    
    with col3:
        st.metric("Model", "V5.0")
    
    st.markdown("---")
    st.subheader("📈 Price Range Estimate")
    
    lower_bound = predicted_price * 0.95
    upper_bound = predicted_price * 1.05
    
    st.markdown('<div class="price-range">', unsafe_allow_html=True)
    st.markdown("#### Expected Price Range")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Lower Estimate", f"₹{lower_bound:,.0f}", "-5%")
    
    with col2:
        st.metric("Upper Estimate", f"₹{upper_bound:,.0f}", "+5%")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### ℹ️ Important Notes")
    st.markdown("""
    - Price based on Cardekho market data
    - 93% accuracy on test dataset
    - Consider local market conditions
    - Professional inspection recommended
    """)
    
    report_text = f"""
CAR PRICE PREDICTION REPORT (CARDEKHO V5.0)
==========================================

Car Details:
- Model: {input_data['name']}
- Year: {input_data['year']}
- Kilometers Driven: {input_data['km_driven']:,} km
- Fuel: {input_data['fuel']}
- Transmission: {input_data['transmission']}
- Seller Type: {input_data['seller_type']}
- Owner: {input_data['owner']}
- Mileage: {input_data['mileage']} kmpl
- Engine: {input_data['engine']} CC
- Max Power: {input_data['max_power']} bhp
- Seats: {input_data['seats']}

Prediction:
- Estimated Price: ₹{predicted_price:,.0f}
- Price Range: ₹{lower_bound:,.0f} - ₹{upper_bound:,.0f}
- Model: V5.0 (93% R² accuracy)
- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    st.download_button(
        "📥 Download Report",
        data=report_text,
        file_name=f"car_price_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True
    )

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    
    inject_custom_css()
    render_animated_header()
    render_professional_sidebar()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_input, col_results = st.columns([1, 1], gap="large")
    
    with col_input:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        input_data = render_input_form()
        predict_button = st.button("🔮 Predict Price Now", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_results:
        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        
        st.markdown("### 📊 Prediction Result")
        st.markdown("---")
        
        if 'prediction_result' not in st.session_state:
            st.session_state.prediction_result = None
        
        if predict_button:
            if not input_data["name"] or input_data["name"].strip() == "":
                st.error("❌ Please enter a valid car model!")
            else:
                placeholder = st.empty()
                
                with placeholder.container():
                    st.markdown('<div class="loading-container" style="text-align: center;">', unsafe_allow_html=True)
                    
                    # Try to load animation, use emoji fallback
                    lottie_loading = load_lottieurl(LOTTIE_LOADING_URL)
                    if lottie_loading:
                        st_lottie(lottie_loading, height=250, key="loading")
                    else:
                        st.markdown("<div style='font-size: 80px; animation: spin 2s linear infinite;'>🔄</div>", 
                                   unsafe_allow_html=True)
                    
                    st.markdown("<h3 style='text-align: center; color: #667eea;'>🤖 AI Analyzing...</h3>", 
                               unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                api_payload = {
                    "name": input_data["name"],
                    "year": input_data["year"],
                    "km_driven": input_data["km_driven"],
                    "fuel": input_data["fuel"],
                    "seller_type": input_data["seller_type"],
                    "transmission": input_data["transmission"],
                    "owner": input_data["owner"],
                    "mileage": input_data["mileage"],
                    "engine": input_data["engine"],
                    "max_power": input_data["max_power"],
                    "seats": input_data["seats"]
                }
                
                time.sleep(1)
                prediction = get_prediction(api_payload)
                
                placeholder.empty()
                
                if prediction:
                    st.session_state.prediction_result = prediction
                    
                    success_placeholder = st.empty()
                    with success_placeholder.container():
                        lottie_success = load_lottieurl(LOTTIE_SUCCESS_URL)
                        if lottie_success:
                            st_lottie(lottie_success, height=150, key="success")
                        else:
                            st.markdown("<div style='text-align: center; font-size: 80px;'>✅</div>", 
                                       unsafe_allow_html=True)
                    
                    time.sleep(1)
                    success_placeholder.empty()
                    
                    render_prediction_result(prediction, input_data)
        
        elif st.session_state.prediction_result is None:
            st.markdown('<div class="loading-container">', unsafe_allow_html=True)
            
            lottie_car_placeholder = load_lottieurl(LOTTIE_CAR_URL)
            if lottie_car_placeholder:
                st_lottie(lottie_car_placeholder, height=300, key="placeholder_car")
            else:
                st.markdown("<div style='text-align: center; font-size: 100px;'>🚗</div>", 
                           unsafe_allow_html=True)
            
            st.markdown("<h3 style='text-align: center; color: #FFFFFF;'>👈 Enter car details and click Predict</h3>", 
                       unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    with st.expander("📚 Example Predictions (Indian Market)"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Maruti Swift Dzire 2014")
            st.markdown("**KM Driven:** 145,500 km")
            st.markdown("**Fuel:** Diesel")
            st.markdown("**Transmission:** Manual")
            st.markdown("**Owner:** First Owner")
            st.markdown("**Est. Price:** ~₹3,50,000")
        
        with col2:
            st.markdown("#### Hyundai i20 2017")
            st.markdown("**KM Driven:** 35,000 km")
            st.markdown("**Fuel:** Petrol")
            st.markdown("**Transmission:** Manual")
            st.markdown("**Owner:** First Owner")
            st.markdown("**Est. Price:** ~₹5,80,000")
        
        with col3:
            st.markdown("#### Honda City 2018")
            st.markdown("**KM Driven:** 20,000 km")
            st.markdown("**Fuel:** Petrol")
            st.markdown("**Transmission:** Automatic")
            st.markdown("**Owner:** First Owner")
            st.markdown("**Est. Price:** ~₹9,50,000")

if __name__ == "__main__":
    main()
