import streamlit as st
import pandas as pd
import joblib
import base64


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# BACKGROUND IMAGE
# =========================================================

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


image_base64 = get_base64_image("assets/house.png")


# =========================================================
# CSS DESIGN
# =========================================================

st.markdown(
f"""
<style>

.stApp {{
    background-image:
        linear-gradient(
            rgba(5, 15, 30, 0.60),
            rgba(5, 15, 30, 0.60)
        ),
        url("data:image/png;base64,{image_base64}");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}


/* MAIN CONTAINER */

.block-container {{
    max-width: 1350px;
    padding-top: 35px;
    padding-bottom: 30px;
}}


/* HIDE STREAMLIT ELEMENTS */

#MainMenu {{
    visibility: hidden;
}}

header {{
    visibility: hidden;
}}

footer {{
    visibility: hidden;
}}


/* =====================================================
   TITLE
   ===================================================== */

.main-title {{
    color: white !important;
    font-size: 52px;
    font-weight: 800;
    text-align: center;
    text-shadow: 2px 2px 10px black;
    margin-bottom: 5px;
}}

.subtitle {{
    color: white !important;
    font-size: 20px;
    text-align: center;
    text-shadow: 1px 1px 6px black;
    margin-bottom: 30px;
}}


/* =====================================================
   INFORMATION CARDS
   ===================================================== */

.info-card {{
    background: rgba(255,255,255,0.96);
    border-radius: 18px;
    padding: 22px;
    min-height: 120px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.30);
}}

.info-label {{
    color: #475569 !important;
    font-size: 17px;
    font-weight: 600;
}}

.info-value {{
    color: #17213a !important;
    font-size: 32px;
    font-weight: 800;
    margin-top: 8px;
}}


/* =====================================================
   SUCCESS MESSAGE
   ===================================================== */

.success-box {{
    background: rgba(30,130,76,0.92);
    border-radius: 14px;
    padding: 17px 22px;
    color: white !important;
    font-size: 18px;
    font-weight: 700;
    margin-top: 22px;
    margin-bottom: 20px;
}}


/* =====================================================
   PREDICTION CARD
   ===================================================== */

.prediction-card {{
    background: rgba(255,255,255,0.97);
    border-radius: 22px;
    padding: 30px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.35);
    margin-bottom: 25px;
}}

.prediction-heading {{
    color: #17213a !important;
    font-size: 28px;
    font-weight: 800;
}}

.price {{
    color: #2563eb !important;
    font-size: 52px;
    font-weight: 900;
    margin-top: 8px;
    margin-bottom: 22px;
}}

.model-box {{
    background: #dbeafe;
    border-radius: 12px;
    padding: 15px;
    color: #174ea6 !important;
    font-size: 18px;
    font-weight: 700;
}}


/* =====================================================
   CONTENT CARDS
   ===================================================== */

.content-card {{
    background: rgba(255,255,255,0.97);
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.30);
    min-height: 330px;
}}

.content-title {{
    color: #17213a !important;
    font-size: 25px;
    font-weight: 800;
    margin-bottom: 18px;
}}


/* =====================================================
   HOUSE TABLE
   ===================================================== */

.house-table {{
    width: 100%;
    border-collapse: collapse;
}}

.house-table th {{
    background: #eef2ff;
    color: #17213a !important;
    text-align: left;
    padding: 13px;
    font-weight: 800;
}}

.house-table td {{
    color: #334155 !important;
    padding: 13px;
    border-bottom: 1px solid #dbe2ea;
    font-size: 16px;
}}


/* =====================================================
   ABOUT PROJECT
   ===================================================== */

.about-item {{
    margin-bottom: 20px;
}}

.about-title {{
    color: #17213a !important;
    font-size: 17px;
    font-weight: 800;
    margin-bottom: 5px;
}}

.about-text {{
    color: #475569 !important;
    font-size: 15px;
    line-height: 1.5;
}}


/* =====================================================
   WORKFLOW
   ===================================================== */

.workflow {{
    background: rgba(8,24,45,0.94);
    border-radius: 18px;
    padding: 20px;
    margin-top: 25px;
    text-align: center;
    box-shadow: 0 8px 25px rgba(0,0,0,0.35);
}}

.workflow-title {{
    color: white !important;
    font-size: 21px;
    font-weight: 800;
    margin-bottom: 12px;
}}

.workflow-text {{
    color: white !important;
    font-size: 15px;
    line-height: 1.8;
}}


/* =====================================================
   SIDEBAR
   ===================================================== */

section[data-testid="stSidebar"] {{
    background: rgba(7,24,45,0.97) !important;
}}

section[data-testid="stSidebar"] * {{
    color: white !important;
}}


/* SIDEBAR BUTTON */

section[data-testid="stSidebar"] .stButton button {{
    width: 100%;
    background: linear-gradient(
        90deg,
        #7c3aed,
        #2563eb
    ) !important;

    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px !important;
    font-size: 17px !important;
    font-weight: 800 !important;
}}


/* =====================================================
   FOOTER
   ===================================================== */

.footer {{
    color: white !important;
    text-align: center;
    font-size: 16px;
    font-weight: 600;
    margin-top: 20px;
    text-shadow: 1px 1px 5px black;
}}

</style>
""",
unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

artifact = joblib.load("house_price_model.pkl")

model = artifact["model"]
model_name = artifact["model_name"]
feature_names = artifact["feature_names"]


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## 🏠 House Details")

st.sidebar.write("Enter the property information")

st.sidebar.divider()


area = st.sidebar.number_input(
    "📐 Area (sq. ft.)",
    min_value=500,
    max_value=10000,
    value=1800,
    step=100
)


bedrooms = st.sidebar.number_input(
    "🛏️ Bedrooms",
    min_value=1,
    max_value=10,
    value=4,
    step=1
)


bathrooms = st.sidebar.number_input(
    "🛁 Bathrooms",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)


stories = st.sidebar.number_input(
    "🏢 Stories",
    min_value=1,
    max_value=5,
    value=3,
    step=1
)


parking = st.sidebar.number_input(
    "🚗 Parking Spaces",
    min_value=0,
    max_value=5,
    value=1,
    step=1
)


age = st.sidebar.number_input(
    "📅 House Age (years)",
    min_value=0,
    max_value=100,
    value=10,
    step=1
)


st.sidebar.divider()


predict_button = st.sidebar.button(
    "🔮 Predict House Price",
    use_container_width=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🏠 House Price Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict the estimated price of a house using Machine Learning.</div>',
    unsafe_allow_html=True
)


# =========================================================
# TOP CARDS
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        f'<div class="info-card">'
        f'<div class="info-label">📐 Area</div>'
        f'<div class="info-value">{area:,}</div>'
        f'<div class="info-label">sq. ft.</div>'
        f'</div>',
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f'<div class="info-card">'
        f'<div class="info-label">🛏️ Bedrooms</div>'
        f'<div class="info-value">{bedrooms}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f'<div class="info-card">'
        f'<div class="info-label">🛁 Bathrooms</div>'
        f'<div class="info-value">{bathrooms}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    input_data = pd.DataFrame([
        {
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "parking": parking,
            "age": age
        }
    ])


    # Make sure feature order matches training

    input_data = input_data[feature_names]


    # Prediction

    prediction = model.predict(input_data)[0]


    # =====================================================
    # SUCCESS
    # =====================================================

    st.markdown(
        '<div class="success-box">'
        '✅ Prediction completed successfully! 🎉'
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # PRICE
    # =====================================================

    st.markdown(
        f'<div class="prediction-card">'
        f'<div class="prediction-heading">💰 Estimated House Price</div>'
        f'<div class="price">₹ {prediction:,.0f}</div>'
        f'<div class="model-box">🤖 Model Used: {model_name}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # INFORMATION
    # =====================================================

    left, right = st.columns([1.5, 1])


    # =====================================================
    # HOUSE INFORMATION
    # =====================================================

    with left:

        st.markdown(
            f'''
            <div class="content-card">

            <div class="content-title">
            🏡 House Information
            </div>

            <table class="house-table">

            <tr>
            <th>Feature</th>
            <th>Value</th>
            </tr>

            <tr>
            <td>📐 Area</td>
            <td>{area:,} sq. ft.</td>
            </tr>

            <tr>
            <td>🛏️ Bedrooms</td>
            <td>{bedrooms}</td>
            </tr>

            <tr>
            <td>🛁 Bathrooms</td>
            <td>{bathrooms}</td>
            </tr>

            <tr>
            <td>🏢 Stories</td>
            <td>{stories}</td>
            </tr>

            <tr>
            <td>🚗 Parking Spaces</td>
            <td>{parking}</td>
            </tr>

            <tr>
            <td>📅 House Age</td>
            <td>{age} years</td>
            </tr>

            </table>

            </div>
            ''',
            unsafe_allow_html=True
        )


    # =====================================================
    # ABOUT PROJECT
    # =====================================================

    with right:

        st.markdown(
            '''
            <div class="content-card">

            <div class="content-title">
            📊 About This Project
            </div>

            <div class="about-item">

            <div class="about-title">
            🤖 Machine Learning
            </div>

            <div class="about-text">
            A regression-based machine learning model
            is used to estimate house prices.
            </div>

            </div>


            <div class="about-item">

            <div class="about-title">
            📈 Input Features
            </div>

            <div class="about-text">
            Area, bedrooms, bathrooms, stories,
            parking and house age.
            </div>

            </div>


            <div class="about-item">

            <div class="about-title">
            💰 Prediction
            </div>

            <div class="about-text">
            The trained model predicts an estimated
            house price based on the entered property
            information.
            </div>

            </div>

            </div>
            ''',
            unsafe_allow_html=True
        )


else:

    st.markdown(
        '''
        <div class="prediction-card">

        <div class="prediction-heading">
        💰 Estimated House Price
        </div>

        <p style="color:#475569 !important;font-size:17px;">
        👈 Enter the house details from the sidebar
        and click <b>🔮 Predict House Price</b>.
        </p>

        </div>
        ''',
        unsafe_allow_html=True
    )


# =========================================================
# WORKFLOW
# =========================================================

st.markdown(
    '''
    <div class="workflow">

    <div class="workflow-title">
    🔄 Project Workflow
    </div>

    <div class="workflow-text">
    📁 Dataset
    →
    🧹 Data Preprocessing
    →
    🤖 Model Training
    →
    📊 Model Evaluation
    →
    🏆 Best Model Selection
    →
    🏠 House Details
    →
    💰 Price Prediction
    </div>

    </div>
    ''',
    unsafe_allow_html=True
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '''
    <div class="footer">
    🏠 House Price Prediction
    &nbsp; | &nbsp;
    Machine Learning Project
    &nbsp; | &nbsp;
    Made with ❤️
    </div>
    ''',
    unsafe_allow_html=True
)