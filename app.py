# CUSTOMER SEGMENTATION SYSTEM
# Professional Streamlit Dashboard

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------------
# 2. CUSTOM CSS
# -----------------------------------------

st.markdown("""
<style>
    .main-title {
        font-size: 34px;
        font-weight: 700;
        color: #2563EB;
    }

    .subtitle {
        color: #64748B;
        font-size: 16px;
    }

    .result-card {
        padding: 22px;
        border-radius: 12px;
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        margin-top: 15px;
    }

    .result-title {
        font-size: 23px;
        font-weight: 700;
        color: #1D4ED8;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# 3. LOAD TRAINED MODEL
# -----------------------------------------

@st.cache_resource
def load_model():

    model_path = (
        Path(__file__).parent /
        "customer_segmentation_model.joblib"
    )

    return joblib.load(model_path)


model_data = load_model()

kmeans = model_data["model"]
scaler = model_data["scaler"]
cluster_labels = model_data["cluster_labels"]

# -----------------------------------------
# 4. CUSTOMER SEGMENT INFORMATION
# -----------------------------------------

segment_info = {
    0: {
        "name": "Low-Income, Low-Spending",
        "income": 26.30,
        "spending": 20.91,
        "customers": 23,
        "description": "Customers with relatively low annual income "
                       "and low spending scores."
    },

    1: {
        "name": "Low-Income, High-Spending",
        "income": 25.73,
        "spending": 79.36,
        "customers": 22,
        "description": "Customers with relatively low annual income "
                       "but high spending scores."
    },

    2: {
        "name": "High-Income, Low-Spending",
        "income": 88.20,
        "spending": 17.11,
        "customers": 35,
        "description": "Customers with relatively high annual income "
                       "and low spending scores."
    },

    3: {
        "name": "High-Income, High-Spending",
        "income": 86.54,
        "spending": 82.13,
        "customers": 39,
        "description": "Customers with relatively high annual income "
                       "and high spending scores."
    },

    4: {
        "name": "Moderate-Income, Moderate-Spending",
        "income": 55.30,
        "spending": 49.52,
        "customers": 81,
        "description": "Customers with moderate annual income "
                       "and spending scores."
    }
}

# -----------------------------------------
# 5. SESSION STATE
# -----------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None

# -----------------------------------------
# 6. SIDEBAR NAVIGATION
# -----------------------------------------

st.sidebar.title("📊 Customer AI")

st.sidebar.caption("Customer Segmentation System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Predict Customer",
        "Segment Overview",
        "Prediction History",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Powered by Python, Streamlit "
    "and K-Means Clustering."
)

# =========================================
# PAGE 1: PREDICT CUSTOMER
# =========================================

if page == "Predict Customer":

    st.markdown(
        '<div class="main-title">Customer Segmentation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Identify customer groups '
        'using your trained Machine Learning model.</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("Enter Customer Information")

    with st.form("customer_form"):

        col1, col2 = st.columns(2)

        with col1:

            annual_income = st.number_input(
                "Annual Income (in thousands $)",
                min_value=0.0,
                max_value=200.0,
                value=50.0,
                step=1.0
            )

        with col2:

            spending_score = st.slider(
                "Spending Score",
                min_value=1,
                max_value=100,
                value=50
            )

        submitted = st.form_submit_button(
            "🔍 Predict Customer Segment",
            use_container_width=True
        )

    if submitted:

        input_data = np.array([
            [annual_income, spending_score]
        ])

        scaled_input = scaler.transform(input_data)

        prediction = int(
            kmeans.predict(scaled_input)[0]
        )

        result = {
            "Income": annual_income,
            "Spending Score": spending_score,
            "Cluster": prediction,
            "Segment": cluster_labels[prediction]
        }

        st.session_state.latest_prediction = result

        st.session_state.history.insert(0, result)

    # Display prediction

    if st.session_state.latest_prediction is not None:

        result = st.session_state.latest_prediction

        info = segment_info[result["Cluster"]]

        st.divider()

        st.subheader("Prediction Result")

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">
                    {result["Segment"]}
                </div>
                <p>{info["description"]}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Assigned Cluster",
            result["Cluster"]
        )

        col2.metric(
            "Annual Income",
            f"${result['Income']}k"
        )

        col3.metric(
            "Spending Score",
            result["Spending Score"]
        )

# =========================================
# PAGE 2: SEGMENT OVERVIEW
# =========================================

elif page == "Segment Overview":

    st.title("Customer Segment Overview")

    st.write(
        "Explore the five customer groups discovered "
        "by the trained K-Means algorithm."
    )

    total_customers = sum(
        item["customers"] for item in segment_info.values()
    )

    col1, col2 = st.columns(2)

    col1.metric("Total Customers", total_customers)
    col2.metric("Identified Segments", len(segment_info))

    st.divider()

    segment_df = pd.DataFrame([
        {
            "Cluster": cluster,
            "Segment": item["name"],
            "Annual Income (k$)": item["income"],
            "Spending Score": item["spending"],
            "Customers": item["customers"]
        }
        for cluster, item in segment_info.items()
    ])

    st.subheader("Customer Distribution")

    chart_data = segment_df.set_index("Segment")["Customers"]

    st.bar_chart(chart_data)

    st.subheader("Segment Details")

    st.dataframe(
        segment_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Segment averages and customer counts are from "
        "your original K-Means training results."
    )

# =========================================
# PAGE 3: PREDICTION HISTORY
# =========================================

elif page == "Prediction History":

    st.title("Prediction History")

    st.write(
        "Previous predictions made during this "
        "application session."
    )

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button("Clear Prediction History"):

            st.session_state.history = []
            st.session_state.latest_prediction = None

            st.rerun()

    else:

        st.info(
            "No predictions available yet. "
            "Go to Predict Customer to begin."
        )

# =========================================
# PAGE 4: ABOUT PROJECT
# =========================================

elif page == "About Project":

    st.title("About the Project")

    st.write(
        "Customer Segmentation System is an "
        "unsupervised Machine Learning application "
        "that identifies customer groups based on "
        "annual income and spending score."
    )

    st.subheader("Machine Learning Techniques")

    st.markdown("""
    - Feature Selection
    - Feature Scaling using StandardScaler
    - K-Means Clustering
    - Agglomerative Hierarchical Clustering
    - Principal Component Analysis (PCA)
    """)

    st.subheader("Technology Stack")

    st.markdown("""
    - Python
    - Streamlit
    - NumPy
    - Pandas
    - Scikit-learn
    - Joblib
    """)

    st.subheader("Model Information")

    st.write("Algorithm: K-Means Clustering")
    st.write("Number of Clusters: 5")
    st.write("Training Records: 200")
    st.write(
        "Input Features: Annual Income and Spending Score"
    )

    st.divider()

    st.caption(
        "Academic Machine Learning Project | "
        "Customer Segmentation System"
    )

# -----------------------------------------
# END OF APPLICATION
# -----------------------------------------