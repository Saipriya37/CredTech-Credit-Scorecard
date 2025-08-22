# dashboard/app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="CredTech Credit Scorecard", layout="wide")

# Load processed data
stock_df = pd.read_csv("data/stock_data.csv")

st.title("💳 CredTech - Explainable Credit Scorecard")

st.subheader("📈 Stock Data (Recent 10 rows)")
st.dataframe(stock_df.tail(10))

# Show SHAP feature importance
st.subheader("📊 Feature Importance (SHAP)")
image_path = "data/shap_feature_importance.png"
st.image(image_path, caption="Feature importance explaining credit risk score")

st.info("✅ This is a prototype dashboard for Round 1 submission.")
