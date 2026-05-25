import streamlit as st
import pickle
import pandas as pd
import numpy as np

# =========================
# LOAD MODELS
# =========================
kmeans = pickle.load(open("models/kmeans.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
features = pickle.load(open("models/features.pkl", "rb"))

le_gender = pickle.load(open("models/le_gender.pkl", "rb"))
le_category = pickle.load(open("models/le_category.pkl", "rb"))
le_payment = pickle.load(open("models/le_payment.pkl", "rb"))
le_mall = pickle.load(open("models/le_mall.pkl", "rb"))

# =========================
# PAGE CONFIG (PRO UI)
# =========================
st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    page_icon="🧠",
    layout="wide"
)

# =========================
# HEADER
# =========================
st.title("🧠 Customer Intelligence Dashboard")
st.markdown("### AI-powered Customer Segmentation using K-Means Clustering")

st.divider()

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("Customer Input Panel")

gender = st.sidebar.selectbox("Gender", le_gender.classes_)
age = st.sidebar.slider("Age", 18, 80, 30)

category = st.sidebar.selectbox("Category", le_category.classes_)
quantity = st.sidebar.number_input("Quantity", 1, 20, 2)
price = st.sidebar.number_input("Price", 1.0, 1000.0, 100.0)

payment_method = st.sidebar.selectbox("Payment Method", le_payment.classes_)
shopping_mall = st.sidebar.selectbox("Shopping Mall", le_mall.classes_)
month = st.sidebar.slider("Month", 1, 12, 6)

# =========================
# FEATURE ENGINEERING
# =========================
total_price = quantity * price

# =========================
# ENCODING
# =========================
gender_encoded = le_gender.transform([gender])[0]
category_encoded = le_category.transform([category])[0]
payment_encoded = le_payment.transform([payment_method])[0]
mall_encoded = le_mall.transform([shopping_mall])[0]

# =========================
# INPUT DATA
# =========================
input_data = pd.DataFrame([[
    gender_encoded,
    age,
    category_encoded,
    quantity,
    price,
    payment_encoded,
    mall_encoded,
    total_price,
    month
]], columns=features)

# =========================
# PREDICTION
# =========================
scaled = scaler.transform(input_data)
cluster = kmeans.predict(scaled)[0]

# =========================
# CLUSTER INSIGHTS (REAL WORLD)
# =========================
cluster_info = {
    0: {
        "name": "💰 Budget Customers",
        "desc": "Low spending, price-sensitive customers",
        "action": "Offer discounts & bundle deals"
    },
    1: {
        "name": "👑 Premium Customers",
        "desc": "High spending loyal customers",
        "action": "VIP membership & exclusive offers"
    },
    2: {
        "name": "🛍️ Frequent Buyers",
        "desc": "Buy often but moderate spending",
        "action": "Loyalty points & reward system"
    },
    3: {
        "name": "💎 Luxury Shoppers",
        "desc": "High value, premium product buyers",
        "action": "Personalized premium service"
    },
    4: {
        "name": "🙂 Average Customers",
        "desc": "Balanced shopping behavior",
        "action": "Targeted marketing campaigns"
    }
}

info = cluster_info.get(cluster, {
    "name": "Unknown Segment",
    "desc": "No data available",
    "action": "Analyze further"
})

# =========================
# RESULT UI
# =========================
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Predicted Cluster", cluster)

with col2:
    st.metric("Customer Segment", info["name"])

with col3:
    st.metric("Total Spend", f"${total_price:.2f}")

st.divider()

# =========================
# INSIGHT CARDS
# =========================
st.subheader("📊 Customer Profile")

st.success(info["name"])
st.write("📌 Description:", info["desc"])
st.warning("🎯 Recommended Action: " + info["action"])

# =========================
# BUSINESS IMPACT SECTION
# =========================
st.subheader("📈 Business Strategy Insight")

if cluster == 1:
    st.info("Focus on retention strategies and premium loyalty programs.")
elif cluster == 0:
    st.info("Offer discounts, coupons, and budget-friendly bundles.")
elif cluster == 2:
    st.info("Increase purchase value using reward points system.")
elif cluster == 3:
    st.info("Provide premium experience and personalized service.")
else:
    st.info("Run targeted marketing campaigns to improve engagement.")

# =========================
# DOWNLOAD REPORT
# =========================
st.divider()

report = pd.DataFrame([{
    "Gender": gender,
    "Age": age,
    "Category": category,
    "Total Price": total_price,
    "Cluster": cluster,
    "Segment": info["name"]
}])

csv = report.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Download Customer Report",
    csv,
    "customer_segment_report.csv",
    "text/csv"
)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Built with ❤️ using K-Means Clustering & Streamlit")