import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Warehouse AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

.main{
    background:#F8FAFC;
}

h1{
    color:#0F172A;
}

[data-testid="stMetricValue"]{
    font-size:30px;
    color:#1E3A8A;
}

[data-testid="stMetricLabel"]{
    font-size:18px;
    font-weight:bold;
}

div[data-testid="stDataFrame"]{
    border-radius:15px;
}

</style>
""",unsafe_allow_html=True)


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Agentic AI Warehouse Decision Support System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

model, label_encoder = load_model()
# ==========================================
# AI REASONING ENGINE
# ==========================================

def generate_reason(row):

    reasons = []

    # Shelf life
    if row["remaining_shelf_life_percent"] <= 20:
        reasons.append("Very low remaining shelf life")

    elif row["remaining_shelf_life_percent"] <= 40:
        reasons.append("Shelf life reducing")

    # Inventory cover
    if row["inv_cover"] == "No Sales":
        reasons.append("No recent sales")

    else:
        try:
            if float(row["inv_cover"]) > 6:
                reasons.append("High inventory cover")
        except:
            pass

    # Monthly sales
    if row["Avg_mon_sales"] == 0:
        reasons.append("Zero monthly sales")

    # Inventory value
    if row["value"] > 100000:
        reasons.append("High inventory value")

    if len(reasons) == 0:
        reasons.append("Inventory within healthy limits")

    return " | ".join(reasons)


# ==========================================
# RECOMMENDATION ENGINE
# ==========================================

def recommendation(action):

    if action == "Move Immediately":
        return "Transfer stock immediately or run clearance sale."

    elif action == "High Priority":
        return "Increase dispatch and monitor every week."

    elif action == "Monitor":
        return "Review inventory during next planning cycle."

    else:
        return "No immediate action required."

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

.block-container{
    padding-top:1rem;
}

.title{
    font-size:42px;
    font-weight:bold;
    color:#003366;
}

.subtitle{
    color:gray;
    font-size:18px;
}

.metric{
    background:white;
    padding:15px;
    border-radius:12px;
    box-shadow:0px 0px 10px rgba(0,0,0,0.08);
}

</style>
""",unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📦 Agentic Warehouse AI")

st.sidebar.markdown("---")

st.sidebar.write("### System Features")

st.sidebar.success("✔ AI Prediction")

st.sidebar.success("✔ Risk Detection")

st.sidebar.success("✔ Recommendation Engine")

st.sidebar.success("✔ Inventory Dashboard")

st.sidebar.success("✔ Download Report")

st.sidebar.markdown("---")

st.sidebar.info(
"""
Developed for

Warehouse Decision Support

using

Random Forest AI
"""
)

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
'<p class="title">📦 Agentic AI Warehouse Decision Support System</p>',
unsafe_allow_html=True
)

st.markdown(
'<p class="subtitle">Predict • Explain • Recommend • Prioritize</p>',
unsafe_allow_html=True
)

st.divider()

# -----------------------------
# FILE UPLOADER
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Inventory Excel",
    type=["xlsx"]
)

if uploaded_file is None:

    st.info("Upload an inventory Excel file to begin.")

    st.stop()

# -----------------------------
# READ EXCEL
# -----------------------------
df = pd.read_excel(uploaded_file)

st.sidebar.markdown("---")
st.sidebar.header("🔍 Filters")

selected_brand = st.sidebar.selectbox(
    "Brand",
    ["All"] + sorted(df["brand"].dropna().unique().tolist())
)

selected_category = st.sidebar.selectbox(
    "Category",
    ["All"] + sorted(df["category"].dropna().unique().tolist())
)

st.success("✅ Inventory uploaded successfully!")

# ==========================
# KPI SECTION
# ==========================

st.subheader("📦 Warehouse Overview")

total_products = len(df)

total_brands = df["brand"].nunique()

total_categories = df["category"].nunique()

inventory_value = df["value"].sum()

# Critical products count
critical = (df["recommended_action"] == "Move Immediately").sum()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "📦 Products",
    f"{total_products:,}"
)

col2.metric(
    "🏷 Brands",
    total_brands
)

col3.metric(
    "📂 Categories",
    total_categories
)

col4.metric(
    "💰 Inventory Value",
    f"₹ {inventory_value:,.0f}"
)

col5.metric(
    "🚨 Critical",
    critical
)

st.divider()

st.subheader("📄 Inventory Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)

# ==========================================
# AI PREDICTION ENGINE
# ==========================================

st.divider()
st.subheader("🤖 AI Prediction Engine")

# Rename columns only if required
rename_dict = {
    "expiry_risk": "expiry_risk.1",
    "overstock_risk": "overstock_risk.1",
    "sales_risk": "sales_risk.1",
    "value_risk": "value_risk.1"
}

for old_col, new_col in rename_dict.items():
    if old_col in df.columns and new_col not in df.columns:
        df.rename(columns={old_col: new_col}, inplace=True)

features = [
    'remaining_shelf_life_percent',
    'days_to_expiry',
    'Total Qty',
    'value',
    'Avg_mon_sales',
    'expiry_risk.1',
    'overstock_risk.1',
    'sales_risk.1',
    'value_risk.1'
]

# Check whether all required columns are present
missing = [col for col in features if col not in df.columns]

if missing:

    st.error("Missing columns required for prediction:")

    st.write(missing)

    st.stop()

# Predict
X = df[features]

predictions = model.predict(X)

df["Predicted Action"] = label_encoder.inverse_transform(predictions)
try:
    probability = model.predict_proba(X)
    df["AI Confidence (%)"] = (
        probability.max(axis=1) * 100
    ).round(2)

except:
    df["AI Confidence (%)"] = 100
df["AI Reason"] = df.apply(generate_reason, axis=1)

df["Recommendation"] = df["Predicted Action"].apply(recommendation)

selected_action = st.sidebar.selectbox(
    "Predicted Action",
    ["All"] + sorted(df["Predicted Action"].unique().tolist())
)

# ==========================================
# APPLY FILTERS
# ==========================================

filtered_df = df.copy()

if selected_brand != "All":
    filtered_df = filtered_df[
        filtered_df["brand"] == selected_brand
    ]

if selected_category != "All":
    filtered_df = filtered_df[
        filtered_df["category"] == selected_category
    ]

if selected_action != "All":
    filtered_df = filtered_df[
        filtered_df["Predicted Action"] == selected_action
    ]

# Confidence Score
try:

    probability = model.predict_proba(X)

    df["AI Confidence (%)"] = probability.max(axis=1) * 100

except:

    df["AI Confidence (%)"] = 100

st.success("✅ AI Prediction Completed Successfully!")

st.subheader("Prediction Results")

display_cols = [
    "sku_name",
    "brand",
    "Predicted Action",
    "AI Confidence (%)",
    "AI Reason",
    "Recommendation"
]

st.dataframe(
    filtered_df[display_cols],
    use_container_width=True
)

st.subheader("AI Prediction Summary")

summary = (
    filtered_df["Predicted Action"]
    .value_counts()
    .reset_index()
)

summary.columns = ["Action", "Count"]

st.dataframe(
    summary,
    use_container_width=True
)

# ==========================================
# AI EXECUTIVE SUMMARY
# ==========================================

st.divider()
st.subheader("🤖 AI Executive Summary")

total_products = len(filtered_df)

move_now = (filtered_df["Predicted Action"] == "Move Immediately").sum()
high_priority = (filtered_df["Predicted Action"] == "High Priority").sum()
monitor = (filtered_df["Predicted Action"] == "Monitor").sum()
healthy = (filtered_df["Predicted Action"] == "Healthy").sum()

critical_value = filtered_df.loc[
    filtered_df["Predicted Action"] == "Move Immediately",
    "value"
].sum()

# Top Brand
top_brand = (
    filtered_df[filtered_df["Predicted Action"] == "Move Immediately"]["brand"]
    .value_counts()
)

top_brand = top_brand.index[0] if len(top_brand) > 0 else "None"

# Top Category
top_category = (
    filtered_df[filtered_df["Predicted Action"] == "Move Immediately"]["category"]
    .value_counts()
)

top_category = top_category.index[0] if len(top_category) > 0 else "None"

st.success(f"""
### 📋 Executive Summary

✅ Total Products Analysed : **{total_products:,}**

🚨 Move Immediately : **{move_now}**

⚠ High Priority : **{high_priority}**

👀 Monitor : **{monitor}**

✅ Healthy : **{healthy}**

💰 Critical Inventory Value :

**₹ {critical_value:,.0f}**

🏆 Highest Risk Brand :

**{top_brand}**

📂 Highest Risk Category :

**{top_category}**
""")

st.info("""
### 💡 AI Recommendation

• Prioritize dispatch of **Move Immediately** products.

• Shift overstock inventory to high-demand warehouses.

• Review products with **No Sales**.

• Increase promotional sales for high-value slow-moving inventory.

• Monitor medium-risk inventory weekly.
""")

# ==========================================
# BUSINESS IMPACT DASHBOARD
# ==========================================

st.divider()
st.subheader("💼 Business Impact Analysis")

critical_products = (
    filtered_df["Predicted Action"] == "Move Immediately"
).sum()

critical_value = filtered_df.loc[
    filtered_df["Predicted Action"] == "Move Immediately",
    "value"
].sum()

high_priority_value = filtered_df.loc[
    filtered_df["Predicted Action"] == "High Priority",
    "value"
].sum()

# Estimated expiry loss (assumption: 20% of critical inventory)
estimated_loss = critical_value * 0.20

# Estimated warehouse space recovered
warehouse_space = (
    critical_products / len(filtered_df)
) * 100

col1, col2 = st.columns(2)

col1.metric(
    "💰 Critical Inventory Value",
    f"₹ {critical_value:,.0f}"
)

col2.metric(
    "📦 Critical Products",
    critical_products
)

col3, col4 = st.columns(2)

col3.metric(
    "📉 Estimated Expiry Loss",
    f"₹ {estimated_loss:,.0f}"
)

col4.metric(
    "🏭 Warehouse Space Recoverable",
    f"{warehouse_space:.1f}%"
)

# ==========================================
# DASHBOARD CHARTS
# ==========================================

st.divider()
st.subheader("📊 AI Dashboard")

col1, col2 = st.columns(2)

# ----------------------------
# Bar Chart
# ----------------------------

action_counts = (
    filtered_df["Predicted Action"]
    .value_counts()
    .reset_index()
)

action_counts.columns = ["Action", "Count"]

fig = px.bar(
    action_counts,
    x="Action",
    y="Count",
    text="Count",
    title="Products by AI Recommendation"
)

col1.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Pie Chart
# ----------------------------

fig2 = px.pie(
    action_counts,
    names="Action",
    values="Count",
    title="Prediction Distribution"
)

col2.plotly_chart(fig2, use_container_width=True)

value_summary = (
    filtered_df.groupby("Predicted Action")["value"]
      .sum()
      .reset_index()
)

fig3 = px.bar(
    value_summary,
    x="Predicted Action",
    y="value",
    text="value",
    title="Inventory Value by AI Recommendation"
)

st.plotly_chart(fig3, use_container_width=True)

st.divider()
st.subheader("🔥 Top 20 Critical Products")

critical_df = filtered_df[filtered_df["Predicted Action"] == "Move Immediately"]

critical_df = critical_df.sort_values(
    by="value",
    ascending=False
)

st.dataframe(
    critical_df[
        [
            "sku_name",
            "brand",
            "value",
            "AI Reason",
            "Recommendation"
        ]
    ].head(20),
    use_container_width=True
)

from io import BytesIO

buffer = BytesIO()

with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    df.to_excel(writer, index=False)

st.download_button(
    label="📥 Download AI Report",
    data=buffer.getvalue(),
    file_name="Warehouse_AI_Report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)