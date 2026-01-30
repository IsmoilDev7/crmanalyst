# ============================================================
# SALES ANALYTICS & FORECAST DASHBOARD
# Streamlit + Excel + ML
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------
# PAGE CONFIG
# Sahifa sozlamalari
# ------------------------------------------------------------
st.set_page_config(page_title="Sales Analytics Dashboard", layout="wide")
st.title("📊 Sales Analytics, Risk & Forecast Dashboard")

# ------------------------------------------------------------
# FILE UPLOAD
# Excel faylni yuklash
# ------------------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Excel file (xlsx format)",
    type=["xlsx"]
)

if uploaded_file is None:
    st.info("⬆️ Analizni boshlash uchun Excel fayl yuklang")
    st.stop()

# ------------------------------------------------------------
# LOAD & CLEAN DATA
# Ma'lumotlarni o‘qish va tozalash
# ------------------------------------------------------------
df = pd.read_excel(uploaded_file)

df["start date"] = pd.to_datetime(
    df["start date"],
    dayfirst=True,
    errors="coerce"
)

df["Sum"] = pd.to_numeric(
    df["Sum"],
    errors="coerce"
).fillna(0)

# ------------------------------------------------------------
# SIDEBAR FILTERS
# Filtrlar
# ------------------------------------------------------------
st.sidebar.header("🔎 Filters")

responsible_filter = st.sidebar.multiselect(
    "Responsible / Mas'ul shaxs",
    options=df["Responsible"].unique(),
    default=df["Responsible"].unique()
)

date_filter = st.sidebar.date_input(
    "Date range / Sana oralig‘i",
    [df["start date"].min(), df["start date"].max()]
)

df_f = df[
    (df["Responsible"].isin(responsible_filter)) &
    (df["start date"].between(
        pd.to_datetime(date_filter[0]),
        pd.to_datetime(date_filter[1])
    ))
]

# ------------------------------------------------------------
# KPI METRICS
# Asosiy biznes ko‘rsatkichlari
# ------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("📦 Deals", len(df_f))
col2.metric("💰 Total Revenue", f"{df_f['Sum'].sum():,.0f}")
col3.metric(
    "✅ Success Revenue",
    f"{df_f[df_f['Transaction stage']=='Success']['Sum'].sum():,.0f}"
)
col4.metric(
    "⚠️ Debtors Amount",
    f"{df_f[df_f['Transaction stage']=='Debtors']['Sum'].sum():,.0f}"
)

# ------------------------------------------------------------
# RESPONSIBLE PERFORMANCE
# Mas'ul shaxslar bo‘yicha analiz
# ------------------------------------------------------------
st.subheader("👤 Responsible Performance Analysis")

resp_df = (
    df_f.groupby("Responsible")["Sum"]
    .sum()
    .reset_index()
    .sort_values(by="Sum", ascending=False)
)

fig_resp = px.bar(
    resp_df,
    x="Responsible",
    y="Sum",
    title="Revenue by Responsible"
)

st.plotly_chart(fig_resp, use_container_width=True)

# ------------------------------------------------------------
# TRANSACTION STAGE ANALYSIS
# Bitim bosqichlari tahlili
# ------------------------------------------------------------
st.subheader("📌 Transaction Stage Analysis")

stage_df = (
    df_f.groupby("Transaction stage")["Sum"]
    .sum()
    .reset_index()
)

fig_stage = px.pie(
    stage_df,
    names="Transaction stage",
    values="Sum",
    title="Stage Distribution"
)

st.plotly_chart(fig_stage, use_container_width=True)

# ------------------------------------------------------------
# TIME SERIES ANALYSIS
# Vaqt bo‘yicha tushum
# ------------------------------------------------------------
st.subheader("📈 Revenue Over Time")

ts = (
    df_f.groupby("start date")["Sum"]
    .sum()
    .reset_index()
)

fig_ts = px.line(
    ts,
    x="start date",
    y="Sum",
    markers=True,
    title="Daily Revenue Trend"
)

st.plotly_chart(fig_ts, use_container_width=True)

# ------------------------------------------------------------
# ADDITIONAL ANALYSIS: GROWTH RATE
# Qo‘shimcha analiz: O‘sish foizi
# ------------------------------------------------------------
st.subheader("📊 Revenue Growth Rate")

ts["Growth %"] = ts["Sum"].pct_change() * 100

fig_growth = px.bar(
    ts,
    x="start date",
    y="Growth %",
    title="Daily Growth Percentage"
)

st.plotly_chart(fig_growth, use_container_width=True)

# ------------------------------------------------------------
# RISK ANALYSIS
# Qarzdorlik riski
# ------------------------------------------------------------
st.subheader("🚨 Risk Analysis")

df_f["Risk Level"] = df_f["Transaction stage"].apply(
    lambda x: "High Risk" if x == "Debtors" else "Normal"
)

risk_df = (
    df_f.groupby("Risk Level")["Sum"]
    .sum()
    .reset_index()
)

st.dataframe(risk_df, use_container_width=True)

# ------------------------------------------------------------
# FORECASTING (ML)
# Kelajak tushum bashorati
# ------------------------------------------------------------
st.subheader("🔮 Revenue Forecast (Next 14 Days)")

if len(ts) >= 2:
    ts["idx"] = np.arange(len(ts))

    X = ts[["idx"]]
    y = ts["Sum"]

    model = LinearRegression()
    model.fit(X, y)

    future_idx = np.arange(len(ts), len(ts) + 14).reshape(-1, 1)
    forecast = model.predict(future_idx)

    forecast_df = pd.DataFrame({
        "Day": range(1, 15),
        "Forecast Revenue": forecast
    })

    fig_forecast = px.line(
        forecast_df,
        x="Day",
        y="Forecast Revenue",
        markers=True,
        title="Forecasted Revenue"
    )

    st.plotly_chart(fig_forecast, use_container_width=True)
else:
    st.warning("Forecast uchun ma'lumot yetarli emas")

# ------------------------------------------------------------
# DATA TABLE
# Yakuniy jadval
# ------------------------------------------------------------
st.subheader("📄 Filtered Data Table")
st.dataframe(df_f, use_container_width=True)
