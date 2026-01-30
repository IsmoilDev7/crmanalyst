# ============================================================
# SALES ANALYTICS & FORECAST DASHBOARD (SENIOR LEVEL)
# Currency-aware • Time filter 2024–2026 • Responsible & Source
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------
# PAGE CONFIGURATION
# Page settings
# Sahifa sozlamalari
# ------------------------------------------------------------
st.set_page_config(page_title="Sales Analytics Dashboard", layout="wide")
st.title("📊 Sales Analytics, Risk & Forecast Dashboard")

# ------------------------------------------------------------
# GLOBAL SETTINGS
# Exchange rate settings
# Valyuta kursi sozlamasi
# ------------------------------------------------------------
USD_TO_UZS = st.sidebar.number_input(
    "USD → UZS exchange rate / Dollar → So'm kursi",
    value=12500,
    step=100
)

# ------------------------------------------------------------
# FILE UPLOAD
# Upload Excel file
# Excel faylni yuklash
# ------------------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Excel file (xlsx)",
    type=["xlsx"]
)

if uploaded_file is None:
    st.info("⬆️ Upload Excel file to start analysis / Analizni boshlash uchun Excel yuklang")
    st.stop()

# ------------------------------------------------------------
# LOAD & PREPROCESS DATA
# Read and clean data
# Ma'lumotlarni o‘qish va tozalash
# ------------------------------------------------------------
df = pd.read_excel(uploaded_file)

df["start date"] = pd.to_datetime(
    df["start date"],
    dayfirst=True,
    errors="coerce"
)

df["Sum"] = pd.to_numeric(df["Sum"], errors="coerce").fillna(0)

# ------------------------------------------------------------
# CURRENCY NORMALIZATION
# Convert all amounts to UZS
# Barcha summalarni so‘mga o‘tkazish
# ------------------------------------------------------------
df["Sum_UZS"] = np.where(
    df["Currency"].str.lower() == "dollar",
    df["Sum"] * USD_TO_UZS,
    df["Sum"]
)

# ------------------------------------------------------------
# DATE FILTER (2024–2026)
# Time filtering
# Vaqt bo‘yicha filtr
# ------------------------------------------------------------
df = df[
    (df["start date"].dt.year >= 2024) &
    (df["start date"].dt.year <= 2026)
]

# ------------------------------------------------------------
# SIDEBAR FILTERS
# Responsible & Source filters
# Mas'ul shaxs va manba filtrlari
# ------------------------------------------------------------
st.sidebar.header("🔎 Filters")

responsible_filter = st.sidebar.multiselect(
    "Responsible / Mas'ul shaxs",
    options=sorted(df["Responsible"].dropna().unique()),
    default=sorted(df["Responsible"].dropna().unique())
)

source_filter = st.sidebar.multiselect(
    "Source / Manba",
    options=sorted(df["Source"].dropna().unique()),
    default=sorted(df["Source"].dropna().unique())
)

date_filter = st.sidebar.date_input(
    "Date range / Sana oralig‘i",
    [pd.to_datetime("2024-01-01"), pd.to_datetime("2026-12-31")]
)

df_f = df[
    (df["Responsible"].isin(responsible_filter)) &
    (df["Source"].isin(source_filter)) &
    (df["start date"].between(
        pd.to_datetime(date_filter[0]),
        pd.to_datetime(date_filter[1])
    ))
]

# ------------------------------------------------------------
# KPI METRICS
# Key business indicators
# Asosiy biznes ko‘rsatkichlari
# ------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("📦 Deals", f"{len(df_f):,}")
col2.metric("💰 Total Revenue (UZS)", f"{df_f['Sum_UZS'].sum():,.0f}")
col3.metric(
    "✅ Success Revenue",
    f"{df_f[df_f['Transaction stage']=='Success']['Sum_UZS'].sum():,.0f}"
)
col4.metric(
    "⚠️ Debtors Amount",
    f"{df_f[df_f['Transaction stage']=='Debtors']['Sum_UZS'].sum():,.0f}"
)

# ------------------------------------------------------------
# RESPONSIBLE × SOURCE ANALYSIS
# Who sold how much from which source
# Kim qaysi manbadan qancha sotdi
# ------------------------------------------------------------
st.subheader("👤 Responsible × Source Revenue Analysis")

resp_source_df = (
    df_f.groupby(["Responsible", "Source"])["Sum_UZS"]
    .sum()
    .reset_index()
    .sort_values(by="Sum_UZS", ascending=False)
)

fig_resp_source = px.bar(
    resp_source_df,
    x="Responsible",
    y="Sum_UZS",
    color="Source",
    title="Revenue by Responsible and Source"
)

fig_resp_source.update_yaxes(tickformat=",")
st.plotly_chart(fig_resp_source, use_container_width=True)

# ------------------------------------------------------------
# TRANSACTION STAGE ANALYSIS
# Deal stages & funnel
# Bitim bosqichlari tahlili
# ------------------------------------------------------------
st.subheader("📌 Transaction Stage Analysis")

stage_df = (
    df_f.groupby("Transaction stage")["Sum_UZS"]
    .sum()
    .reset_index()
)

fig_stage = px.pie(
    stage_df,
    names="Transaction stage",
    values="Sum_UZS",
    title="Stage Distribution (UZS)"
)

st.plotly_chart(fig_stage, use_container_width=True)

# ------------------------------------------------------------
# TIME SERIES & GROWTH
# Revenue trend and growth rate
# Vaqt bo‘yicha tushum va o‘sish
# ------------------------------------------------------------
st.subheader("📈 Revenue Trend & Growth")

ts = (
    df_f.groupby("start date")["Sum_UZS"]
    .sum()
    .reset_index()
    .sort_values("start date")
)

ts["Growth %"] = ts["Sum_UZS"].pct_change() * 100

fig_ts = px.line(
    ts,
    x="start date",
    y="Sum_UZS",
    markers=True,
    title="Revenue Over Time (UZS)"
)

fig_ts.update_yaxes(tickformat=",")
st.plotly_chart(fig_ts, use_container_width=True)

fig_growth = px.bar(
    ts,
    x="start date",
    y="Growth %",
    title="Revenue Growth Rate (%)"
)

fig_growth.update_yaxes(tickformat=".2f")
st.plotly_chart(fig_growth, use_container_width=True)

# ------------------------------------------------------------
# RISK ANALYSIS
# Debtors monitoring
# Qarzdorlik riski
# ------------------------------------------------------------
st.subheader("🚨 Risk Analysis (Debtors)")

risk_df = (
    df_f[df_f["Transaction stage"] == "Debtors"]
    .groupby("Responsible")["Sum_UZS"]
    .sum()
    .reset_index()
    .sort_values(by="Sum_UZS", ascending=False)
)

risk_df["Sum_UZS"] = risk_df["Sum_UZS"].map("{:,.0f}".format)
st.dataframe(risk_df, use_container_width=True)

# ------------------------------------------------------------
# FORECASTING (ML)
# Future revenue prediction
# Kelajak tushum bashorati
# ------------------------------------------------------------
st.subheader("🔮 Revenue Forecast (Next 14 Days)")

if len(ts) >= 3:
    ts["idx"] = np.arange(len(ts))

    X = ts[["idx"]]
    y = ts["Sum_UZS"]

    model = LinearRegression()
    model.fit(X, y)

    future_idx = np.arange(len(ts), len(ts) + 14).reshape(-1, 1)
    forecast = model.predict(future_idx)

    forecast_df = pd.DataFrame({
        "Day": range(1, 15),
        "Forecast Revenue (UZS)": forecast
    })

    fig_forecast = px.line(
        forecast_df,
        x="Day",
        y="Forecast Revenue (UZS)",
        markers=True,
        title="14-Day Revenue Forecast"
    )

    fig_forecast.update_yaxes(tickformat=",")
    st.plotly_chart(fig_forecast, use_container_width=True)
else:
    st.warning("Not enough data for forecast / Bashorat uchun ma'lumot yetarli emas")

# ------------------------------------------------------------
# FINAL DATA VIEW
# Filtered dataset
# Filtrlangan ma'lumotlar
# ------------------------------------------------------------
st.subheader("📄 Filtered Data")

df_f_display = df_f.copy()
df_f_display["Sum_UZS"] = df_f_display["Sum_UZS"].map("{:,.0f}".format)

st.dataframe(df_f_display, use_container_width=True)
