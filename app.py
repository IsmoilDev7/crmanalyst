# =========================================================
# IMPORT LIBRARIES
# Import required libraries
# Kerakli kutubxonalarni chaqiramiz
# =========================================================

import pandas as pd                      # Data analysis / Ma'lumotlar tahlili
import numpy as np                       # Numerical calculations / Sonli hisoblar
import streamlit as st                   # Web dashboard / Web interfeys
import plotly.express as px              # Interactive charts / Interaktiv grafiklar
from sklearn.linear_model import LinearRegression  # ML Forecast / Bashorat modeli

# =========================================================
# STREAMLIT PAGE SETTINGS
# Configure page layout
# Sahifa sozlamalari
# =========================================================

st.set_page_config(
    page_title="Sales Analytics & Forecast Dashboard",
    layout="wide"
)

st.title("📊 Sales Analytics, Risk & Forecast Dashboard")
st.caption("Excel upload • Filters • KPI • Risk • Forecast")

# =========================================================
# FILE UPLOAD SECTION
# Upload Excel file from computer
# Excel faylni kompyuterdan yuklash
# =========================================================

uploaded_file = st.file_uploader(
    "📂 Upload Excel file / Excel faylni yuklang",
    type=["xlsx", "xls"]
)

if uploaded_file is None:
    st.warning("❗ Please upload Excel file to start analysis / Analiz uchun Excel yuklang")
    st.stop()

# =========================================================
# LOAD & CLEAN DATA
# Read Excel and preprocess data
# Excel o‘qish va tozalash
# =========================================================

df = pd.read_excel(uploaded_file)

df['start date'] = pd.to_datetime(
    df['start date'],
    dayfirst=True,
    errors='coerce'
)

df['Sum'] = pd.to_numeric(
    df['Sum'],
    errors='coerce'
).fillna(0)

# =========================================================
# SIDEBAR FILTERS
# Responsible and Date filters
# Mas'ul shaxs va Sana filtrlari
# =========================================================

st.sidebar.header("🔎 Filters / Filtrlar")

responsible_filter = st.sidebar.multiselect(
    "Responsible / Mas'ul shaxs",
    options=df['Responsible'].unique(),
    default=df['Responsible'].unique()
)

date_filter = st.sidebar.date_input(
    "Date range / Sana oralig‘i",
    [df['start date'].min(), df['start date'].max()]
)

df_f = df[
    (df['Responsible'].isin(responsible_filter)) &
    (df['start date'].between(
        pd.to_datetime(date_filter[0]),
        pd.to_datetime(date_filter[1])
    ))
]

# =========================================================
# KPI METRICS
# Main business indicators
# Asosiy biznes ko‘rsatkichlari
# =========================================================

total_deals = len(df_f)                                   # Total deals / Jami bitimlar
total_sum = df_f['Sum'].sum()                             # Total revenue / Jami summa
success_sum = df_f[df_f['Transaction stage']=="Success"]['Sum'].sum()
debtors_sum = df_f[df_f['Transaction stage']=="Debtors"]['Sum'].sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("📦 Deals", total_deals)
col2.metric("💰 Revenue", f"{total_sum:,.0f}")
col3.metric("✅ Success", f"{success_sum:,.0f}")
col4.metric("⚠️ Debtors", f"{debtors_sum:,.0f}")

# =========================================================
# RESPONSIBLE PERFORMANCE ANALYSIS
# Sales by responsible person
# Mas'ul shaxslar bo‘yicha tahlil
# =========================================================

st.subheader("👤 Responsible Performance / Mas'ul shaxslar tahlili")

resp_summary = (
    df_f.groupby('Responsible')['Sum']
    .sum()
    .reset_index()
    .sort_values(by='Sum', ascending=False)
)

fig_resp = px.bar(
    resp_summary,
    x='Responsible',
    y='Sum',
    title="Revenue by Responsible / Mas'ul shaxslar bo‘yicha tushum"
)

st.plotly_chart(fig_resp, use_container_width=True)

# =========================================================
# TRANSACTION STAGE ANALYSIS
# Funnel & risk analysis
# Bitim bosqichlari va risk
# =========================================================

st.subheader("📌 Transaction Stage Analysis / Bitim bosqichlari")

stage_summary = (
    df_f.groupby('Transaction stage')['Sum']
    .sum()
    .reset_index()
)

fig_stage = px.pie(
    stage_summary,
    names='Transaction stage',
    values='Sum',
    title="Stage Distribution / Bosqichlar taqsimoti"
)

st.plotly_chart(fig_stage, use_container_width=True)

# =========================================================
# TIME SERIES ANALYSIS
# Daily revenue trend
# Kunlik tushum trendi
# =========================================================

st.subheader("📈 Revenue Over Time / Vaqt bo‘yicha tushum")

time_series = (
    df_f.groupby('start date')['Sum']
    .sum()
    .reset_index()
)

fig_time = px.line(
    time_series,
    x='start date',
    y='Sum',
    markers=True,
    title="Daily Revenue Trend / Kunlik tushum grafigi"
)

st.plotly_chart(fig_time, use_container_width=True)

# =========================================================
# RISK ANALYSIS
# Debtors risk indicator
# Qarzdorlik riski
# =========================================================

st.subheader("🚨 Risk Analysis / Risk tahlili")

df_f['Risk_Flag'] = df_f['Transaction stage'].apply(
    lambda x: "High Risk" if x == "Debtors" else "Normal"
)

risk_table = (
    df_f.groupby('Risk_Flag')['Sum']
    .sum()
    .reset_index()
)

st.dataframe(risk_table, use_container_width=True)

# =========================================================
# FORECASTING (ML)
# Future revenue prediction
# Kelajak tushum bashorati
# =========================================================

st.subheader("🔮 Revenue Forecast (Next 14 Days) / 14 kunlik bashorat")

if len(time_series) >= 2:
    time_series['day_index'] = np.arange(len(time_series))

    X = time_series[['day_index']]
    y = time_series['Sum']

    model = LinearRegression()
    model.fit(X, y)

    future_days = 14
    future_index = np.arange(len(time_series), len(time_series) + future_days).reshape(-1, 1)
    forecast_values = model.predict(future_index)

    forecast_df = pd.DataFrame({
        "Day": range(1, future_days + 1),
        "Forecast_Sum": forecast_values
    })

    fig_forecast = px.line(
        forecast_df,
        x='Day',
        y='Forecast_Sum',
        markers=True,
        title="Forecasted Revenue / Bashorat qilingan tushum"
    )

    st.plotly_chart(fig_forecast, use_container_width=True)
else:
    st.info("Not enough data for forecast / Bashorat uchun ma'lumot yetarli emas")

# =========================================================
# RAW DATA VIEW
# Show filtered table
# Filtrlangan jadval
# =========================================================

st.subheader("📄 Filtered Data / Filtrlangan ma'lumotlar")
st.dataframe(df_f, use_container_width=True)
