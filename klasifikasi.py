import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

# ------------------------
# Load data & model
# ------------------------
df = pd.read_csv("E-commerce Customer Behavior.csv")

# Handle missing values seperti yang kamu lakukan di Colab
df.dropna(subset=['Satisfaction Level'], inplace=True)
df.dropna(subset=['Average Rating', 'Discount Applied', 'Days Since Last Purchase'], inplace=True)

try:
    model = joblib.load("logistic_regression_model.pkl")
    scaler = joblib.load("scaler_satisfaction.pkl")  # scaler dari training
except:
    model = None
    scaler = None

# ------------------------
# Sidebar Navigasi
# ------------------------
menu = st.sidebar.radio("Pilih Menu", ["Dashboard", "Predict Satisfaction"])

# ------------------------
# MENU 1: DASHBOARD
# ------------------------
if menu == "Dashboard":
    st.title("📊 Dashboard Analisis Kepuasan Pelanggan")

    st.markdown("Berikut adalah visualisasi data berdasarkan analisis EDA dan korelasi terhadap tingkat kepuasan pelanggan.")

    # Pie Chart Kepuasan
    fig1 = px.pie(df, names='Satisfaction Level',
                  title='Distribusi Tingkat Kepuasan Pelanggan',
                  color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig1)

    # Boxplot Rating
    fig2 = px.box(df, x='Satisfaction Level', y='Average Rating',
                  color='Satisfaction Level', title='Rating vs Kepuasan')
    st.plotly_chart(fig2)

    # Boxplot Days Since Last Purchase
    fig3 = px.box(df, x='Satisfaction Level', y='Days Since Last Purchase',
                  color='Satisfaction Level', title='Days Since Last Purchase vs Kepuasan')
    st.plotly_chart(fig3)

    # Countplot Diskon
    fig4 = px.histogram(df, x='Satisfaction Level', color='Discount Applied',
                        barmode='group', title='Diskon vs Kepuasan')
    st.plotly_chart(fig4)

# ------------------------
# MENU 2: PREDIKSI KEPUASAN
# ------------------------
elif menu == "Predict Satisfaction":
    st.title("🧪 Prediksi Kepuasan Pelanggan")

    if model is None or scaler is None:
        st.error("Model atau scaler belum tersedia. Silakan latih dan simpan dulu.")
    else:
        st.markdown("Masukkan data berikut untuk memprediksi apakah customer puas, netral, atau tidak puas.")

        avg_rating = st.slider("Average Rating (1.0 - 5.0)", min_value=1.0, max_value=5.0, step=0.1)
        discount = st.selectbox("Apakah Customer Mendapatkan Diskon?", ["Ya", "Tidak"])
        days_since = st.number_input("Berapa Hari Sejak Pembelian Terakhir?", min_value=0)

        if st.button("Prediksi Sekarang"):
            # Encode input
            discount_val = 1 if discount == "Ya" else 0

            # Buat DataFrame dan scale
            input_data = pd.DataFrame([{
                'Average Rating': avg_rating,
                'Discount Applied': discount_val,
                'Days Since Last Purchase': days_since
            }])
            input_scaled = scaler.transform(input_data)

            # Prediksi
            pred = model.predict(input_scaled)[0]

            # Decode label
            label_map = {0: 'Unsatisfied', 1: 'Neutral', 2: 'Satisfied'}
            st.success(f"Hasil Prediksi: **{label_map[pred]}** 🎉")