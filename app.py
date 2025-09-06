import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Konfigurasi App
# -------------------------------
st.set_page_config(page_title="Prediksi Nilai Buku Aset", layout="wide")
st.title("📊 Prediksi Nilai Buku Aset Tetap Menggunakan Machine Learning")

# -------------------------------
# Upload dataset
# -------------------------------
uploaded_file = st.file_uploader("Upload Dataset Bersih (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    st.subheader("📂 Preview Dataset")
    st.dataframe(df.head())

    # -------------------------------
    # Pilih fitur dan target
    # -------------------------------
    fitur = [
        'Tahun_Perolehan','Masa_Manfaat_Tahun','Tarif_Penyusutan',
        'Nilai_Perolehan','Biaya_Penyusutan_Bulan',
        'Biaya_Penyusutan_Sampai_Bulan','Akumulasi_Penyusutan'
    ]
    X = df[fitur]
    y = df['Nilai_Buku_Bulan_Ini']

    # -------------------------------
    # Split data
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------------
    # Pilih model
    # -------------------------------
    model_option = st.radio("Pilih Algoritma:", ["Linear Regression", "Random Forest"])

    if model_option == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(random_state=42, n_estimators=100)

    # -------------------------------
    # Training model
    # -------------------------------
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # -------------------------------
    # Evaluasi
    # -------------------------------
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    col1, col2 = st.columns(2)
    col1.metric("📉 RMSE", f"{rmse:,.0f}")
    col2.metric("📈 R² Score", f"{r2:.3f}")

    # -------------------------------
    # Hasil Prediksi vs Aktual
    # -------------------------------
    st.subheader("🔍 Hasil Prediksi vs Aktual (Sample)")
    hasil = pd.DataFrame({'Aktual': y_test, 'Prediksi': y_pred})
    st.dataframe(hasil.head(10))

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax.set_xlabel("Aktual")
    ax.set_ylabel("Prediksi")
    ax.set_title("Aktual vs Prediksi Nilai Buku")
    st.pyplot(fig)

else:
    st.info("⬅️ Silakan upload file dataset bersih (.xlsx)")
