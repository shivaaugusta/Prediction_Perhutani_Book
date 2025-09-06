import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard Penyusutan Aset", layout="wide")
st.title("📊 Dashboard Analisis & Prediksi Nilai Buku Aset Tetap")

# Upload dataset
uploaded_file = st.file_uploader("Upload File Excel", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    st.subheader("📂 Preview Dataset")
    st.dataframe(df.head())

    # -------------------------
    # Ringkasan Total
    # -------------------------
    st.subheader("📌 Ringkasan Total")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Nilai Perolehan", f"Rp {df['Nilai_Perolehan'].sum():,.0f}")
    col2.metric("Total Penyusutan Bulan", f"Rp {df['Biaya_Penyusutan_Bulan'].sum():,.0f}")
    col3.metric("Total Nilai Buku", f"Rp {df['Nilai_Buku_Bulan_Ini'].sum():,.0f}")

    # -------------------------
    # Prediksi Nilai Buku
    # -------------------------
    st.subheader("🤖 Prediksi Nilai Buku Aset Tetap")

    # Pilih fitur (X) dan target (y)
    fitur = [
        'Tahun_Perolehan',
        'Masa_Manfaat_Tahun',
        'Tarif_Penyusutan',
        'Nilai_Perolehan',
        'Biaya_Penyusutan_Bulan',
        'Biaya_Penyusutan_Sampai_Bulan',
        'Akumulasi_Penyusutan'
    ]

    # Bersihkan data (isi NaN dengan 0)
    df[fitur] = df[fitur].fillna(0)
    df['Nilai_Buku_Bulan_Ini'] = df['Nilai_Buku_Bulan_Ini'].fillna(0)

    X = df[fitur]
    y = df['Nilai_Buku_Bulan_Ini']

    # Scaling untuk model linier
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Pilihan model
    model_option = st.radio("Pilih Algoritma:", ["Linear Regression", "Random Forest"])

    if model_option == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(random_state=42, n_estimators=200)

    # Training
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluasi
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.write(f"📉 **RMSE:** Rp {rmse:,.0f}")
    st.write(f"📈 **R² Score:** {r2:.3f}")

    # Hasil sample prediksi
    st.subheader("🔍 Hasil Prediksi vs Aktual (Sample)")
    hasil = pd.DataFrame({'Aktual': y_test.values[:10], 'Prediksi': y_pred[:10]})
    st.dataframe(hasil)

    # Plot hasil prediksi
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax.set_xlabel("Aktual")
    ax.set_ylabel("Prediksi")
    ax.set_title("Aktual vs Prediksi")
    st.pyplot(fig)

else:
    st.info("⬅️ Upload dataset Excel untuk mulai analisis & prediksi.")
