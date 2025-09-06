import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Clustering Aset", layout="wide")
st.title("🔍 Clustering Aset Tetap (Segmentasi)")

# Upload dataset
uploaded_file = st.file_uploader("Upload Dataset Bersih (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    st.subheader("📂 Preview Dataset")
    st.dataframe(df.head())

    # Pilih fitur numerik untuk clustering
    fitur = ["Nilai_Perolehan", "Masa_Manfaat_Tahun", "Biaya_Penyusutan_Bulan"]
    X = df[fitur]

    # Normalisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Tentukan jumlah cluster optimal (Elbow Method)
    inertia = []
    K = range(2, 8)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    ax.plot(K, inertia, "bo-")
    ax.set_xlabel("Jumlah Cluster (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")
    st.pyplot(fig)

    # Input jumlah cluster
    n_clusters = st.slider("Pilih jumlah cluster:", 2, 6, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    # Mapping cluster ke nama segmen (contoh awal)
    label_map = {i: f"Segmen {i}" for i in range(n_clusters)}
    df["Cluster_Label"] = df["Cluster"].map(label_map)

    # -------------------------
    # 📊 Ringkasan per Cluster
    # -------------------------
    st.subheader("📌 Ringkasan per Cluster")
    summary = df.groupby("Cluster_Label").agg({
        "Jenis_Aktiva_Tetap": "count",
        "Nilai_Perolehan": "sum",
        "Masa_Manfaat_Tahun": "mean",
        "Biaya_Penyusutan_Bulan": "mean"
    }).rename(columns={
        "Jenis_Aktiva_Tetap": "Jumlah Aset",
        "Nilai_Perolehan": "Total Nilai Perolehan",
        "Masa_Manfaat_Tahun": "Rata-rata Masa Manfaat",
        "Biaya_Penyusutan_Bulan": "Rata-rata Biaya Penyusutan"
    })
    st.dataframe(summary)

    # -------------------------
    # 📊 Hasil Clustering Detail
    # -------------------------
    st.subheader("📊 Detail Hasil Clustering")
    st.dataframe(df[["Jenis_Aktiva_Tetap"] + fitur + ["Cluster_Label"]].head(20))

    # -------------------------
    # 📈 Visualisasi Clustering
    # -------------------------
    st.subheader("📈 Visualisasi Clustering (2D)")

    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.scatterplot(
        x=X_scaled[:,0], y=X_scaled[:,1],
        hue=df["Cluster_Label"], palette="tab10", s=60, ax=ax2
    )
    ax2.set_xlabel(fitur[0])
    ax2.set_ylabel(fitur[1])
    ax2.set_title("Visualisasi Clustering Aset")
    st.pyplot(fig2)

    # Evaluasi dengan silhouette score
    score = silhouette_score(X_scaled, df["Cluster"])
    st.success(f"📊 Silhouette Score: {score:.3f} (semakin mendekati 1 semakin baik)")

else:
    st.info("⬅️ Upload dataset Excel bersih untuk mulai clustering.")
