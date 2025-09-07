import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

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

    # Tentukan jumlah cluster optimal (Elbow)
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

    st.subheader("📊 Hasil Clustering")
    st.dataframe(df[["Jenis_Aktiva_Tetap"] + fitur + ["Cluster"]].head(20))

    # Visualisasi 2D (pakai 2 fitur utama)
    fig2, ax2 = plt.subplots()
    scatter = ax2.scatter(X_scaled[:,0], X_scaled[:,1], c=df["Cluster"], cmap="viridis", alpha=0.7)
    ax2.set_xlabel(fitur[0])
    ax2.set_ylabel(fitur[1])
    ax2.set_title("Visualisasi Clustering (2 Fitur)")
    plt.colorbar(scatter, ax=ax2, label="Cluster")
    st.pyplot(fig2)

    # Evaluasi dengan silhouette score
    score = silhouette_score(X_scaled, df["Cluster"])
    st.success(f"Silhouette Score: {score:.3f}")

    # ===============================
    # 🔎 Ringkasan & Interpretasi
    # ===============================
    st.subheader("📊 Ringkasan Statistik per Cluster")
    cluster_summary = df.groupby("Cluster")[fitur].mean().round(2)
    st.dataframe(cluster_summary)

    st.subheader("📝 Interpretasi Cluster")
    for c in cluster_summary.index:
        avg_val = cluster_summary.loc[c, "Nilai_Perolehan"]
        avg_manfaat = cluster_summary.loc[c, "Masa_Manfaat_Tahun"]
        avg_penyusutan = cluster_summary.loc[c, "Biaya_Penyusutan_Bulan"]

        if avg_val < 1e8:
            tipe = "Aset bernilai kecil (peralatan kantor, inventaris sederhana)"
        elif avg_val < 1e9:
            tipe = "Aset menengah (kendaraan, mesin kecil)"
        else:
            tipe = "Aset bernilai sangat besar (tanah, gedung)"

        st.markdown(f"""
        **Cluster {c}**  
        - 📌 Rata-rata Nilai Perolehan: Rp {avg_val:,.0f}  
        - ⏳ Rata-rata Masa Manfaat: {avg_manfaat:.1f} tahun  
        - 💸 Rata-rata Biaya Penyusutan Bulanan: Rp {avg_penyusutan:,.0f}  
        - ✅ **Interpretasi:** {tipe}
        """)

else:
    st.info("⬅️ Upload dataset Excel bersih untuk mulai clustering.")
