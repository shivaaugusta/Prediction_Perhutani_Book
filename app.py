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

    # ---------------- Elbow Method ----------------
    inertia = []
    K = range(2, 8)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    ax.plot(K, inertia, "bo-")
    ax.set_xlabel("Jumlah Cluster (k)")
    ax.set_ylabel("Inertia (Total Jarak ke Pusat Cluster)")
    ax.set_title("Elbow Method")
    st.pyplot(fig)

    # Narasi otomatis untuk membantu user
    st.markdown(
        """
        ℹ️ **Cara membaca grafik Elbow Method**:  
        - Grafik akan selalu menurun, karena semakin banyak cluster → data makin terpisah.  
        - Pilih jumlah cluster di **titik siku (elbow)**, yaitu saat grafik mulai melandai.  
        - Dari grafik di atas, kemungkinan elbow ada di sekitar **k=3 atau k=4**.  
        """
    )

    # ---------------- Clustering ----------------
    n_clusters = st.slider("👉 Pilih jumlah cluster:", 2, 6, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    st.subheader("📊 Hasil Clustering")
    st.dataframe(df[["Jenis_Aktiva_Tetap"] + fitur + ["Cluster"]].head(20))

   # ---------------- Visualisasi 2D ----------------
    st.subheader("🎨 Visualisasi Clustering (2 Fitur)")
    
    fig2, ax2 = plt.subplots()
    scatter = ax2.scatter(
        X_scaled[:, 0], X_scaled[:, 1],
        c=df["Cluster"], cmap="viridis", alpha=0.7
    )
    
    ax2.set_xlabel(fitur[0])
    ax2.set_ylabel(fitur[1])
    ax2.set_title("Clustering Berdasarkan 2 Fitur")
    
    # Tambahkan legend untuk tiap cluster
    for cluster_id in sorted(df["Cluster"].unique()):
        ax2.scatter([], [], c=scatter.cmap(scatter.norm(cluster_id)),
                    label=f"Cluster {cluster_id}")
    
    ax2.legend(title="Segmen Aset", bbox_to_anchor=(1.05, 1), loc="upper left")
    
    st.pyplot(fig2)
    
    st.markdown(
        """
        ✨ **Cara membaca visualisasi:**  
        - Setiap titik = 1 aset.  
        - Warna berbeda = cluster berbeda (lihat legend → segmen aset).  
        - Sumbu X = Nilai Perolehan (dinormalisasi).  
        - Sumbu Y = Masa Manfaat Tahun (dinormalisasi).  
        - Kalau warna terpisah jelas → clustering efektif.  
        """
    )

    # ---------------- Ringkasan Statistik ----------------
    st.subheader("📌 Ringkasan Statistik per Cluster")
    cluster_summary = df.groupby("Cluster")[fitur].mean().round(2)
    st.dataframe(cluster_summary)

    st.markdown(
        """
        ✨ **Interpretasi contoh** (bisa berbeda tergantung dataset):  
        - Cluster dengan **Nilai Perolehan tinggi & Penyusutan rendah** biasanya = tanah/gedung.  
        - Cluster dengan **Masa Manfaat pendek & Penyusutan kecil** = peralatan kecil.  
        - Cluster dengan **Nilai Perolehan sedang & Penyusutan rutin** = kendaraan/mesin.  
        """
    )

    # ---------------- Evaluasi ----------------
    score = silhouette_score(X_scaled, df["Cluster"])
    st.success(f"📈 Silhouette Score: {score:.3f} → semakin dekat ke 1, semakin baik pemisahan cluster.")

else:
    st.info("⬅️ Upload dataset Excel bersih untuk mulai clustering.")
