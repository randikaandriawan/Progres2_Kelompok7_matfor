# PROGRAM K-MEANS CLUSTERING
# ANALISIS POLA BELAJAR MAHASISWA

# ==============================================================================
# 1. IMPORT LIBRARY DAN PERSIAPAN DATA
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score



print("\nPROGRAM K-MEANS CLUSTERING")
print("Analisis Pola Belajar Mahasiswa")

# MEMBACA DATASET FROM CSV
data = pd.read_csv("dataset_mahasiswa.csv", sep=';')

# MENAMPILKAN DATASET ASLI
print("\nDATASET MAHASISWA\n")
print(
    f"{'Nama':<15}"
    f"{'Belajar':<10}"
    f"{'Coding':<10}"
    f"{'Menunda':<10}"
    f"{'MediaSosial':<15}"
    f"{'Diskusi':<10}"
    f"{'AI'}"
)
print("-" * 80)

for index, row in data.iterrows():
    print(
        f"{row['Nama']:<15}"
        f"{row['Belajar']:<10}"
        f"{row['Coding']:<10}"
        f"{row['Menunda']:<10}"
        f"{row['MediaSosial']:<15}"
        f"{row['Diskusi']:<10}"
        f"{row['AI']}"
    )

# ==============================================================================
# 2. PRAPROSES DATA (REVERSE SCORING)
# ==============================================================================
print("\nKETERANGAN:")
print("- Menunda dan MediaSosial merupakan atribut negatif")
print("- Nilai tinggi berarti kurang baik")
print("- Kedua atribut dibalik menggunakan rumus: nilai baru = 6 - nilai lama")

data_proses = data.copy()
data_proses['Menunda'] = 6 - data_proses['Menunda']
data_proses['MediaSosial'] = 6 - data_proses['MediaSosial']

# MENGAMBIL ARRAY NILAI ATRIBUT UNTUK KOMPUTASI
X = data_proses[['Belajar', 'Coding', 'Menunda', 'MediaSosial', 'Diskusi', 'AI']].values

# ==============================================================================
# 3. PENENTUAN JUMLAH CLUSTER DAN CENTROID AWAL
# ==============================================================================
# Menentukan koordinat awal secara manual (Knowledge-based Seeding)
centroids = np.array([
    [5, 5, 5, 5, 5, 5],   # Cluster 1 = Aktif
    [3, 3, 3, 3, 4, 4],   # Cluster 2 = Sedang
    [1, 1, 1, 1, 2, 2]    # Cluster 3 = Kurang Konsisten
])

print("\nCENTROID AWAL\n")
for i, c in enumerate(centroids, start=1):
    print(f"Cluster {i} : {c}")

# ==============================================================================
# 4. PROSES ITERASI K-MEANS CLUSTERING
# ==============================================================================
maks_iterasi = 50

for iterasi in range(maks_iterasi):
    print("\n" + "=" * 110)
    print(f"ITERASI {iterasi + 1}")
    print("=" * 110)

    # Menampilkan posisi centroid pada iterasi berjalan
    print("\n[POSISI CENTROID]")
    print("-" * 110)
    for i, c in enumerate(centroids, start=1):
        print(
            f"C{i} : "
            f"{c[0]:6.2f} "
            f"{c[1]:6.2f} "
            f"{c[2]:6.2f} "
            f"{c[3]:6.2f} "
            f"{c[4]:6.2f} "
            f"{c[5]:6.2f}"
        )

    cluster = []
    semua_jarak = []

    # Menghitung jarak Euclidean data ke setiap centroid
    for data_ke in X:
        jarak = []
        for centroid in centroids:
            d = sqrt(
                (data_ke[0] - centroid[0])**2 +
                (data_ke[1] - centroid[1])**2 +
                (data_ke[2] - centroid[2])**2 +
                (data_ke[3] - centroid[3])**2 +
                (data_ke[4] - centroid[4])**2 +
                (data_ke[5] - centroid[5])**2
            )
            jarak.append(d)

        semua_jarak.append(jarak.copy())
        cluster_terdekat = jarak.index(min(jarak)) + 1
        cluster.append(cluster_terdekat)

    data['Cluster'] = cluster

    # Menampilkan tabel penempatan cluster sementara
    print("\n[HASIL CLUSTER SEMENTARA]")
    print("-" * 110)
    print(
        f"{'No':<4}"
        f"{'Nama':<25}"
        f"{'C1':<10}"
        f"{'C2':<10}"
        f"{'C3':<10}"
        f"{'Cluster'}"
    )
    print("-" * 110)

    for i, row in data.iterrows():
        print(
            f"{i + 1:<4}"
            f"{row['Nama']:<25}"
            f"{semua_jarak[i][0]:<10.2f}"
            f"{semua_jarak[i][1]:<10.2f}"
            f"{semua_jarak[i][2]:<10.2f}"
            f"C{row['Cluster']}"
        )

    # Rekap jumlah anggota cluster sementara
    print("\n[REKAP CLUSTER]")
    print("-" * 110)
    jumlah_iterasi = data['Cluster'].value_counts().sort_index()
    for nomor_cluster, total in jumlah_iterasi.items():
        print(f"Cluster {nomor_cluster} : {total} mahasiswa")

    # Update posisi centroid baru berdasarkan nilai rata-rata (mean) anggota
    centroid_baru = []
    for i in range(1, 4):
        anggota_cluster = X[np.array(cluster) == i]
        if len(anggota_cluster) == 0:
            rata_rata = centroids[i - 1]
        else:
            rata_rata = anggota_cluster.mean(axis=0)
        centroid_baru.append(rata_rata)
    centroid_baru = np.array(centroid_baru)

    # Menampilkan posisi centroid baru hasil update
    print("\n[CENTROID BARU]")
    print("-" * 110)
    for i, c in enumerate(centroid_baru, start=1):
        print(
            f"C{i} : "
            f"{c[0]:6.2f} "
            f"{c[1]:6.2f} "
            f"{c[2]:6.2f} "
            f"{c[3]:6.2f} "
            f"{c[4]:6.2f} "
            f"{c[5]:6.2f}"
        )

    # Cek konvergensi (apakah posisi cluster sudah stabil dan berhenti berubah)
    if np.allclose(centroids, centroid_baru):
        print("\n" + "=" * 110)
        print("CLUSTER SUDAH STABIL")
        print("=" * 110)
        break

    centroids = centroid_baru

# ==============================================================================
# 5. HASIL AKHIR CLUSTERING
# ==============================================================================
print("\n========== HASIL AKHIR CLUSTERING ==========")
karakteristik = {
    1: "Mahasiswa aktif belajar",
    2: "Mahasiswa dengan pola belajar sedang",
    3: "Mahasiswa kurang konsisten"
}

for i in range(1, 4):
    anggota = data[data['Cluster'] == i]
    print(f"\nCluster {i}")
    print(f"Karakteristik : {karakteristik[i]}")
    print(f"Jumlah anggota : {len(anggota)}")
    print("-" * 30)
    for nama in anggota['Nama']:
        print(nama)

# Tingkat sebaran akhir total anggota
print("\nJUMLAH ANGGOTA CLUSTER\n")
jumlah = data['Cluster'].value_counts().sort_index()
for nomor_cluster, total in jumlah.items():
    print(f"Cluster {nomor_cluster} : {total} mahasiswa")

# Koordinat pusat klaster final
print("\nCENTROID AKHIR\n")
for i, c in enumerate(centroids, start=1):
    print(f"Cluster {i} : {np.round(c, 2)}")

# ==============================================================================
# 6. EVALUASI MODEL
# ==============================================================================
nilai_silhouette = silhouette_score(X, cluster)

print("\n========== EVALUASI MODEL ==========")
print(f"Silhouette Score : {nilai_silhouette:.4f}")

interpretasi = ""
if nilai_silhouette > 0.7:
    interpretasi = "Struktur cluster sangat baik"
elif nilai_silhouette > 0.5:
    interpretasi = "Struktur cluster baik"
elif nilai_silhouette > 0.25:
    interpretasi = "Struktur cluster cukup"
else:
    interpretasi = "Struktur cluster kurang baik"

print(f"Interpretasi     : {interpretasi}")

# Menampilkan statistik deskriptif dataset hasil transformasi
print("\n[STATISTIK DESKRIPTIF DATA PROSES]")
print(data_proses.describe())

# ==============================================================================
# 7. VISUALISASI DATA AWAL DAN SEBARAN HASIL CLUSTER
# ==============================================================================
fitur = ['Belajar', 'Coding', 'Menunda', 'MediaSosial', 'Diskusi', 'AI']
sns.set_theme(style="whitegrid")

# Visualisasi 1: Histogram Distribusi Atribut
plt.figure(figsize=(15, 10))
for i, kolom in enumerate(fitur, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data=data_proses[kolom], bins=5, kde=True, color="skyblue")
    plt.title(f"Distribusi Atribut {kolom}")
    plt.ylabel("Jumlah Mahasiswa")
plt.tight_layout()
plt.show()

# Visualisasi 2: Box Plot Deteksi Outlier
plt.figure(figsize=(12, 6))
sns.boxplot(data=data_proses[fitur], palette="Set2")
plt.title("Visualisasi Box Plot Seluruh Atribut Perilaku Belajar (N = 30)")
plt.ylabel("Skala Likert")
plt.xlabel("Atribut Fitur")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()

# Visualisasi 3: Scatter Plot Hasil Klasterisasi Menggunakan Reduksi Dimensi PCA 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
for i in range(1, 4):
    plt.scatter(
        X_pca[np.array(cluster) == i, 0],
        X_pca[np.array(cluster) == i, 1],
        s=100,
        label=f'Cluster {i}'
    )

plt.title('Visualisasi Hasil Clustering K-Means Berbasis PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()
