import pandas as pd
import numpy as np
from math import sqrt

#Orang 1

# PROGRAM K-MEANS CLUSTERING
# ANALISIS POLA BELAJAR MAHASISWA

print("\nPROGRAM K-MEANS CLUSTERING")
print("Analisis Pola Belajar Mahasiswa")

# MEMBACA DATASET

data = pd.read_csv("dataset_mahasiswa.csv", sep=';')

# MENAMPILKAN DATASET

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

# PREPROCESSING DATA

print("\nKETERANGAN:")
print("- Menunda dan MediaSosial merupakan atribut negatif")
print("- Nilai tinggi berarti kurang baik")
print("- Kedua atribut dibalik menggunakan rumus:")
print("  nilai baru = 6 - nilai lama")

#Orang 2

data_proses = data.copy()

data_proses['Menunda'] = 6 - data_proses['Menunda']
data_proses['MediaSosial'] = 6 - data_proses['MediaSosial']

# MENGAMBIL ATRIBUT

X = data_proses[['Belajar','Coding','Menunda','MediaSosial','Diskusi','AI']].values

# MENENTUKAN CENTROID AWAL

centroids = np.array([
    [5, 5, 5, 5, 5, 5],  # Cluster 1 = aktif
    [3, 3, 3, 3, 4, 4],  # Cluster 2 = sedang
    [1, 1, 1, 1, 2, 2]   # Cluster 3 = kurang konsisten
])

print("\nCENTROID AWAL\n")

for i, c in enumerate(centroids, start=1):
    print(f"Cluster {i} : {c}")

# PROSES ITERASI K-MEANS

maks_iterasi = 10

for iterasi in range(maks_iterasi):
    print(f"\nITERASI KE-{iterasi + 1}")
    cluster = []

    # MENGHITUNG JARAK KE SETIAP CENTROID

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

        cluster_terdekat = jarak.index(min(jarak)) + 1
        cluster.append(cluster_terdekat)

    # MENAMPILKAN HASIL CLUSTER SEMENTARA

    data['Cluster'] = cluster
    print("\nHASIL CLUSTER SEMENTARA\n")

    for index, row in data.iterrows():
        print(f"{row['Nama']:<15} -> Cluster {row['Cluster']}")

#orang 3

    # UPDATE CENTROID

    centroid_baru = []

    for i in range(1, 4):
        anggota_cluster = X[np.array(cluster) == i]

        # Jika cluster kosong
        if len(anggota_cluster) == 0:
            rata_rata = centroids[i - 1]
        else:
            rata_rata = anggota_cluster.mean(axis=0)
        centroid_baru.append(rata_rata)
    centroid_baru = np.array(centroid_baru)

    # MENAMPILKAN CENTROID BARU
   


    print("\nCENTROID BARU\n")

    for i, c in enumerate(centroid_baru, start=1):
        print(f"Cluster {i} : {np.round(c, 2)}")

    # CEK APAKAH SUDAH STABIL

    if np.allclose(centroids, centroid_baru):
        print("\nCluster sudah stabil")
        break

    centroids = centroid_baru

#Orang 4

# HASIL AKHIR CLUSTERING

print("\nHASIL AKHIR CLUSTERING")

for i in range(1, 4):
    print(f"\nCluster {i}")

    if i == 1:
        print("Karakteristik : Mahasiswa aktif belajar")
    elif i == 2:
        print("Karakteristik : Mahasiswa dengan pola belajar sedang")
    elif i == 3:
        print("Karakteristik : Mahasiswa kurang konsisten")

    anggota = data[data['Cluster'] == i]
    print()

    for nama in anggota['Nama']:
        print(f"- {nama}")

# JUMLAH ANGGOTA CLUSTER

print("\nJUMLAH ANGGOTA CLUSTER\n")

jumlah = data['Cluster'].value_counts().sort_index()
for cluster, total in jumlah.items():
    print(f"Cluster {cluster} : {total} mahasiswa")

# CENTROID AKHIR

print("\nCENTROID AKHIR\n")
for i, c in enumerate(centroids, start=1):
    print(f"Cluster {i} : {np.round(c, 2)}")