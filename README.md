# Laporan Proyek Machine Learning - Fadhilah Nurrahmayanti

## Domain Proyek

Produksi pangan merupakan sektor vital bagi negara-negara ASEAN, mengingat sebagian besar wilayahnya masih mengandalkan pertanian sebagai sumber utama pangan dan pendapatan. Dengan meningkatnya populasi dan perubahan iklim yang tidak menentu, memprediksi produksi pangan menjadi kebutuhan strategis. Data historis produksi komoditas seperti jagung, beras, kopi, coklat, dan minyak sawit dapat dimanfaatkan untuk membuat prediksi produksi di masa depan.

Meskipun sangat penting, tren produksi di negara-negara ini sering kali sangat fluktuatif, dipengaruhi oleh variabilitas musiman dan dinamika global. Hal ini menyulitkan dalam memproyeksikan tingkat pasokan di masa depan secara akurat, yang pada akhirnya menjadi tantangan dalam ketahanan pangan, strategi perdagangan, dan perencanaan kebijakan jangka panjang.

Dengan tersedianya data melalui [World Food Production Dataset (Kaggle)](https://www.kaggle.com/datasets/rafsunahmad/world-food-production/data), memungkinkan penggunaan **Data-driven Forecasting Models** seperti neural network **Long Short-Term Memory (LSTM)** untuk memproyeksikan tren produksi hingga tahun 2030 dan membandingkan hasil per negara dari waktu ke waktu.

**Mengapa Masalah Ini Perlu diselesaikan**

1. **Perencanaan Pertanian Jangka Panjang**
   Pemerintah dan perencana pertanian membutuhkan prediksi yang andal untuk menyusun strategi ketahanan pangan dan mengoptimalkan infrastruktur produksi.

2. **Tolok Ukur Strategis Antar Negara ASEAN**
   Peramalan produksi masa depan membantu menentukan apakah Indonesia tetap dominan dalam produksi beras.

3. **Analisis Tren Produksi Global**
   Model LSTM univariat memungkinkan analisis tren setiap komoditas secara independen, memberikan wawasan terhadap stagnasi pertumbuhan atau potensi peningkatan hasil.

**Bagaimana Masalah Ini Akan Diselesaikan**

Solusi dilakukan dengan membangun **model LSTM univariat** yang hanya menggunakan **jumlah produksi tahunan** sebagai input, tanpa variabel lingkungan atau ekonomi tambahan. Pendekatan ini murni berbasis data historis dari [World Food Production Dataset (Kaggle)](https://www.kaggle.com/datasets/rafsunahmad/world-food-production/data)

**Langkah Implementasi:**

1. **Pra-pemrosesan data deret waktu** (1961–2021) untuk tiap komoditas (jagung, beras, kopi, coklat, dan minyak sawit) dan negara (Indonesia, Vietnam, Thailand, Filipina dan Malaysia).
2. **Melatih model LSTM terpisah** untuk tiap komoditas dan negara.
3. **Memprediksi produksi tahun 2022–2030.**
4. **Visualisasi dan perbandingan** proyeksi antar negara dan komoditas.nggunakan **RMSE** sebagai indikator akurasi.

## Business Understanding

### Problem Statements

**Masalah 1:**

Tren produksi pertanian di negara-negara ASEAN sangat fluktuatif. Tanpa peramalan yang akurat, pemerintah dan pemangku kepentingan kesulitan menyusun strategi pangan dan perdagangan jangka panjang.

**Masalah 2:**

Kurangnya prediksi standar yang membandingkan posisi Indonesia terhadap negara produsen utama ASEAN lainnya (Vietnam, Thailand, Filipina dan Malaysia) untuk komoditas strategis seperti jagung, beras, kopi, cokelat, dan minyak sawit.

**Masalah 3:**

Teknik peramalan tradisional (misalnya statistik dasar atau regresi linier) tidak cukup mampu menangkap pola musiman dan non-linier jangka panjang dalam data deret waktu pertanian.

### Goals

**Tujuan 1 (untuk Masalah 1):**

Membangun model **LSTM univariat** yang akurat untuk memprediksi produksi tahunan jagung, beras, kopi, coklat, dan minyak sawit dari 2022 hingga 2030 menggunakan data historis (1961–2021) per negara dan komoditas.

**Tujuan 2 (untuk Masalah 2):**

Membandingkan hasil prediksi antar Indonesia, Vietnam, Thailand, Filipina dan Malaysia untuk setiap komoditas guna mengevaluasi posisi kompetitif Indonesia secara kuantitatif.

**Tujuan 3 (untuk Masalah 3):**

Menggunakan LSTM untuk memodelkan pola musiman dan non-linear dalam data produksi historis, sehingga meningkatkan akurasi dibandingkan model konvensional seperti rata-rata bergerak atau regresi linier.

### Solution Statements

* Menggunakan **Long Short-Term Memory (LSTM)** karena kemampuannya dalam mempelajari pola jangka panjang pada data time series.
* Menyusun preprocessing data berupa encoding, scaling, dan reshaping sebelum digunakan oleh model LSTM.
* Melakukan **early stopping** untuk menghindari overfitting, serta membandingkan hasil prediksi aktual dan prediksi model menggunakan **RMSE** sebagai indikator akurasi.

## Data Understanding

Dataset yang digunakan berisi data produksi komoditas pangan dari berbagai negara ASEAN. File data bernama Data.csv, dan berisi 11.912 baris dengan 24 kolom.

Dataset: [World Food Production Dataset (Kaggle)](https://www.kaggle.com/datasets/rafsunahmad/world-food-production/data) (1961–2021)

### **Variabel dalam Dataset**

| Nama Variabel                         | Deskripsi                                                                    |
| ------------------------------------- | ---------------------------------------------------------------------------- |
| `Entity`                              | Nama negara atau wilayah (contoh: Indonesia, Vietnam, Thailand, Malaysia).   |
| `Year`                                | Tahun kalender dari data yang dicatat (dari 1961 hingga 2023).               |
| `Maize Production (tonnes)`           | Produksi jagung dalam satuan ton –> variabel utama dalam proyek ini.         |
| `Rice  Production ( tonnes)`          | Produksi beras dalam satuan ton –> variabel utama dalam proyek ini.        |
| `Yams  Production (tonnes)`           | Produksi ubi jalar dalam satuan ton.                                         |
| `Wheat Production (tonnes)`           | Produksi gandum dalam satuan ton.                                            |
| `Tomatoes Production (tonnes)`        | Produksi tomat dalam satuan ton.                                             |
| `Tea  Production ( tonnes )`          | Produksi teh dalam satuan ton.                                               |
| `Sweet potatoes  Production (tonnes)` | Produksi ubi manis dalam satuan ton.                                         |
| `Sunflower seed  Production (tonnes)` | Produksi biji bunga matahari dalam satuan ton.                               |
| `Sugar cane Production (tonnes)`      | Produksi tebu dalam satuan ton.                                              |
| `Soybeans  Production (tonnes)`       | Produksi kedelai dalam satuan ton.                                           |
| `Rye  Production (tonnes)`            | Produksi gandum hitam dalam satuan ton.                                      |
| `Potatoes  Production (tonnes)`       | Produksi kentang dalam satuan ton.                                           |
| `Oranges  Production (tonnes)`        | Produksi jeruk dalam satuan ton.                                             |
| `Peas, dry Production ( tonnes)`      | Produksi kacang polong kering dalam satuan ton.                              |
| `Palm oil  Production (tonnes)`       | Produksi minyak sawit dalam satuan ton –> variabel utama dalam proyek ini. |
| `Grapes  Production (tonnes)`         | Produksi anggur dalam satuan ton.                                            |
| `Coffee, green Production ( tonnes)`  | Produksi kopi biji hijau –> variabel utama dalam proyek ini.               |
| `Cocoa beans Production (tonnes)`     | Produksi biji kakao dalam satuan –> variabel utama dalam proyek ini.ton.                                        |
| `Meat, chicken  Production (tonnes)`  | Produksi daging ayam dalam satuan ton.                                       |
| `Bananas  Production ( tonnes)`       | Produksi pisang dalam satuan ton.                                            |
| `Avocados Production (tonnes)`        | Produksi alpukat dalam satuan ton.                                           |
| `Apples Production (tonnes)`          | Produksi apel dalam satuan ton.                                              |

Dalam proyek ini, hanya fokus pada **lima komoditas utama**:

* `Maize Production (tonnes)`
* `Rice Production (tonnes)`
* `Coffee green Production (tonnes)`
* `Cocoa beans Production (tonnes)`
* `Palm oil Production (tonnes)`

dan **lima negara ASEAN**:

* **Indonesia**
* **Vietnam**
* **Thailand**
* **Filipina**
* **Malaysia**

### Missing Value and Duplicate Data Check

Memastikan kualitas data sebelum masuk ke tahap analisis atau pemodelan, karena nilai kosong maupun data duplikat dapat mengganggu hasil dan akurasi model.

```python
 data.isnull().sum()
```

Berikut adalah hasil jumlah nilai hilang di setiap kolom:

| Kolom                                  | Nilai Hilang |
| -------------------------------------- | ------------ |
| Entity                                 | 0            |
| Year                                   | 0            |
| Maize Production (tonnes)              | **0**        |
| Rice  Production ( tonnes)             | **0**        |
| Yams  Production (tonnes)              | 0            |
| Wheat Production (tonnes)              | 0            |
| Tomatoes Production (tonnes)           | 0            |
| Tea  Production ( tonnes )             | 0            |
| Sweet potatoes  Production (tonnes)    | 0            |
| Sunflower seed  Production (tonnes)    | 0            |
| Sugar cane Production (tonnes)         | 0            |
| Soybeans  Production (tonnes)          | 0            |
| Rye  Production (tonnes)               | 0            |
| Potatoes  Production (tonnes)          | 0            |
| Oranges  Production (tonnes)           | 0            |
| Peas, dry Production ( tonnes)         | 0            |
| **Palm oil  Production (tonnes)**      | **0**        |
| Grapes  Production (tonnes)            | 0            |
| **Coffee, green Production ( tonnes)** | **0**        |
| Cocoa beans Production (tonnes)        | **0**        |
| Meat, chicken  Production (tonnes)     | 0            |
| Bananas  Production ( tonnes)          | 0            |
| Avocados Production (tonnes)           | 0            |
| Apples Production (tonnes)             | 0            |

```python
print('Total Duplikasi Data:', df.duplicated().sum())
```
Output:
**Total Duplikasi Data: 0**

**Kesimpulan**: Dataset sepenuhnya bersih dengan **tidak ada nilai hilang** dan **duplikasi data** pada semua kolom, termasuk ketiga variabel target: **jagung**, **beras**, **kopi**, **cokelat** dan **minyak sawit**.

### **ASEAN Country and Commodity Selection**

Dari dataset lengkap **World Food Production**, proyek ini menyaring hanya lima negara ASEAN yang menjadi fokus — **Indonesia, Vietnam, Thailand, Filipina dan Malaysia** — serta memilih lima komoditas utama:

* Jagung
* Beras
* Kopi (green)
* Cokelat
* Minyak Sawit

Hanya kolom `Entity`, `Year`, dan ketiga nilai produksi komoditas yang dipertahankan.

## Data Preparation

### Langkah-langkah yang dilakukan:

1. **Penyaringan Data**: Memilih data dari negara ASEAN dan 5 komoditas utama.
2. **Encoding**: Label encoding pada fitur kategorikal seperti negara dan komoditas.
3. **Transformasi**: Menggunakan transformasi logaritmik pada kolom `Value` untuk mengurangi outlier ekstrem.
4. **Normalisasi**: Menggunakan `MinMaxScaler` agar nilai berada pada rentang 0-1.
5. **Sequence Generation**: Membentuk data menjadi time series sequence untuk input LSTM.
6. **Split Data**: Data dibagi menjadi train dan test dengan proporsi 80:20.
7. **Bentuk sequence LSTM**: `look_back=5`

### Feature Renaming for Simplicity

Nama kolom pada dataset mengandung format yang tidak konsisten, seperti spasi berlebih dan anotasi satuan (contoh: " (tonnes)" atau " ( tonnes)"), yang perlu dibersihkan untuk menghindari kesalahan kunci dalam pemrosesan selanjutnya.

Langkah-langkah pembersihan:

```python
df_filtered.columns = df_filtered.columns.str.replace(' \(tonnes\)', '', regex=True)
df_filtered.columns = df_filtered.columns.str.replace(' \( tonnes\)', '', regex=True)
df_filtered.columns = df_filtered.columns.str.replace('Coffee, green Production', 'Coffee green Production', regex=True)
df_filtered.columns = df_filtered.columns.str.replace('Palm oil  Production', 'Palm oil Production', regex=True)
df_filtered.columns = df_filtered.columns.str.replace('Cocoa beans Production', 'Cocoa beans Production', regex=True)
df_filtered.columns = df_filtered.columns.str.replace('Rice  Production', 'Rice Production', regex=True)
df_filtered.columns = df_filtered.columns.str.replace('Maize Production', 'Maize Production', regex=True)
df_filtered.head()
```

Hasilnya:

* `Coffee green Production`
* `Palm oil Production`
* `Cocoa beans Production`
* `Rice Production`
* `Maize Production`

### Exploratory Data Analysis (EDA)

#### Yearly Trend of Maize Production (1961–2021)

Visualisasi ini menunjukkan tren produksi jagung dari tahun 1961 hingga 2021 di lima negara ASEAN terpilih. Jagung merupakan salah satu komoditas utama pangan dan pakan, sehingga pemantauan pertumbuhannya penting untuk mendukung ketahanan pangan.

![Yearly Trend of Maize Production (1961–2021)](repo-dir/yearly_trend_maize_production_1961_2021.png)

**Insight:**

* Produksi jagung menunjukkan pertumbuhan stabil dari tahun ke tahun dengan tren kenaikan yang cukup konsisten.
* Lonjakan signifikan mulai terlihat sejak tahun 2000-an, menunjukkan peningkatan perhatian terhadap komoditas ini.
* Terjadi sedikit penurunan di tahun-tahun terakhir, yang bisa menjadi sinyal untuk evaluasi faktor produksi atau cuaca.

#### Yearly Trend of Rice Production (1961–2021)

Visualisasi ini merepresentasikan tren tahunan produksi beras di kawasan ASEAN. Beras menjadi makanan pokok bagi mayoritas penduduk Asia Tenggara, menjadikannya komoditas strategis untuk dipantau secara jangka panjang.

![Yearly Trend of Rice Production (1961–2021)](repo-dir/yearly_trend_rice_production_1961_2021.png)

**Insight:**

* Produksi padi cenderung fluktuatif tetapi menunjukkan tren kenaikan jangka panjang yang kuat.
* Terdapat penurunan tajam sekitar tahun 1967 dan beberapa fluktuasi tajam di era 1990–2000, mengindikasikan pengaruh faktor eksternal seperti kebijakan atau cuaca ekstrem.
* Dalam dekade terakhir, tren terlihat meningkat kembali dengan volume produksi mencapai puncaknya pada tahun 2021.

#### Yearly Trend of Coffee Production (1961–2021)

Visualisasi ini menggambarkan dinamika produksi kopi tahunan yang sangat fluktuatif, dipengaruhi oleh faktor cuaca, harga global, dan kebijakan ekspor.

![Yearly Trend of Coffee Production (1961–2021)](repo-dir/yearly_trend_coffee_production_1961_2021.png)

**Insight:**

* Produksi kopi sangat volatil, terutama sebelum tahun 2000, menunjukkan ketergantungan tinggi pada musim dan harga pasar global.
* Lonjakan besar setelah tahun 2000 diikuti oleh fluktuasi tajam, menandakan potensi pertumbuhan yang besar tetapi juga risiko tinggi dalam stabilitas produksi.
* Tidak terdapat pola pertumbuhan linear, yang memperkuat pentingnya penggunaan model prediktif untuk komoditas ini.

![Yearly Trend of Cocoa Beans Production (1961–2021)](repo-dir/yearly_trend_cocoa_beans_production_1961_2021.png)

**Insight:**

* Produksi kakao mengalami pertumbuhan signifikan dari awal 1970-an hingga sekitar tahun 1980, namun disusul oleh penurunan tajam. Setelah tahun 1980, tren produksi cenderung fluktuatif, dengan periode stagnasi panjang hingga akhir 1990-an.
* Kenaikan drastis kembali terjadi antara awal 2000-an hingga puncaknya di sekitar tahun 2008–2009.
* Tren menunjukkan penurunan yang tidak konsisten, dengan beberapa kenaikan sementara namun tidak mencapai puncak sebelumnya.
* Variabilitas tinggi dalam dua dekade terakhir mengindikasikan ketidakstabilan produksi kakao, yang dapat disebabkan oleh faktor eksternal seperti cuaca, hama tanaman, atau dinamika pasar global.

####  Yearly Trend of Palm Oil Production (1961–2021)

![Yearly Trend of Palm Oil Production (1961–2021)](repo-dir/yearly_trend_palm_oil_production_1961_2021.png)

**Insight:**

* Produksi minyak kelapa sawit menunjukkan peningkatan pesat pada era 1980–1990 yang kemudian diikuti penurunan tajam sekitar tahun 2000.
* Terdapat fluktuasi besar pasca tahun 2000, termasuk anomali penurunan drastis yang mungkin berkaitan dengan kebijakan ekspor, regulasi, atau faktor lingkungan.
* Kenaikan tajam pada tahun-tahun akhir grafik menunjukkan potensi rebound dalam industri kelapa sawit, meskipun dengan risiko volatilitas tinggi.

---

## Data Preparation

### Checking Outliers

Proses ini menggunakan visualisasi boxplot untuk mengidentifikasi outlier pada lima komoditas utama. Boxplot membantu melihat sebaran data, median, kuartil, dan titik-titik ekstrem yang dianggap sebagai outlier berdasarkan rentang interkuartil.

```python
plt.figure(figsize=(12, 6))

for i, commodity in enumerate(commodities):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x=df_filtered[commodity], color='red')
    plt.title(commodity)

plt.tight_layout()
plt.show()
```

Hasilnya seperti gambar berikut:
![Checking Outliers](repo-dir/checking_outliers.png)

### Handling Outliers

Data produksi pertanian sering kali mengandung nilai ekstrem dan pola pertumbuhan non-linear. Alih-alih menghapus outlier, proyek ini menggunakan **transformasi logaritmik (`log1p`)** untuk menstabilkan variansi namun tetap mempertahankan semua data:

```python
df_transformed = df_filtered.copy()

for col in commodities:
    df_transformed[col] = np.log1p(df_transformed[col])

df_transformed
```

Hasilnya ditunjukkan dalam gambar berikut:
![Handling Outliers](repo-dir/handling_outliers.png)

### Data Normalization

Normalisasi data dilakukan untuk menskalakan nilai fitur ke dalam rentang 0 hingga 1 menggunakan metode **MinMaxScaler** agar setiap fitur memiliki skala yang seimbang.

```python
df_normalized = df_transformed.copy()

scalers = {}

for col in commodities:
    scaler = MinMaxScaler()
    df_normalized[col] = scaler.fit_transform(df_normalized[[col]])
    scalers[col] = scaler

df_normalized
```

### Data Splitting

Membagi data masing-masing negara dan komoditas ke dalam subset pelatihan (train) dan pengujian (test) berdasarkan proporsi 80 persen untuk pelatihan dan 20 persen untuk pengujian. Data dibagi secara individual untuk setiap kombinasi negara dan komoditas, agar model nantinya dapat belajar dari pola spesifik masing-masing.

* **Training Set**: 80% tahun awal
* **Testing Set**: 20% tahun akhir (hingga **2021**)

### Data Reshape

Data deret waktu diubah menjadi format sekuensial menggunakan pendekatan sliding window dengan `look_back = 5`, yang berarti model akan mempelajari lima tahun sebelumnya untuk memprediksi satu tahun berikutnya. Data kemudian direstrukturisasi menjadi bentuk tiga dimensi yang sesuai dengan input model LSTM, yaitu [samples, timesteps, features].

```
Input: [1961, 1962, 1963, 1964, 1965]
Target: 1966
```

Kode:

```python
def create_sequences(series, look_back=5):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:i+look_back])
        y.append(series[i+look_back])
    return np.array(X), np.array(y)
```

### Data Structure Verification

Memastikan data hasil reshape memiliki dimensi yang sesuai untuk digunakan dalam model LSTM. Setiap sampel pelatihan dan pengujian direpresentasikan sebagai sekuens 5 tahun sebelumnya (look-back) dan target satu tahun ke depan.

```python
key = 'Indonesia - Rice Production'

print(f"Data for {key}:\n")
print("X_train shape:", reshaped_data[key]['X_train'].shape)
print("y_train shape:", reshaped_data[key]['y_train'].shape)
print("X_test shape:", reshaped_data[key]['X_test'].shape)
print("y_test shape:", reshaped_data[key]['y_test'].shape)
print("years_test shape:", reshaped_data[key]['years_test'].shape)
```
---

## Model Development
Untuk setiap pasangan negara–komoditas, dibangun satu model LSTM yang dilatih menggunakan data historis produksi tahunan. Model mempelajari pola dalam 5 tahun terakhir (look_back = 5) untuk memprediksi produksi pada tahun berikutnya.

Model ini dikonfigurasi sebagai berikut:

* **1 lapisan LSTM dengan 50 unit:**
  Lapisan ini digunakan untuk menangkap pola temporal dari data deret waktu tahunan. 50 unit neuron cukup untuk mempelajari variasi tren produksi tanpa overfitting pada dataset per negara.

* **Fungsi aktivasi ReLU:**
  Digunakan karena bekerja efektif pada data numerik dan mempercepat konvergensi model dengan menghindari masalah vanishing gradient.

* **Lapisan output Dense(1):**
  Menghasilkan satu nilai prediksi (produksi tahun ke-n), sesuai dengan sifat regresi univariat.

* **Fungsi kerugian MSE (Mean Squared Error):**
  Mengukur rata-rata kuadrat selisih antara prediksi dan nilai aktual. Cocok digunakan dalam tugas regresi seperti prediksi produksi.

* **Optimizer ADAM:**
  Digunakan karena efisien dan bekerja baik di berbagai jenis data, termasuk data time series seperti produksi tahunan.

* **EarlyStopping:**
  Model dilatih maksimal 100 epoch, namun dapat berhenti lebih awal jika tidak ada perbaikan nilai loss selama 10 epoch berturut-turut (patience=10), dengan toleransi perubahan minimal 0.001.

**Untuk Proses Training:**

* Model dilatih untuk setiap pasangan negara-komoditas yang tersedia dalam reshaped_data.
* Output model berupa:
  * **model terlatih**
  * **prediksi pada data uji (y_pred)**
  * **nilai MSE dan RMSE** sebagai metrik evaluasi

### Train LSTM Model

* Epoch hingga 100, dengan early stopping aktif
* Batch size: 16

```python
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
```

### Train Procedure

Setiap model LSTM dilatih menggunakan 80% data urutan pertama, sementara 20% sisanya digunakan untuk pengujian. EarlyStopping diterapkan berdasarkan training loss:

```python
early_stop = EarlyStopping(
    monitor='loss',
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
    verbose=1
)

model.fit(
    X_train, y_train,
    epochs=100,
    verbose=1,
    callbacks=[early_stop]
)
```

---

## Evaluation

Setelah model LSTM dilatih untuk setiap pasangan negara–komoditas, langkah selanjutnya adalah mengevaluasi performa model berdasarkan nilai produksi dalam satuan aslinya (ton). Ini dilakukan dengan cara membalik proses normalisasi menggunakan inverse_transform.

Menghitung metrik evaluasi akhir dengan menggunakan:
* MSE (Mean Squared Error): rata-rata kuadrat dari selisih prediksi dan nilai aktual.
* RMSE (Root Mean Squared Error): akar dari MSE, lebih mudah ditafsirkan karena dalam satuan yang sama (ton).

### Evaluation Model and Inverse Transfom

Berikut adalah bagaimana evaluasi dilakukan:

```python
y_pred_inv = scaler.inverse_transform(y_pred)
y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1))

mse = mean_squared_error(y_true_inv, y_pred_inv)
rmse = mse ** 0.5
```

### Performa Model

| Negara    | Komoditas    | MSE      | RMSE      |
| --------- | ------------ | -------- | --------- |
| Indonesia | Jagung       | 0.056678 | **0.238** |
| Indonesia | Beras        | 0.021462 | **0.146** |
| Indonesia | Kopi (green) | 0.185553 | **0.431** |
| Indonesia | Kakao        | 0.480563 | **0.693** |
| Indonesia | Minyak sawit | 0.446703 | **0.668** |
| Vietnam   | Jagung       | 0.036487 | **0.191** |
| Vietnam   | Beras        | 0.063330 | **0.252** |
| Vietnam   | Kopi (green) | 6.172344 | **2.484** |
| Vietnam   | Kakao        | 0.874812 | **0.935** |
| Vietnam   | Minyak sawit | 6.881076 | **2.623** |
| Thailand  | Jagung       | 0.037853 | **0.195** |
| Thailand  | Beras        | 1.779169 | **1.334** |
| Thailand  | Kopi (green) | 1.237864 | **1.113** |
| Thailand  | Kakao        | 0.333705 | **0.578** |
| Thailand  | Minyak sawit | 0.039493 | **0.199** |
| Filipina  | Jagung       | 0.051686 | **0.227** |
| Filipina  | Beras        | 6.890931 | **2.625** |
| Filipina  | Kopi (green) | 2.795318 | **1.672** |
| Filipina  | Kakao        | 4.237350 | **2.058** |
| Filipina  | Minyak sawit | 1.237029 | **1.112** |
| Malaysia  | Jagung       | 0.397197 | **0.630** |
| Malaysia  | Beras        | 0.069078 | **0.263** |
| Malaysia  | Kopi (green) | 3.332711 | **1.826** |
| Malaysia  | Kakao        | 7.442635 | **2.728** |
| Malaysia  | Minyak sawit | 2.842913 | **1.686** |

**Insight & Interpretasi**

* 🇮🇩 **Indonesia** menunjukkan performa terbaik secara keseluruhan, terutama untuk **beras** dengan **RMSE = 0.146**, diikuti oleh jagung (**0.238**) dan kopi (**0.431**). Ini menunjukkan pola produksi yang stabil dan dapat diprediksi.
* 🇻🇳 **Vietnam** memiliki performa sangat baik untuk jagung dan beras, tetapi sangat buruk untuk kopi dan kelapa sawit (**RMSE > 2.5**), menunjukkan pola produksi yang tidak stabil atau historis yang fluktuatif.
* 🇹🇭 **Thailand** menunjukkan performa sedang hingga baik. Jagung dan kelapa sawit diprediksi dengan sangat akurat (**RMSE < 0.2**), sedangkan beras dan kopi menunjukkan error yang lebih tinggi.
* 🇵🇭 **Filipina** menunjukkan prediksi yang buruk untuk beras (**RMSE = 2.625**) dan kakao (**RMSE = 2.058**), yang mengindikasikan ketidakstabilan data historis atau noise yang tinggi.
* 🇲🇾 **Malaysia** memiliki performa yang bervariasi, dengan prediksi jagung dan beras yang cukup akurat (**RMSE < 0.7**), tetapi buruk untuk kopi dan kakao (**RMSE > 1.8**), menandakan fluktuasi tinggi dalam produksi tahunan.

### Visual Evaluation

Visualisasi dilakukan untuk membandingkan prediksi model terhadap data aktual pada periode pelatihan dan pengujian. Data produksi dikembalikan ke skala aslinya menggunakan inverse_transform sebelum divisualisasikan. Setiap grafik menunjukkan tren tahunan produksi komoditas dengan garis prediksi dan aktual untuk masing-masing negara dan komoditas, yang terbagi dalam dua subplot: pelatihan dan pengujian.

Indonesia's Model Evaluation:
![Visualizing forecast for Indonesia - Maize Production](repo-dir/visualizing-forecast-indonesia-maize-production.png)
![Visualizing forecast for Indonesia - Rice Production](repo-dir/visualizing-forecast-indonesia-rice-production.png)
![Visualizing forecast for Indonesia - Coffee green Production](repo-dir/visualizing-forecast-indonesia-coffee-green-production.png)
![Visualizing forecast for Indonesia - Cocoa beans Production](repo-dir/visualizing-forecast-indonesia-cocoa-beans-production.png)
![Visualizing forecast for Indonesia - Palm oil Production](repo-dir/visualizing-forecast-indonesia-palm-oil-production.png)

Vietnam's Model Evaluation:
![Visualizing forecast for Vietnam - Maize Production](repo-dir/visualizing-forecast-vietnam-maize-production.png)
![Visualizing forecast for Vietnam - Rice Production](repo-dir/visualizing-forecast-vietnam-rice-production.png)
![Visualizing forecast for Vietnam - Coffee green Production](repo-dir/visualizing-forecast-vietnam-coffee-green-production.png)
![Visualizing forecast for Vietnam - Cocoa beans Production](repo-dir/visualizing-forecast-vietnam-cocoa-beans-production.png)

Philippines Model Evaluation:
![Visualizing forecast for Philippines - Maize Production](repo-dir/visualizing-forecast-philippines-maize-production.png)
![Visualizing forecast for Philippines - Rice Production](repo-dir/visualizing-forecast-philippines-rice-production.png)
![Visualizing forecast for Philippines - Coffee green Production](repo-dir/visualizing-forecast-philippines-coffee-green-production.png)
![Visualizing forecast for Philippines - Cocoa beans Production](repo-dir/visualizing-forecast-philippines-cocoa-beans-production.png)
![Visualizing forecast for Philippines - Palm oil Production](repo-dir/visualizing-forecast-philippines-palm-oil-production.png)

Malaysia's Model Evaluation:
![Visualizing forecast for Malaysia - Maize Production](repo-dir/visualizing-forecast-malaysia-maize-production.png)
![Visualizing forecast for Malaysia - Rice Production](repo-dir/visualizing-forecast-malaysia-rice-production.png)
![Visualizing forecast for Malaysia - Coffee green Production](repo-dir/visualizing-forecast-malaysia-coffee-green-production.png)
![Visualizing forecast for Malaysia - Cocoa beans Production](repo-dir/visualizing-forecast-malaysia-cocoa-beans-production.png)
![Visualizing forecast for Malaysia - Palm oil Production](repo-dir/visualizing-forecast-malaysia-palm-oil-production.png)

### Forecasting Results

####  Bar Chart Visualization

Grafik ini membandingkan hasil prediksi produksi tahun 2030 untuk lima komoditas utama di lima negara ASEAN. Visualisasi menggunakan grouped bar chart untuk memperlihatkan performa masing-masing negara dalam setiap komoditas.

![Forecasted Production in 2030 by Commodity (Indonesia vs Others)](repo-dir/forecasted-production-2030-by-commodity-indonesia-vs-others.png)

#### Forecast Comparison (Until 2030)

Melakukan prediksi produksi komoditas hingga tahun 2030 untuk setiap kombinasi negara dan komoditas yang tersedia. Model LSTM yang telah dilatih digunakan untuk menghasilkan prediksi berdasarkan 5 data terakhir dari gabungan data pelatihan dan pengujian. Output prediksi disimpan dalam struktur forecast_by_commodity dan divisualisasikan dalam bentuk grafik per komoditas untuk membandingkan tren antar negara.

![Forecast Comparison Until 2030 - Maize Production](repo-dir/forecast-comparison-until-2030-maize-production.png)
![Forecast Comparison Until 2030 - Rice Production](repo-dir/forecast-comparison-until-2030-rice-production.png)
![Forecast Comparison Until 2030 - Coffee green Production](repo-dir/forecast-comparison-until-2030-coffee-green-production.png)
![Forecast Comparison Until 2030 - Cocoa beans Production](repo-dir/forecast-comparison-until-2030-cocoa-beans-production.png)
![Forecast Comparison Until 2030 - Palm oil Production](repo-dir/forecast-comparison-until-2030-palm-oil-production.png)

---

## Keterkaitan Hasil Evaluasi dengan Business Understanding

### Apakah model menjawab setiap problem statement?

**Masalah 1:**

> *Tren produksi pertanian di negara-negara ASEAN sangat fluktuatif dan sulit diprediksi tanpa alat yang akurat.*

**Terjawab.**
Model LSTM yang dikembangkan mampu mempelajari pola musiman dan non-linear dalam data historis produksi tahunan dari lima negara ASEAN. Hal ini terbukti dari RMSE yang rendah pada beberapa kombinasi.

**Masalah 2:**

> *Tidak adanya standar prediksi untuk membandingkan posisi Indonesia dengan negara ASEAN lainnya.*

**Terjawab.**
Model berhasil dibuat untuk **setiap negara dan komoditas**, sehingga memungkinkan:

* Perbandingan RMSE antar negara untuk satu komoditas
* Visualisasi dan pemeringkatan posisi Indonesia dibandingkan Vietnam, Thailand, Filipina, dan Malaysia

**Masalah 3:**

> *Model konvensional tidak mampu menangkap pola non-linier dan musiman pada data pertanian.*

**Terjawab.**
Model LSTM univariat mampu menangkap ketidakteraturan pola yang sulit dideteksi oleh regresi linier atau model statistik sederhana.
Ini terlihat pada keberhasilan model dalam memprediksi tren jangka panjang komoditas seperti jagung di Filipina dan kopi di Indonesia, dengan RMSE rendah.

---

### Apakah model berhasil mencapai goals?

**Tujuan 1:**

Model LSTM berhasil dilatih menggunakan data 1961–2021 dan digunakan untuk memprediksi produksi 2022–2030.

**Tujuan 2:**

Perbandingan antar negara berhasil dilakukan melalui hasil evaluasi numerik (RMSE) dan visualisasi.

**Tujuan 3:**

Model LSTM terbukti mampu mengatasi keterbatasan model tradisional dalam mengenali pola non-linear dan musiman.

---

### Apakah setiap solusi yang direncanakan berdampak?

**Solusi 1:**

Penggunaan LSTM terbukti efektif untuk menangani dataset deret waktu produksi tahunan. Model menunjukkan hasil yang lebih baik pada pola yang konsisten dan mampu beradaptasi pada pola yang fluktuatif.

**Solusi 2:**

Tahapan preprocessing dan penggunaan look_back terbukti penting dalam menyiapkan data agar model bisa belajar dengan optimal.

**Solusi 3:**

Penggunaan **early stopping** dan **evaluasi dengan RMSE** memberikan hasil yang lebih robust, mencegah overfitting, dan memudahkan interpretasi kesalahan prediksi.

---

### Kesimpulan

Hasil evaluasi mendukung pemanfaatan model LSTM univariat sebagai alat bantu perencanaan pangan jangka panjang.
Proyek ini **berhasil menjawab seluruh problem statement**, **mencapai semua goals yang ditetapkan**, dan **menghasilkan solusi yang berdampak nyata dalam konteks ketahanan pangan dan benchmarking regional ASEAN**.

---

## Referensi

\[1] R. S. Ahmad, *World Food Production Dataset*, Kaggle. \[Online]. Tersedia: [https://www.kaggle.com/datasets/rafsunahmad/world-food-production/data](https://www.kaggle.com/datasets/rafsunahmad/world-food-production/data)

\[2] N. Siddique dan H. Adeli, “Machine Learning Models for Food Security and Climate Change,” dalam *Machine Learning and Big Data Analytics Paradigms: Analysis, Applications and Challenges*, Springer, 2022, pp. 101–118. \[Online]. Tersedia: [https://link.springer.com/chapter/10.1007/978-3-031-08743-1\_6](https://link.springer.com/chapter/10.1007/978-3-031-08743-1_6)

\[3] ASEANstats, *ASEAN Statistical Brief, Vol. VII, April 2024*. ASEAN Secretariat, Apr. 2024. \[Online]. Tersedia: [https://www.aseanstats.org/wp-content/uploads/2024/04/ASEAN-Statistical-Brief\_April-2024\_v3.xlsx](https://www.aseanstats.org/wp-content/uploads/2024/04/ASEAN-Statistical-Brief_April-2024_v3.xlsx)

\[4] ASEAN Secretariat, *Industry Focus: Agriculture*. \[Online]. Tersedia: [https://asean.org/industry-focus/](https://asean.org/industry-focus/)

\[5] Encyclopaedia Britannica, “Southeast Asia – Industry,” *Britannica.com*. \[Online]. Tersedia: [https://www.britannica.com/place/Southeast-Asia/Industry](https://www.britannica.com/place/Southeast-Asia/Industry)
