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

![Yearly Trend of Maize Production (1961–2021)](repo-dir/Yearly Trend of Maize Production (1961–2021).png)

#### Yearly Trend of Rice Production (1961–2021)

Visualisasi ini merepresentasikan tren tahunan produksi beras di kawasan ASEAN. Beras menjadi makanan pokok bagi mayoritas penduduk Asia Tenggara, menjadikannya komoditas strategis untuk dipantau secara jangka panjang.

![Yearly Trend of Rice Production (1961–2021)](repo-dir/Yearly Trend of Rice Production (1961–2021).png)


## Modeling

### Model yang digunakan: **LSTM (Long Short-Term Memory)**

* **Arsitektur:**

  * 1 lapisan LSTM (50 unit)
  * 1 lapisan Dense
* **Optimizer:** Adam
* **Loss function:** Mean Squared Error (MSE)
* **EarlyStopping:** digunakan dengan `patience=10` untuk mencegah overfitting

### Proses training:

* Epoch hingga 100, dengan early stopping aktif
* Batch size: 16

## Evaluation

### Metrik yang digunakan:

* **Mean Squared Error (MSE)**: Mengukur rata-rata kuadrat selisih antara prediksi dan aktual.
* **Root Mean Squared Error (RMSE)**: Mengukur kesalahan dalam satuan asli (ton).

### Hasil evaluasi:

* **Akurasi terbaik** ditemukan pada:

  * *Thailand – Maize* (RMSE: 0.0656)
  * *Vietnam – Maize* (RMSE: 0.1006)
  * *Indonesia – Rice* (RMSE: 0.1906)

* **RMSE tinggi** ditemukan pada:

  * *Vietnam – Palm oil* (RMSE: 2.6808)
  * *Philippines – Rice* (RMSE: 2.7448)
  * *Malaysia – Cocoa beans* (RMSE: 2.8684)

* Ini menunjukkan bahwa beberapa negara dan komoditas memiliki **pola yang lebih mudah diprediksi**, sedangkan yang lain sangat fluktuatif.

### Visualisasi:

* Grafik prediksi vs aktual untuk tiap negara dan komoditas
* Prediksi produksi hingga tahun 2030 menunjukkan tren naik pada komoditas seperti beras dan kopi di beberapa negara


## Hasil Prediksi

Proyek ini memprediksi produksi jagung, beras, kopi, kakao dan kelapa sawit dari tahun **2022 hingga 2030** untuk empat negara utama ASEAN. Berikut adalah hasil utama dari prediksi tersebut:

### Maize Production

![Forecasted Maize Production](repo_dir/Forecast-Maize-Until-2030.png)

### Rice Production

![Forecasted Rice Production](repo_dir/Forecast-Rice-Until-2030.png)

### Coffee Green Production

![Forecasted Coffee Green Production](repo_dir/Forecast-CoffeeGreen-Until-2030.png)

### Cocoa beans Production
![Forecasted Cocoa Beans Production](repo_dir/Forecast-Cocoa-Until-2030)

### Palm Oil Production
![Forecasted Palm Oil Production](repo_dir/Forecast-PalmOil-Until-2030)

Prediksi ini memberikan wawasan penting bagi pengambil kebijakan dan pelaku industri untuk merencanakan strategi pertanian dan mengantisipasi tren produksi lintas negara ASEAN.

---

## Referensi

1. [World Food Production Dataset (Kaggle)](https://www.kaggle.com/datasets/rafsunahmad/world-food-production/data)
2. [ML Models: Food Security and Climate Change](https://link.springer.com/chapter/10.1007/978-3-031-08743-1_6)
3. [Predicting Agricultural Commodities with Machine Learning](https://arxiv.org/abs/2310.18646)
4. Food and Agriculture Organization (FAO). (2023). *World Food and Agriculture Statistical Yearbook*. [https://www.fao.org](https://www.fao.org)
5. World Bank. (2022). *Agricultural Indicators*. [https://data.worldbank.org](https://data.worldbank.org)

----


