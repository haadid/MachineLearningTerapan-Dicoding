# Laporan Proyek Machine Learning - Marwan Hadid

## Project Overview
Ponsel cerdas atau *smartphone* adalah perangkat yang mendominasi kehidupan sehari-hari dan terus berkembang dengan teknologi baru. Proyek ini menggunakan teknik *Content Based Filtering* untuk merekomendasikan ponsel berdasarkan fiturnya dan *Collaborative Filtering* untuk merekomendasikan ponsel kepada pengguna berdasarkan preferensi sebelumnya. Judul proyek ini adalah "Sistem Rekomendasi Ponsel Pintar"

Dalam era teknologi yang terus berkembang, kehadiran ponsel pintar telah menjadi bagian integral dari kehidupan sehari-hari. Pengguna dihadapkan pada banyak pilihan ponsel dengan berbagai fitur dan spesifikasi. Pada proyek ini, kita bertujuan untuk menyederhanakan proses pemilihan ponsel dengan memanfaatkan teknologi _machine learning_ untuk memberikan rekomendasi yang lebih personal dan relevan kepada pengguna.

Dengan menerapkan teknik _Content Based Filtering_ dan _Collaborative Filtering_, proyek ini tidak hanya memberikan rekomendasi berdasarkan fitur-fitur ponsel, tetapi juga mempertimbangkan preferensi pengguna sebelumnya. Ini bertujuan untuk meningkatkan pengalaman pengguna dalam menemukan ponsel yang sesuai dengan kebutuhan dan preferensi mereka. Dalam konteks masyarakat yang semakin terhubung, diharapkan proyek ini dapat memberikan kontribusi positif dengan memudahkan konsumen dalam mengambil keputusan yang cerdas saat memilih ponsel pintar.

### Urgensi dan Relevansi
1. **Kemudahan Konsumen**
    <br> Dalam lingkungan yang dipenuhi dengan berbagai pilihan ponsel, pemilihan yang tepat menjadi krusial. Pengguna sering kali menghadapi dilema dalam memilih ponsel dengan fitur yang sesuai dengan kebutuhan mereka. Sistem rekomendasi ini memiliki urgensi untuk menyederhanakan proses ini, membantu pengguna menemukan ponsel yang paling relevan dengan preferensi dan kebutuhan individu. <br>
2.  **Penyesuaian Rekomendasi Berdasarkan Preferensi Pengguna**
    <br> Pengguna modern memiliki preferensi yang unik terkait dengan ponsel, seperti merek tertentu, ukuran layar, atau kamera yang kuat. _Content Based Filtering_ memungkinkan penyesuaian rekomendasi berdasarkan preferensi ini, yang dapat meningkatkan kepuasan pengguna dan mengarah pada pengalaman pengguna yang lebih positif. <br>
3.  **Efisiensi Waktu dan Pengambilan Keputusan yang Cerdas**
    <br> Dengan banyaknya opsi di pasaran, mencari ponsel yang sesuai dapat menjadi tugas yang memakan waktu. Sistem ini membantu efisiensi waktu dengan menyajikan opsi terbaik secara cepat, memungkinkan pengguna membuat keputusan yang lebih cerdas dan sesuai dengan kebutuhan mereka. <br>

## Business Understanding

### Problem Statements

Berdasarkan latar belakang diatas, permasalahan yang dapat diselesaikan pada proyek ini adalah sebagai berikut:
-   Bagaimana sistem rekomendasi yang baik untuk diterapkan pada kasus ini?
-   Bagaimana cara membuat sistem rekomendasi ponsel pintar yang dapat merekomendasikan ponsel berdasarkan fitur dan preferensi pengguna?


### Goals
Tujian dari pernyataan masalah adalah sebagai berikut:
-   Membuat sistem rekomendasi ponsel pintar yang dapat merekomendasikan ponsel berdasarkan fitur dan preferensi pengguna.
- 	Memberikan rekomendasi ponsel yang mungkin disukai dan belum pernah digunakan oleh pengguna.


### Solution statements
Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini diantaranya :
- **Pra-pemrosesan Data**. Pada tahap pra-pemrosesan data, beberapa langkah dapat diambil, antara lain:
    -   Menghapus kolom/fitur yang tidak diperlukan.
    -   Menangani nilai-nilai yang hilang atau data kosong.
    -   Melakukan encoding pada fitur kategorikal.

- **Persiapan Data**. Persiapan data melibatkan langkah-langkah seperti:
    -   Persiapan data untuk model Content-Based Filtering.
        -   Melakukan seleksi fitur-fitur penting dari data ponsel.
        -   Menyusun matriks representasi fitur ponsel.
        -   Melakukan perhitungan matriks kemiripan antar ponsel berdasarkan fitur-fitur yang dipilih.

    -   Persiapan data untuk model Collaborative Filtering dengan KNN.
        -   Mengubah format dataset menjadi *pivot table*.

    -   Persiapan data untuk model Collaborative Filtering dengan *Deep Learning*.
        -   Melakukan *encoding* fitur kedalam indeks integer.
        -   Pembagian dataset untuk pelatihan dan validasi.

-   **Pembangunan Model**.
    <br> Sistem rekomendasi ini menggunakan dua pendekatan utama, yaitu model *Content Based Filtering* dengan teknik *Cosine Similarity* dan *Collaborative Filtering* dengan *Deep Learning* dan *K-Nearest Neighbors* (KNN). Model *Content Based Filtering* ini akan memberikan rekomendasi berdasarkan kesamaan fitur dengan ponsel yang telah disukai oleh pengguna. Sementara itu, model *Collaborative Filtering* menggunakan pendekatan *Deep Learning* untuk memahami pola kompleks dari data historis pengguna dan *K-Nearest Neighbors* (KNN) untuk menangani situasi cold start dan memberikan rekomendasi berdasarkan kesamaan pengguna.

    -   **Content-Based Filtering dengan *Cosine Similarity***.
        <br> Pada pendekatan ini, ponsel direkomendasikan berdasarkan kesamaan fitur dengan ponsel yang telah disukai oleh pengguna. Nilai kesamaan diukur menggunakan metrik Cosine Similarity, yang mengukur sudut kosinus antara dua vektor fitur. Semakin kecil sudutnya, semakin besar kesamaan antara dua ponsel. Meskipun referensi tidak menyebutkan secara eksplisit metode ini sebagai Content-Based Filtering, namun penggunaan fitur untuk merekomendasikan item yang mirip dengan item yang disukai pengguna sesuai dengan konsep dasar Content-Based Filtering. Metode ini bekerja dengan mengukur kesamaan antara item berdasarkan preferensi pengguna. Prosesnya melibatkan perhitungan kesamaan antara item, dan hasilnya digunakan untuk memberikan rekomendasi item yang belum dilihat oleh pengguna. Jarak kedekatan (kemiripan) antar item diukur menggunakan metrik. Item yang memiliki nilai kesamaan tertinggi dengan item yang disukai pengguna dijadikan rekomendasi. Metode ini efektif untuk memberikan rekomendasi item yang serupa dengan item yang disukai oleh pengguna.

    -   **Collaborative Filtering dengan *K-Nearest Neighbor***.
        <br> Pada pendekatan *Collaborative Filtering*, terdapat dua jenis, yaitu *User-Based* dan *Item-Based*, pada proyek ini akan digunakan *Item-Based*. Metode *Item-Based* *Collaborative Filtering* ini memanfaatkan kesamaan antara pemberi rating terhadap item untuk memberikan rekomendasi. Pada tahap awal, dilakukan perhitungan kesamaan antara item berdasarkan preferensi pengguna. Setelah itu, rekomendasi diberikan berdasarkan item yang mirip dengan item yang disukai pengguna.

    -   **Collaborative Filtering dengan *Deep Learning***.
        <br> Model *deep learning* digunakan untuk memodelkan hubungan yang kompleks antara pengguna dan ponsel berdasarkan data historis. *Deep Learning* adalah subbidang dari *machine learning* yang terinspirasi oleh struktur otak manusia, disebut *Artificial Neural Networks* (ANN). ANN adalah jaringan saraf tiruan yang memiliki struktur mirip dengan otak manusia. Ia terdiri dari lapisan-lapisan (layers) dengan neuron-neuron yang saling terhubung. Pada model *Collaborative Filtering*, ANN dapat memahami pola yang sulit diidentifikasi melalui metode tradisional. Model ini dapat memberikan rekomendasi yang lebih personal dan akurat dengan memahami konteks dan preferensi pengguna secara mendalam. Metode Deep Learning lebih efisien dalam menangani data historis dan memberikan rekomendasi yang lebih tepat sasaran berdasarkan pemahaman yang lebih mendalam terhadap pola-pola kompleks dalam data.


## Data Understanding
- **Informasi Dataset**
  <br> Dataset yang digunakan pada proyek ini bernama `Cellphones Recommendations`. Berikut informasi lebih lanjut mengenai dataset:

  | Jenis           | Keterangan                                                                                                  |
  |-----------------|-------------------------------------------------------------------------------------------------------------|
  | Sumber          | Cellphones Recommendations - [Kaggle](https://www.kaggle.com/datasets/meirnizri/cellphones-recommendations) |
  | Pemilik         | [Meir Nizri](https://www.kaggle.com/meirnizri)                                                              |
  | Lisensi         | [Open Data Commons Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/)       |
  | Kategori        | Pre-Trained Model, Electronics, E-Commerce Services, Mobile and Wireless, Recommender Systems               |
  | Tipe dan Ukuran | CSV (5 kB)                                                                                                  |
  
  **Tabel 1.** Informasi Dataset

  Berikut informasi lebih lanjut mengenai berkas dataset yang telah diunduh:
    - Terdapat tiga berkas csv dari dataset yaitu `cellphones data.csv`, `cellphones rating`, dan `cellphones users`.
    - Terdapat 959 baris data dan 19 kolom pada dataset.
    - STerdapat 14 fitur bertipe numerik dan 5 bertipe obyek.
    - Fitur-fitur pada dataset adalah sebagai berikut:
        - `user_id`: Penomoran unik untuk setiap pengguna.
        - `cellphone_id`: Penomoran indeks unik pada setiap ponsel.
        - `age`: Umur pengguna dalam tahun.
        - `gender`: Jenis kelamin pengguna.
        - `occupation`: Jenis pekerjaan pengguna.
        - `brand`: Nama merek produsen ponsel
        - `model`: Tipe spesifik tiap ponsel.
        - `operating system`: Sistem operasi pada ponsel.
        - `internal memory`: Ukuran memori internal yang tersedia dalam skala *giga byte* (GB).
        - `RAM`: Ukuran RAM pada ponsel dalam skala giga byte (GB).
        - `performance`: Rating performa ponsel berdasarkan skor *AnTuTu*.
        - `main camera`: Resolusi kamera utama (belakang) dalam skala megapiksel (MP).
        - `selfie camera`: Resolusi kamera *selfie* (depan) dalam skala megapiksel (MP).
        - `battery size`: Kapasitas baterai pada ponsel dalam miliamper perjam (mAh).
        - `screen size`: Ukuran ponsel dalam ukuran inci (*inches*).
        - `weight`: Berat ponsel dalam gram (g).
        - `price`: Harga ponsel dalam mata uang dollar (USD).
        - `release date`: Tanggal rilis ponsel.

- **Data Visualization**
    - **Jumlah Ponsel Tiap Brand**
      <br> Terdapat 10 brand yang terdapat pada dataset.
      ![jumlah_ponsel_tiap_brand](https://user-images.githubusercontent.com/92203636/285137599-a3c05129-2a4d-4d19-b776-e87af062fa78.png)
      <br> **Gambar 1.** Jumlah Ponsel Tiap Brand <br>
      <br> Pada Gambar 1, ponsel pintar merek Samsung memiliki banyak model yang terdapat dalam dataset. Sedangkan  ponsel pintar model Asus, Oppo, Sony, dan Vivo hanya terdapat masing-masing 1.

    - **Rata-rata Rating Ponsel Tiap Brand**
      <br> ![rata_rata_rating_tiap_brand](https://user-images.githubusercontent.com/92203636/285138711-58ec6647-c4b5-4a1e-a531-572b2a1283dc.png)
      <br> **Gambar 2.** Rata-rata Rating Ponsel Tiap Brand <br>
      <br> Pada Gambar 2, ponsel pintar merek Apple memiliki nilai rata-rata yang paling baik. *Chart* ini tidak bisa wakilkan dengan baik karena jumlah ponsel tiap brand berbeda.

    - **Distribusi Rating**
      <br> ![distribusi_rating](https://user-images.githubusercontent.com/92203636/285139930-44a855ca-3d34-4334-b3de-581c17ce028d.png)
      <br> **Gambar 3.** Distribusi Rating <br>
      <br> Pada Gambar 3, distribusi rating terlihat tidak beraturan dan ada sedikit pola yang menunjukkan *negative skew* pada rating 3-8.

    - **Wordcloud Model Ponsel**
       <br> ![wordcloud_model_ponsel](https://github.com/haadid/MachineLearningTerapan-Dicoding/assets/92203636/84923aee-0229-45e9-9070-51ccbf6bfc1c)
       <br> **Gambar 4.** Worldcloud Model Ponsel <br>
       <br> Pada Gamabr 4, ditampilkan daftar kata-kata yang sering digunakan pada penamaan model semakin sering kata tersebut digunakan maka semakin besar kata tersebut. Terlihat kata yang paling sering digunakan adalah 'Moto G'


## Data Preparation
Berikut tahapan-tahapan dalam menyiapkan data:
- **Menggabungkan data dalam satu tabel**
  <br> Terdapat tiga berkas csv yang diunduh dari sumber laman. Penggabungan dilakukan masing-masing dengan menyesuaikan fitur yang ada pada berkas csv sebelumnya.

- **Menghapus fitur yang tidak diperlukan**
  <br> Terdapat fitur yang tidak digunakan pada dataset yang dihapus yaitu `release date` karena tidak memiliki dampak yang signifikan terhadap model.

- **Membersihkan data yang tidak valid**
  <br> Pada kolom `model`, terdapat penamaan yang tidak sesuai standar sehingga harus diubah. Pada kolom `gender` terdapat entri yang tidak sesuai antara pria atau wanita sehingga harus dihapus. Untuk kolom `rating`, terdapat nilai yang tidak sesuai rentang skala sehingga harus dihapus.

- **Persiapan data untuk Content Based Filtering**
    -   **Seleksi Fitur**
        <br> Fitur-fitur yang akan digunakan untuk *Content-Based Filtering( dipilih. Fitur yang dipilih akan disimpan kedalam list bernama **fitur**.
    -   **Penggabungan Fitur**
        <br> Melakukan penggabungan fitur yang telah dipilih pada setiap baris (row) dalam dataset untuk membentuk satu kolom baru yang berisi representasi gabungan fitur tersebut. 
    -   **Pembuatan Matriks Hitung**
        <br> Kemudian menggunakan fungsi [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) dari modul *scikit-learn* untuk mengonversi teks yang telah digabungkan menjadi matriks hitungan. Matriks ini mengukur frekuensi kemunculan setiap kata pada seluruh dataset.
    -   **Perhitungan Kemiripan Kosinus**
        <br> Matriks kemiripan kosinus dihitung dari matriks hitungan yang telah dibuat dengan memanfaatkan fungsi [cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) dari modul *scikit-learn*. Hal ini dilakukan untuk mengukur seberapa mirip setiap ponsel dengan yang lain berdasarkan fitur yang telah dipilih.
   
- **Persiapan data untuk Collaborative Filtering dengan KNN**
    -   **Mengubah format data menjadi pivot tabel**
        <br> Sebelum melanjutkan ke pembuatan model rekomendasi menggunakan KNN untuk sistem rekomendasi ponsel, perlu dilakukan transformasi data untuk memenuhi format yang dibutuhkan oleh model. Data rating ponsel akan diubah menjadi matriks dengan ukuran m x n, di mana m merupakan jumlah ponsel dan n merupakan jumlah pengguna. Transformasi ini dilakukan dengan membuat pivot tabel menggunakan modul [pivot_table](https://pandas.pydata.org/docs/reference/api/pandas.pivot_table.html) dari pandas. Pada pivot tabel ini, judul ponsel akan menjadi indeks, id pengguna akan menjadi kolom, dan nilai rating akan menjadi entri pada setiap sel tabel.
        <br> <br> Berikut adalah hasil *pivot table* yang dibentuk:
        ![hasil_pivot_table](https://user-images.githubusercontent.com/92203636/285189933-8a59be56-3bc8-4350-b8e8-a7912d356f77.png)
        <br> **Gambar 5.** Pivot Tabel Untuk Model KNN

- **Persiapan data untuk Collaborative Filtering dengan Deep Learning**. Persiapan data melibatkan langkah-langkah seperti:
    -   **Encoding Fitur User dan Ponsel (Cellphone)**
        <br> Dilakukan proses encoding pada fitur `user_id` dan `cellphone_id` untuk mengubah data non-numerik menjadi representasi numerik. Pada tahap ini, fungsi *enumerate* dimanfaatkan untuk memberikan indeks integer unik untuk setiap `user_id` dan `cellphone_id`. Proses ini memungkinkan model untuk memproses data tersebut secara efektif. Hasil encoding tersebut kemudian dipetakan kembali ke dataframe yang relevan, menciptakan representasi numerik dari entitas user dan ponsel.

    -   **Pembagian Data untuk Pelatihan dan Validasi**
        <br> Langkah selanjutnya adalah pembagian dataset menjadi dua bagian utama: data pelatihan dan data validasi. Sebelum pembagian dilakukan, dataset diacak terlebih dahulu untuk memastikan distribusi yang random. Variabel x dibuat untuk menggabungkan data user dan ponsel menjadi satu nilai, sedangkan variabel y dibuat untuk menangkap peringkat (rating) hasil rekomendasi. Proses pembagian dilakukan dengan alokasi 70% data untuk pelatihan dan 30% data untuk validasi. Dari proses pembagian data, diperoleh 671 sampel untuk data pelatihan dan 288 sampel untuk data validasi. 

## Modeling
Pada proyek ini, model yang akan dibuat berupa sistem rekomendasi untuk merekomendasikan ponsel pintar kepada pengguna. Pada proyek ini sistem rekomendasi yang dibuat menggunakan teknik *content based filtering* dan *collaborative filtering* dengan menggunakan 2 pendekatan yaitu pendekatan Item-Based dengan algoritma *K-Nearest Neighbor* dan pendekatan *Deep learning* atau *Neural Network*.

- **Content Based Filtering**
  <br> Untuk membangun model Content-Based Filtering, langkah pertama melibatkan persiapan data dengan memilih fitur-fitur utama yang akan menjadi dasar rekomendasi. Selanjutnya, fitur-fitur yang dipilih digabungkan ke dalam satu kolom baru yang diberi nama `combinedFeatures`. Proses penggabungan ini melibatkan penyatuan nilai-nilai fitur tersebut ke dalam satu string yang mencerminkan karakteristik ponsel secara menyeluruh. Hasil penggabungan ini akan menjadi dasar untuk mengukur kesamaan antarponsel. Dalam langkah mengukur kesamaan, *CountVectorizer* digunakan untuk mengonversi teks pada kolom `combinedFeatures` menjadi vektor numerik. Hal ini memungkinkan perhitungan kemiripan antarponsel. Selanjutnya menghitung matriks kemiripan menggunakan kemiripan kosinus. Matriks ini memberikan nilai kemiripan antara setiap pasang ponsel dalam dataset. Dengan matriks kemiripan yang dihasilkan, fungsi rekomendasi Content-Based Filtering dapat dibuat. Fungsi ini mempertimbangkan indeks model tertentu dan menghasilkan rekomendasi ponsel berdasarkan kesamaan fitur. Rekomendasi ini dapat disesuaikan dengan kebutuhan, misalnya, menampilkan 10 ponsel teratas. Dengan langkah-langkah tersebut, model Content-Based Filtering dapat memberikan rekomendasi ponsel berdasarkan kemiripan fitur-fitur yang dipilih.
  <br>
  <br> Berikut hasil rekomendasi *Content Based Filtering* dengan *Cosine Similarity*:
  
	|   | Top 5 Rekomendasi ponsel berdasarkan model 10T: |                 |     |             |                 |       |
	|---|-------------------------------------------------|-----------------|-----|-------------|-----------------|-------|
	|   |                 operating system                | internal memory | RAM | main camera |      model      | score |
	| 0 | Android                                         | 128             | 8   | 50          | Pixel 6         | 1.0   |
	| 1 | Android                                         | 128             | 8   | 50          | Nord 2T         | 1.0   |
	| 2 | Android                                         | 128             | 8   | 50          | Galaxy S22 Plus | 1.0   |
	| 3 | Android                                         | 128             | 6   | 50          | Moto G Stylus   | 1.0   |
	| 4 | Android                                         | 128             | 8   | 50          | 12 Pro          | 1.0   |

	**Tabel 2.** Hasil Rekomendasi Content Based Filtering
  
    -   Kelebihan:
	    -   Mampu memberikan rekomendasi berdasarkan fitur yang disukai pengguna.
	    -   Efektif untuk mengatasi cold start problem.
	    -   Pemahaman terhadap preferensi individual pengguna.
    -   Kekurangan:
	    -   Rentan terhadap over-specialization, di mana rekomendasi hanya didasarkan pada preferensi sebelumnya tanpa variasi.
	    -   Bergantung pada kualitas deskripsi dan representasi fitur dari item.
	    -   Tidak efektif pada fitur yang banyak.
	    -   Kurang akurat jika suatu fitur memiliki banyak sekali keberagaman.

- **Collaborative Filtering dengan KNN**
  <br> Untuk membangun model ini, digunakan fungsi NearestNeighbors dari sklearn dengan parameter metrik 'cosine'. Algoritma ini menghitung kesamaan cosinus antara vektor rating. Penggunaan algoritma 'brute' pada parameter mengindikasikan bahwa algoritma akan menghitung tetangga terdekat dengan mencari kesamaan langsung dengan seluruh data. Model ini kemudian diinisialisasi sebagai model_knn dan difitting terhadap data yang telah diubah menjadi pivot table. <br>
  <br> Setelah tahap inisialisasi dan fitting, dibuat fungsi recommend_cellphone untuk memberikan rekomendasi terhadap suatu model ponsel pintar. Hasil rekomendasi ini disajikan dengan menyertakan model ponsel yang memiliki kesamaan dengan model yang diberikan. <br>
  <br> Berikut adalah hasil rekomendasi *Collaborative Filtering* dengan KNN:
  
	| 10 Rekomendasi untuk ponsel Xperia Pro sebagai berikut : |
	|----------------------------------------------------------|
	| 0: Nord N20, with distance of 0.7391863503412192         |
	| 1: Find X5 Pro, with distance of 0.7221312781196743      |
	| 2: 10T, with distance of 0.706235262124101               |
	| 3: Redmi Note 11, with distance of 0.6847865811144636    |
	| 4: Moto G Pure, with distance of 0.6800465276651233      |
	| 5: Pixel 6a, with distance of 0.6640232136428437         |
	| 6: Moto G Play, with distance of 0.6449885098433434      |
	| 7: Pixel 6, with distance of 0.620879067430839           |
	| 8: X80 Pro, with distance of 0.5922716936733281          |
	| 9: Galaxy A13, with distance of 0.5309814131625528       |

  	**Tabel 3.** Hasil Rekomendasi Collaborative Filtering Dengan KNN

    -   Kelebihan:
	    -   Mampu memberikan rekomendasi berdasarkan perilaku dan preferensi serupa antar pengguna.
	    -   Dapat menangani cold start problem melalui item-based collaborative filtering.
	    -   Tidak memerlukan informasi eksplisit mengenai item.
    -   Kekurangan:
	    -   Rentan terhadap masalah sparsity pada data pengguna-item.
	    -   Performa dapat menurun jika jumlah pengguna dan item sangat besar.
	    -   Tidak dapat menangani perubahan cepat dalam preferensi pengguna.
  
- **Collaborative Filtering dengan Deep Learning**
  <br> Untuk mengembangkan model ini, digunakan metode Deep Learning atau Jaringan Saraf. Model yang dikonstruksi akan menghitung skor kesesuaian antara pengguna dan ponsel dengan menggunakan teknik embedding. Proses pertama melibatkan embedding data pengguna dan ponsel. Selanjutnya, operasi perkalian dot product dilakukan antara embedding pengguna dan ponsel. Terdapat juga penambahan bias untuk setiap pengguna dan ponsel. Skor kesesuaian didefinisikan dalam rentang [0,1] dengan menggunakan fungsi aktivasi sigmoid. Model dengan pendekatan Deep Learning ini dirancang dengan membuat kelas RecommenderNet menggunakan kelas Model dari Keras. Proses kompilasi model melibatkan Binary Crossentropy untuk menghitung fungsi kerugian, Adam (Adaptive Moment Estimation) sebagai pengoptimalkan, dan root mean squared error (RMSE) sebagai evaluasi metrik. Langkah selanjutnya adalah melakukan proses pelatihan terhadap model. <br>
  <br> Untuk mendapatkan rekomendasi ponsel, pertama-tama diambil sampel pengguna secara acak dan ditetapkan variabel cellphones_not_rated, yang merupakan daftar ponsel yang belum pernah diulas oleh pengguna. Daftar cellphones_not_rated ini kemudian menjadi kandidat ponsel yang direkomendasikan. Variabel cellphones_not_rated diperoleh dengan menggunakan operator bitwise (~) pada variabel cellphones_rated_by_user. Sebelumnya, pengguna telah memberikan rating pada beberapa ponsel yang telah mereka lihat. Informasi rating ini digunakan untuk menyusun rekomendasi ponsel yang dapat sesuai dengan preferensi pengguna. Untuk mendapatkan rekomendasi ponsel, digunakan fungsi model.predict() dari pustaka Keras. <br>
  <br> Berikut adalah hasil rekomendasi *Collaborative Filtering* dengan *Deep Learning*:
  
	| Showing recommendations for user: 255  |
	|----------------------------------------|
	| ====================================   |
	| Cellphones with high ratings from user |
	| --------------------------------       |
	| iPhone 13 Pro Max : Apple              |
	| Galaxy Z Flip 3 : Samsung              |
	| 11T Pro : Xiaomi                       |
	| Galaxy Z Fold 3 : Samsung              |
	| Zenfone 8 : Asus                       |
	| --------------------------------       |
	| Top 10 Cellphones recommendations      |
	| --------------------------------       |
	| iPhone XR : Apple                      |
	| Galaxy S22 : Samsung                   |
	| X80 Pro : Vivo                         |
	| Find X5 Pro : Oppo                     |
	| iPhone 13 Pro : Apple                  |
	| Pixel 6 : Google                       |
	| Xperia Pro : Sony                      |
	| Pixel 6 Pro : Google                   |
	| 10 Pro : OnePlus                       |
	| iPhone 13 : Apple                      |

  	**Tabel 4.** Hasil Rekomendasi Collaborative Filtering Dengan Deep Learning

    -   Kelebihan:
	    -   Mampu menangani pola kompleks dalam data pengguna dan item.
	    -   Dapat memberikan rekomendasi yang lebih personal dan akurat.
	    -   Tidak terlalu bergantung pada representasi fitur yang didefinisikan secara manual.
    -   Kekurangan:
	    -   Memerlukan jumlah data historis yang cukup untuk pelatihan model deep learning.
	    -   Pemrosesan yang lebih intensif secara komputasional, terutama pada model deep learning.
	    -   Rentan terhadap overfitting jika data pelatihan tidak cukup diversifikasi.


## Evaluation
Pada proyek ini terdapat dua model yang akan dievaluasi. Model *Content Based Filtering* akan dievaluasi menggukanan metrik *Precision Recommender System* dan model *Collaborative Filtering* dengan metode *Deep Learning* akan dievaluasi menggunakan metrik *Root Mean Squared Error* (RMSE). 

### Content Based Filtering dengan Cosine Similarity
*Precision* atau presisi adalah metrik yang mengukur keberhasilan model dalam memberikan rekomendasi yang relevan. Nilai presisi didapatkan dengan menghitung jumlah rekomendasi yang relevan dibagi dengan jumlah rekomendasi yang diberikan.
- **Precision** 
  <br> Perumusan Presisi adalah sebagai berikut:
  $${Precision} = \frac{\text{Number of Relevant Recommendations}}{\text{Total Number of Recommendations}}$$ <br>

Dari hasil rekomendasi model *Content Based Filtering*, terdapat 5 rekomendasi dan hanya 4 yang relevan terhadap *item* yang dicari. Sehingga presisi akhir dari model adalah 80%.


### Collaborative Filtering dengan Deep Learning
*Root Mean Square Error* (RMSE) adalah metrik evaluasi yang umum digunakan untuk mengukur seberapa baik model regresi dapat memprediksi nilai. RMSE dihitung dengan mengukur perbedaan antara nilai prediksi dan nilai sebenarnya. Dalam konteks ini, nilai RMSE yang lebih kecil menunjukkan tingkat akurasi yang lebih tinggi. RMSE mengukur seberapa besar deviasi antara nilai prediksi model dengan nilai sebenarnya. Metrik evaluasi ini terkait erat dengan Mean Squared Error (MSE), yang mengukur rata-rata dari kuadrat selisih antara nilai prediksi dan nilai sebenarnya. MSE dihitung dengan menjumlahkan selisih kuadrat antara setiap nilai prediksi dan nilai sebenarnya, kemudian hasilnya dibagi dengan jumlah total observasi.

Untuk menghitung RMSE, hasil perhitungan MSE kemudian diakar kuadratkan untuk mendapatkan nilai akhir RMSE. Hasil RMSE yang rendah menunjukkan bahwa model memiliki kemampuan yang baik dalam memprediksi peringkat pengguna terhadap item, dan rekomendasi yang dihasilkan cenderung lebih sesuai dengan preferensi pengguna. Sebaliknya, nilai RMSE yang tinggi menunjukkan bahwa model memiliki tingkat kesalahan yang signifikan dalam memprediksi peringkat, dan rekomendasi mungkin tidak sesuai dengan preferensi pengguna secara akurat. <br>

- **MSE**
  <br> Perumusan MSE adalah sebagai berikut:
  $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
  
- **RMSE**
  <br> Perumusan RMSE adalah sebagai berikut:
  $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$ <br>


Berikut merupakan visualisasi metrik pada training terhadap model *deep learning* : <br>
![hasil_training](https://github.com/haadid/MachineLearningTerapan-Dicoding/assets/92203636/b0caecab-06ad-47a2-9277-328a740819d9)
<br> **Gambar 6.** Plot Nilai RMSE Pada Pelatihan Deep Learning

Pada Gambar 6, Proses training model cukup smooth dan model konvergen pada epochs. Hasil pelatihan tampaknya berada dalam kondisi *good fit*. Performa pada data validasi tidak menunjukkan tanda-tanda *overfitting*, dan RMSE yang rendah pada kedua dataset menunjukkan kemampuan model dalam memprediksi peringkat dengan baik. Dari proses ini, diperoleh nilai error akhir sebesar sekitar 0.2040 dan error pada data validasi sebesar 0.2944. Nilai tersebut cukup bagus untuk sistem rekomendasi. <br>

<br> ![hasil_evaluate](https://github.com/haadid/MachineLearningTerapan-Dicoding/assets/92203636/d71e24dc-ea3f-4809-9e05-7b6647dcde99)
<br> **Gambar 7.** Evaluasi Akhir Model Deep Learning

Pada Gambar 7, dilakukan evaluasi menggunakan seluruh data memperoleh nilai error sebesar 0.3281.


## Kesimpulan
Dari hasil evaluasi, dapat disimpulkan bahwa kedua model, baik Content-Based Filtering maupun Collaborative Filtering, memberikan kinerja yang baik dalam memberikan sistem rekomendasi. Hasil dari model *Content Based Filtering* menunjukkan tingkat presisi yang tinggi sebesar 80%. Presisi yang tinggi pada model Content-Based Filtering menunjukkan kecakapan model dalam menghasilkan rekomendasi yang sesuai dengan preferensi pengguna. Kemudian setelah melakukan evaluasi terhadap model *Collaborative Fitlering* dengan metode *Deep Learning*, nilai RMSE yang dijadikan sebagai metrik pada model adalah 0.3281. Model dengan nilai tersebut memiliki tingkat akurasi yang baik dalam memberikan rekomendasi.


## Penutup
Sistem rekomendasi dengan model *Content Based Filtering* dan *Collaborative Filtering* telah selesai dibuat. Pada model *Content Based Filtering* dibuat menggunakan metode *cosine similarity*. Sedangkan model *Collaborative Filtering* menggunakan dua pendekatan, yaitu *K-Nearest Neighbor* dan *Deep Learning*.


## Referensi
Dewi, K., & Ciptayani, P. (2022). [PEMODELAN SISTEM REKOMENDASI CERDAS MENGGUNAKAN HYBRID DEEP LEARNING. _Jurnal Sistem Informasi dan Sains Teknologi, 4_(2)](https://doi.org/10.31326/sistek.v4i2.1157)

Muhammad Haris Diponegoro, Sri Suning Kusumawardani, & Indriana Hidayah. (2021).  [Implementation of Deep Learning Methods in Predicting Student Performance: A Systematic Literature Review. _Jurnal Nasional Teknik Elektro Dan Teknologi Informasi_, _10_(2), 131-138.](https://doi.org/10.22146/jnteti.v10i2.1417)

Theodorus, D., Defit, S., & Nurcahyo, G. W. (2021). [Machine Learning Rekomendasi Produk dalam Penjualan Menggunakan Metode Item-Based Collaborative Filtering. _Jurnal Informasi Dan Teknologi_, _3_(4), 202-208.](https://doi.org/10.37034/jidt.v3i4.151)

Prayogo, J., Suharso, A., & Rizal, A. (2020). [Analisis Perbandingan Model Matrix Factorization dan K-Nearest Neighbor dalam Mesin Rekomendasi Collaborative Berbasis Prediksi Rating. _Jurnal Informatika Universitas Pamulang, 5_(4), 506-514.](http://dx.doi.org/10.32493/informatika.v5i4.7379)

Romindo, Jefri Junifer Pangaribuan, Okky Putra Barus, & Jusin. (2022) [Penerapan Metode Collaborative Filtering Dan Knowledge Item Based Terhadap Sistem Rekomendasi Kamera DSLR. _SATIN - Sains Dan Teknologi Informasi_, _8_(2), 89-100.](https://doi.org/10.33372/stn.v8i2.883)

Fiarni, C., & Maharani, H. (2019). [Product Recommendation System Design Using Cosine Similarity and Content-based Filtering Methods. _IJITEE (International Journal of Information Technology and Electrical Engineering), 3_(2), 42-48.](https://doi.org/10.22146/ijitee.45538)
