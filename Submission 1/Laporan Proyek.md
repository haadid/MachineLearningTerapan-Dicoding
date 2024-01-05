# Laporan Proyek Machine Learning - Marwan Hadid

## Domain Proyek

Domain proyek ini mengambil tema kesehatan dan befokus pada prediksi risiko diabetes. Judul proyek ini adalah "Prediksi Risiko Diabetes"

### Latar Belakang
Diabetes melitus, yang umumnya disebut diabetes, adalah penyakit kronis yang memengaruhi jutaan orang di seluruh dunia. Diabetes terjadi akibat gangguan metabolisme yang memengaruhi kemampuan tubuh untuk mengatur kadar gula darah. Hal ini sering disebabkan oleh penurunan produksi insulin oleh pankreas, yang mengakibatkan hiperglikemia, yaitu peningkatan kadar gula darah dalam tubuh.

Diabetes memiliki dampak serius pada kesehatan individu, salah satunya adalah masalah jantung, gangguan penglihatan, kerusakan ginjal, dan komplikasi lainnya. Oleh karena itu, penting untuk mengidentifikasi individu yang berisiko tinggi mengembangkan diabetes agar tindakan pencegahan dan manajemen yang tepat dapat dilakukan.

Melalui proyek ini, teknik dan algoritma dalam *machine learning* dimanfaatkan untuk memprediksi risiko diabetes berdasarkan atribut kesehatan individu. Dengan cara ini, model *machine learning* dapat mengidentifikasi individu yang berisiko tinggi dan menyediakan intervensi yang sesuai untuk mencegah perkembangan penyakit ini.

### Urgensi dan Relevansi
1. **Masalah Kesehatan Global**
   <br> Diabetes adalah masalah kesehatan global yang memengaruhi banyak individu dari berbagai latar belakang dan usia. Prediksi risiko diabetes dapat memberikan pelayanan perawatan yang lebih optimal dan bermanfaat.
2.  **Pencegahan Komplikasi**
    <br> Diabetes dapat menyebabkan berbagai komplikasi serius, termasuk kerusakan organ. Identifikasi dini individu yang berisiko dapat membantu mencegah komplikasi ini.
3.  **Efisiensi Perawatan Kesehatan**
    <br> Proyek ini juga dapat membantu sistem perawatan kesehatan dalam mengalokasikan sumber daya dengan lebih efisien. Pasien dengan risiko tinggi dapat mendapatkan perawatan yang lebih intensif, sementara yang memiliki risiko rendah dapat dipantau dengan cara yang lebih sederhana.
4.  **Pendukung Penelitian Medis**
    <br> Model prediksi risiko diabetes juga membantu para peneliti medis dalam mengidentifikasi faktor-faktor yang berkontribusi terhadap penyakit ini. Hal ini memberikan wawasan yang berharga untuk penelitian lebih lanjut dalam pencegahan dan pengobatan diabetes.


## Business Understanding

### Problem Statements

Berdasarkan latar belakang diatas, permasalahan yang dapat diselesaikan pada proyek ini adalah sebagai berikut:
-   Bagaimana fitur-fitur pada dataset menjadi risiko terjadinya penyakit diabetes?
-   Bagaimana cara melakukan pra-pemrosesan data agar dapat digunakan untuk membangun model yang baik?
-   Bagaimana cara membangun model *machine learning*  untuk mengklasifikasi risiko penyakit diabetes berdasarkan fitur-fitur tersebut?


### Goals
Tujian dari pernyataan masalah adalah sebagai berikut:
- 	Mengidentifikasi faktor-faktor utama yang berkaitan dengan kemungkinan risiko diabetes.
- 	Melakukan tahap prapmrosesan data dan memilah fitur yang berkaitan dengan penyakit diabetes.
- 	Mengembangkan model *machine learning* yang dapat mengklasifikasikan risiko diabetes berdasarkan atribut-atribut kesehatan.


### Solution statements
Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini diantaranya :
- **Pra-pemrosesan Data**. Pada pra-pemrosesan data dapat dilakukan beberapa tahapan, antara lain :
    -   Mengatasi anomali pada data dengan menghilangkan nilai dibawah batas kuartil bawah.
    -   Pemisahan fitur dan label.
    -   Pembagian dataset.
    -   Standardisasi pada fitur numerik dataset.

- **Pembangungan Model**. Dalam membangun model terdapat beberapa algoritma yang akan digunakan, antara lain:
    -   ***Logistic Regression***
        <br> *Logistic Regression* atau Regresi Logistik adalah algoritma yang digunakan untuk masalah klasifikasi. Algoritma ini bekerja dengan memodelkan hubungan antara variabel input (fitur) dan probabilitas bahwa input tersebut termasuk dalam satu dari dua kelas (misalnya, 0 atau 1). Hasilnya adalah prediksi berupa probabilitas, yang kemudian bisa diubah menjadi prediksi biner dengan menggunakan ambang tertentu.

        Kelebihan dan kekurangan algoritma Regresi Logistik adalah:
        -   Kelebihan :
            -   Sederhana dan mudah dimengerti.
            -   Cocok untuk masalah klasifikasi binomial.
            -   Menghasilkan probabilitas sebagai output.
        -   Kekurangan :
            -   Hanya sesuai untuk masalah klasifikasi binomial.
            -   Kurang mampu menangani hubungan non-linear antara fitur dan target.
            -   Rentan terhadap overfitting jika jumlah fitur terlalu besar atau data sangat tidak seimbang.

    -   ***K-Nearest Neighbor***
        <br> *K-Nearest Neighbor* atau KNN adalah algoritma yang digunakan dalam kasus klasifikasi dan regresi. Algoritma ini bekerja berdasarkan prinsip dasar "kedekatan" atau "jarak" antara data. Dalam algoritma KNN, ketika ingin memprediksi kelas atau nilai target dari suatu titik data yang baru, algoritma akan mencari K tetangga terdekat dari titik tersebut dalam data pelatihan. Nilai K dalam KNN adalah jumlah tetangga terdekat yang akan digunakan untuk membuat prediksi. Namun, pemilihan nilai K yang tepat dan penanganan skala fitur yang baik adalah pertimbangan penting saat menggunakannya. Berikut tahapan kerja algoritma KNN:
        - Tentukan nilai K (jumlah tetangga).
        - Hitung jarak antara data yang akan diprediksi dan semua data pelatihan.
        - Urutkan data pelatihan berdasarkan jarak terdekat. dan tetapkan jumlah tetangga sesuai dengan nilai K.
        - Pilih sejumlah K data dengan jarak terdekat.
        - Untuk masalah klasifikasi, tentukan kelas mayoritas dari K tetangga tersebut sebagai prediksi.

        Kekurangan algoritma *K-Nearest Neighbor* adalah:
        -   Kelebihan :
            -   Mudah diimplementasikan.
            -   Kemampuan beradaptasi dengan perubahan dataset.
            -   Memiliki _hyperparameter_ yang sedikit.
        -   Kekurangan :
            -   Tidak cocok untuk dataset berukuran besar dan berdimensi tinggi.
            -   Sensitif terhadap _noise_, _missing value_, dan _outlier_.

    -   ***Support Vector Machine***
        <br> *Support Vector Machine* (SVM) adalah algoritma pembelajaran mesin yang digunakan untuk masalah klasifikasi dan regresi. SVM mencari hyperplane yang memisahkan dua kelas dengan margin maksimum. Dalam kasus klasifikasi, SVM mencoba menemukan hyperplane terbaik yang memisahkan dua kelas dengan jarak (margin) sebesar mungkin.

        Kelebihan dan kekurangan algoritma *Support Vector Machine* adalah:
        -   Kelebihan :
            -   Efektif dalam menangani data dengan banyak dimensi (fitur).
            -   Bekerja baik dalam kasus kelas yang tidak seimbang.
            -   Dapat menangani masalah klasifikasi non-linear dengan menggunakan kernel.
        -   Kekurangan :
            -   Memerlukan pemilihan kernel yang tepat.
            -   Memerlukan penyetelan parameter (seperti parameter penalti C) yang tepat.
            -   Pelatihan SVM bisa memakan waktu jika dataset besar.

    -   ***Random Forest***
        <br> *Random Forest* adalah algoritma *ensemble* yang terdiri dari beberapa pohon keputusan. Setiap pohon dalam hutan membuat prediksi, dan hasil dari beberapa pohon digabungkan untuk menghasilkan prediksi yang lebih akurat. Metode Ini akan mengurangi masalah overfitting yang sering terjadi dalam pohon keputusan tunggal.

        Kelebihan dan kekurangan algoritma *Random Forest* adalah:
        -   Kelebihan :
            -   Efektif dalam menangani data dengan banyak dimensi (fitur).
            -   Bekerja baik dalam kasus kelas yang tidak seimbang.
            -   Dapat menangani masalah klasifikasi non-linear dengan menggunakan kernel.
        -   Kekurangan :
            -   Memerlukan pemilihan kernel yang tepat.
            -   Memerlukan penyetelan parameter (seperti parameter penalti C) yang tepat.
            -   Pelatihan SVM bisa memakan waktu jika dataset besar.

    -   ***Extreme Gradient Boost***
        <br> *XGBoost* atau *eXtreme Gradient Boosting* adalah implementasi open-source yang populer dan efisien dari algoritma pohon yang ditingkatkan gradien. Peningkatan gradien adalah algoritma pembelajaran yang diawasi yang mencoba memprediksi variabel target secara akurat dengan menggabungkan ansambel perkiraan dari serangkaian model yang lebih sederhana dan lebih lemah. Algoritma *XGBoost* bekerja baik dalam kompetisi machine learning karena penanganannya yang kuat dari berbagai jenis data, hubungan, distribusi, dan variasi hyperparameter yang dapat disesuaikan sendiri. XGBoost dapat digunakan untuk kasus regresi, klasifikasi (biner dan multiclass), dan masalah peringkat.

        Kelebihan *XGBoost* adalah:
        -   Kelebihan :
            -   Kinerja unggulnya menghasilkan prediksi yang sangat akurat.
            -   Menyediakan fitur regularisasi yang membantu mencegah overfitting, membuatnya lebih stabil.
            -   Dapat memproses komputasi secara paralel, yang sangat berguna untuk dataset besar.
            -   Dilengkapi dengan built-in _cross-validation_, memudahkan proses evaluasi model.

## Data Understanding
- **Informasi Dataset**
  <br> Dataset yang digunakan pada proyek ini bernama `Healthcare Diabetes Dataset`. Berikut informasi lebih lanjut mengenai dataset:

   | Jenis           | Keterangan                                                                                              |
   |-----------------|---------------------------------------------------------------------------------------------------------|
   | Sumber          | Healthcare Diabetes Dataset - [Kaggle](https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes) |
   | Pemilik         | [Nandita Pore](https://www.kaggle.com/nanditapore)                                                      |
   | Lisensi         | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)                                               |
   | Kategori        | Exploratory Data Analysis, Diabetes, Healthcare, Regression, Binary Classification                      |
   | Tipe dan Ukuran | CSV (98.57 kB)                                                                                          |

  Berikut informasi lebih lanjut mengenai berkas dataset yang telah diunduh:
    - Tidak terdapat *missing value* dan duplikasi pada dataset.
    - Terdapat 2768 baris data dan 10 kolom pada dataset.
    - Semua fitur termasuk dalam tipe data numerik yaitu int64 dan 2 sisanya float64.
    - Terdapat 1 kolom terakhir bernama `Outcome` yang memiliki nilai antara 1 sebagai adanya diabetes dan 0 sebagai tidak adanya diabetes. Kolom ini merupakan kolom target prediksi.
    - Fitur-fitur pada dataset adalah sebagai berikut:
        - `Id`: Penomoran unik untuk tiap data.
        - `Pregnancies`: Banyak kali hamil.
        - `Glucose`: Konsentrasi plasma glukosa selama 2 jam dalam tes toleransi glukosa.
        - `BloodPressure`: Tekanan darah diastolik (mm Hg).
        - `SkinThickness`: Ketebalan lapisan kulit trisep (mm).
        - `Insulin`: Insulin serum 2 jam (mu U/ml).
        - `BMI`: _Body Mass Index_ atau indeks massa tubuh (berat badan dalam kg / tinggi badan dalam m^2).
        - `DiabetesPedigreeFunction`: Fungsi silsilah diabetes, skor genetik diabetes.
        - `Age`: Umur dalam satuan tahun.
        - `Outcome`: Klasifikasi yang biner menentukan ada (1) atau tidak adanya (0) diabetes.

- **Exploratory Data Analysis**
    - **Analisis Outlier**
      <br> Fitur-fitur yang menjadi tolak ukur suatu penyakit diabetes yaitu `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, dan `Age` akan dianalisa distribusi datanya untuk mengetahui persebaran data dan mendeteksi adanya *outlier*.
      ![outlier](https://user-images.githubusercontent.com/92203636/274642804-dca237e8-1527-404b-b656-6c85104dbcc6.png)
      <br> Gambar 1. Analisis *Outlier* <br>
      <br> Pada Gambar 1, sayangnya, semua fitur memiliki *outlier*. *Outlier* merupakan nilai pada suatu fitur yang berada lebih tinggi atau lebih rendah dari nilai dalam fitur tersebut pada umumnya. Fitur `Glucose`, `BloodPressure`, dan`BMI` memiliki anomali pada angka 0 karena tidak mungkin adanya fitur dengan nilai tersebut. Fitur `Insulin` dan `DiabetesPedigreeFunction` memiliki outlier terbanyak.

    - **Distribusi Data**
      <br> Berikut merupakan distribusi data dengan fitur-fitur yang sama.
      ![distribusi](https://user-images.githubusercontent.com/92203636/274645518-46f0f177-ae15-4056-809a-145e849b238f.png)
      <br> Gambar 2. Distribusi Data <br>
      <br> Pada Gambar 2, distribusi data kebanyakan tidak normal. Hanya fitur `Glucose` dan `BloodPressure` yang terlihat memiliki distribusi normal meskipun terdapat outlier dinilai 0 pada kedua fitur. Sedangkan fitur lainnya cenderung _right-skewed_. Nilai data 0 pada `Glucose` dan `BloodPressure` akan di _drop_ karena merupakan suatu anomali.

    - **Keseimbangan Data Prediksi** <br>
      <br> ![outcome](https://user-images.githubusercontent.com/92203636/274654637-ede4894a-563b-480f-8af1-d5eeb08fe726.png)
      <br> Gambar 3. Analisis Keseimbangan Data Prediksi <br>
      <br> Pada Gambar 3, data klasifikasi biner tidak seimbang pada fitur `Outcome`. Data tidak adanya diabetes (0) lebih banyak setengah dari data ada diabetes (1).

    -  **Hubungan antara Fitur** <br>
       <br> ![pairplot](https://user-images.githubusercontent.com/92203636/274648451-73211ad7-c831-4dee-a9e4-fc6d1ae1fea6.png)
       <br> Gambar 4. _Pairplot_ Untuk Setiap Fitur <br>
       <br> Pada Gambar 4, tidak terdapat informasi yang relevan untuk mengetahui hubungan antar fitur pada _pairplot_, baik hubungan positif maupun negatif.

    -  **Korelasi antara Fitur** <br>
       <br> ![corrplot](https://user-images.githubusercontent.com/92203636/274657001-21a699ab-d03b-48f5-92c9-2f3ceb27b6fa.png)
       <br> Gambar 5. _Heatmap_ Korelasi Setiap Fitur <br>
       <br> Pada Gambar 5, matriks korelasi menunjukkan adanya hubungan terkait dengan fitur yang ada. Fitur pada `Glucose` (0.49), `BMI` (0.27), `Age` (0.25), dan `Pregnancies` (0.23) terlihat mempunyai korelasi terhadap `Outcome` sebagai penentu ada atau tidaknya diabetes dibandingkan fitur lainnya. Sedangkan `BloodPressure` (0.18), `DiabetesPedigreeFunction` (0.17), dan `Insulin` (0.14) juga memiliki korelasi terhadap `Outcome` meskipun dengan nilai lebih kecil. Sementara itu ada beberapa fitur menarik yang saling berkorelasi namun tidak memiliki korelasi tinggi pada fitur `Outcome` yaitu `Age` dan `Pregnancies` (0.55), `Insulin` dan `SkinThickness` (0.43), `BMI` dan `SkinThickness` (0.39), `Age` dan `BloodPressure` (0.33), `Glucose` dan `Insulin` (0.33), dst.

       Dikarenakan model hanya akan memprediksi `Outcome` sebagai ada tidaknya diabetes, maka hanya fitur yang memliki korelasi signifikan yaitu `Pregnancies`, `Glucose`, `Insulin`, `BMI`, dan `Age` akan digunakan sebagai penentu prediksi model. Fitur yang memiliki tingkat korelasi terhadap `Outcome` akan dianalisa lebih lanjut.

    -  Hubungan Outcome Dengan Berbagai Fitur Penentu <br> <br>
       ![glucose_outcome](https://user-images.githubusercontent.com/92203636/274666151-ab136652-020f-41b5-b2f2-2d7409295804.png)
       <br> Gambar 6. Hubungan `Glucose` dan `Outcome` <br> <br>
       
       ![bmi_outcome](https://user-images.githubusercontent.com/92203636/274666280-e217a64c-5773-4e95-b8b4-96e8966e7a32.png)
       <br> Gambar 7. Hubungan `BMI` dan `Outcome` <br> <br>
       
       ![age_outcome](https://user-images.githubusercontent.com/92203636/274666371-420162ae-7ed7-4d56-b27e-1427b3ea9512.png)
       <br> Gambar 8. Hubungan `Age` dan `Outcome` <br> <br>
       
       ![pregnancies_outcome](https://user-images.githubusercontent.com/92203636/274666435-1bcf9c21-8002-4b63-afd1-0c8c3e47fb0b.png)
       <br> Gambar 9. Hubungan `Pregnancies` dan `Outcome` <br>
       
       <br> Pada Gambar 6, Gambar 7, Gambar, 8, dan Gambar 9, nilai rata-rata penderita diabetes pada keempat fitur tersebut lebih tinggi dibandingkan yang bukan penderitanya.

## Data Preparation
Berikut tahapan-tahapan dalam menyiapkan data:
- **Menghapus data yang terindikasi anomali**
  <br> Berdasarkan obeservasi data yang dilakukan pada tahap *Data Understanding*, terdapat data yang terindikasi sebagai suatu anomali. Fitur `Glucose`, `BloodPressure`, dan `BMI` memiliki _outlier_ pada nilai 0  yang merupakan suatu anomali. Disebut anomali karena nilai 0 pada fitur-fitur tersebut mustahil ada dalam seorang manusia. Anomali ini diatasi dengan cara menghitung nilai IQR pada masing-masing fitur dan menghapus data yang berada di-IQR bawah. Setelah melakukan proses ini, jumlah sampel berkurang menjadi 2604 data.

- **Pemisahan fitur dan label**
  <br> Setelah menghapus anomali pada fitur, dataset kemudian dipisah sebagai masukan`X` yang berisi fitur-fitur dan keluaran`Y` sebagai label. masukan `X` hanya terdapat empat fitur yaitu `Pregnancies`, `Glucose`, `Insulin`, `BMI`, dan `Age` yang didasarkan pada tingkat korelasinya terhadap keluaran `Y` yaitu `Outcome`.

- **Pembagian dataset**
  <br> Untuk mengevaluasi kinerja model ketika dihadapkan dengan data yang belum pernah dilihat sebelumnya, maka langkah dilakukan selanjutnya adalah membagi dataset. Dataset akan dibagi menjadi dua, data latih dan data uji dengan rasio 80% untuk data latih dan 20% untuk data uji. Data latih adalah data yang digunakan untuk membangun kecerdasan model _machine learning_ berdasarkan fitur pada data. Data uji adalah data yang belum pernah dilihat oleh model dan digunakan untuk mengevaluasi kinerja model yang telah dibangung berdasarkan data latih. Pembagian data latih dan data uji dilakukan dengan modul [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) dari library *scikit-learn*. Setelah proeses pembagian dateset, ditemukan jumlah sampel data latih sebanyak 2083 data latih dan data uji sebanyak 521 dari total 2604 sampel data.

- **Standarisasi fitur**
  <br> Standarisasi merupakan salah satu tahapan umum yang dilakukan dalam fase persiapan data. Tujuannya adalah membuat semua fitur numerik berada dalam skala data yang sama dan membuat fitur data diolah lebih mudah oleh algoritma model. Pada dataset ini, standarisasi yang digunakan adalah [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) dari library *scikit-learn*. StandardScaler melakukan standarisasi fitur dengan cara mengurangkan nilai rata-rata dari setiap fitur dan kemudian membaginya dengan standar deviasi, sehingga mengubah distribusi data. Hasil dari penggunaan StandardScaler adalah distribusi data dengan standar deviasi setara dengan 1 dan nilai rata-rata setara dengan 0.

## Modeling
Model yang akan dibangun merupakan kasus *Binary Classification* dimana tugas model untuk mengklasifikasikan ada atau tidaknya penyakit diabetes dengan mempelajari fitur **x** dan mencocokkannya dengan label **y** yaitu Outcome. Proses pembuatan model ini menggunakan 5 algoritma *machine learning* yaitu `Logistic Regression`, `K-Nearest Neighbor`, `Support Vector Machine`, `Random Forest`, `Extreme Gradient Boost` yang kemudian akan dibandingkan performarnya masing-masing.

- **Logistic Regression**
  <br> Pembuatan model dengan algoritma Regresi Logistik menggunakan modul [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) dari *scikit-learn*. Model akan dilatih dengan data latih lalu kemudian dites dengan data uji untuk melihat performanya. Hasil pengujian menggunakan algoritma Regresi Logsitik adalah sebagai berikut:
    - Classification Report <br>
      
            |              | precision | recall | f1-score | support |
            |:------------:|:---------:|:------:|:--------:|:-------:|
            | 0            | 0.78      | 0.86   | 0.82     | 322     |
            | 1            | 0.73      | 0.60   | 0.66     | 199     |
            | accuracy     |           |        | 0.76     | 521     |
            | macro avg    | 0.76      | 0.73   | 0.74     | 521     |
            | weighted avg | 0.76      | 0.76   | 0.76     | 521     |
      Model mendapatkan akurasi sebesar 0.76

    - *Hyperparameter Tuning*
      <br> Untuk menemukan parameter yang optimal, dilakukan _hyperparameter tuning_ menggunakan [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)  yang merupakan modul dari library _scikit-learn_. GridSearchCV adalah metode pemilihan kombinasi model dan hyperparameter dengan cara menguji coba satu persatu kombinasi dan melakukan validasi untuk setiap kombinasi. Tujuannya adalah menentukan kombinasi yang menghasilkan performa model terbaik yang dapat dipilih untuk dijadikan model untuk prediksi. Hasil dari *GridSearchCV* untuk algoritma Regresi Logistik adalah sebagai berikut:
      ![logreg_gridsearch](https://user-images.githubusercontent.com/92203636/274841802-a8a24ebc-27d7-427b-91a6-112697ec1112.png)
      <br> Sayangnya, model dengan _hyperparamter tuning_ terbaik tetap memiliki nilai akurasi yang sama.

    - Confusion Matrix: <br>
      ![logreg_cm](https://user-images.githubusercontent.com/92203636/274842356-ebed4b2a-21a7-45b7-881f-ed4080191960.png)
      <br> Berdasarkan hasil pengujian diatas dapat dilihat bahwa model dengan algoritma *Logistic Regression* memperoleh nilai sebesar 0.76

- **K-Nearest Neighbor**
  <br> Model dengan algoritma KNN akan dilatih menggunakan modul [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) dari library _scikit-learn_. Nilai K awal diambil angka acak yaitu 5. Model dilatih dan kemudian diuji menggunakan data uji. Berikut hasil pengujian algoritma KNN dengan nilai K adalah 5:
    -  Classification Report
       
            |              | precision | recall | f1-score | support |
            |:------------:|:---------:|:------:|:--------:|:-------:|
            | 0            | 0.84      | 0.85   | 0.84     | 322     |
            | 1            | 0.75      | 0.73   | 0.74     | 199     |
            | accuracy     |           |        | 0.80     | 521     |
            | macro avg    | 0.79      | 0.79   | 0.79     | 521     |
            | weighted avg | 0.80      | 0.80   | 0.80     | 521     |
          Model telah mencapai akurasi yang cukup memuaskan di angka 0.80
       
    -  *Hyperparameter Tuning*
       <br> Nilai akurasi dengan algoritma masih dapat ditingkatkan, yaitu dengan metode Elbow. Tujuan metode ini adalah untuk mencari titik di mana penambahan klaster baru tidak lagi signifikan meningkatkan pemahaman atau kualitas pemisahan data dalam klaster. Metode *Elbow* dimulai dengan menentukan berbagai nilai K yang memungkinkan pada data. Selanjutnya melakukan klasterisasi pada nilai yang telah ditentukan. Lalu membuat plot untuk mengevaluasi nilai error dari tiap K yang ditentukan. Dari peneletian ini, ditemukan bahwa nilai K yang optimal adalah 1. Berikut hasilnya:
       
            |              | precision | recall | f1-score | support |
            |:------------:|:---------:|:------:|:--------:|:-------:|
            | 0            | 1.00      | 0.99   | 0.99     | 322     |
            | 1            | 0.98      | 0.99   | 0.99     | 199     |
            | accuracy     |           |        | 0.99     | 521     |
            | macro avg    | 0.99      | 0.99   | 0.99     | 521     |
            | weighted avg | 0.99      | 0.99   | 0.99     | 521     |
    -  Confusion Matrix <br>
       ![knn_cm](https://user-images.githubusercontent.com/92203636/274916695-ad83a163-89e5-4cfd-b007-252ea85c1367.png)
       <br> Berdasarkan hasil pengujian diatas dapat dilihat bahwa model dengan algoritma *K-Nearest Neighbor* memperoleh nilai sebesar 0.99

- **Support Vector Machine**
  <br> Ditahap ini, pembuatan model akan dilakukan dengan menggunakann modul [SVC](https://scikit-learn.org/stable/modules/svm.html) dari library *scikit-learn*.  Model dilatih dan diuji menggunakan data uji. Berikut hasil pengujiannya:
    -  Classification Report
 
            |              | precision | recall | f1-score | support |
            |:------------:|:---------:|:------:|:--------:|---------|
            | 0            | 0.79      | 0.89   | 0.84     | 322     |
            | 1            | 0.78      | 0.62   | 0.69     | 199     |
            | accuracy     |           |        | 0.79     | 521     |
            | macro avg    | 0.79      | 0.76   | 0.77     | 521     |
            | weighted avg | 0.79      | 0.79   | 0.79     | 521     |

          Model meraih skor akurasi 0.79

    -  *Hyperparameter Tuning*
       <br> Untuk menemukan parameter yang optimal, dilakukan _hyperparameter tuning_ menggunakan [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)  yang merupakan modul dari library _scikit-learn_. GridSearchCV adalah metode pemilihan kombinasi model dan hyperparameter dengan cara menguji coba satu persatu kombinasi dan melakukan validasi untuk setiap kombinasi. Tujuannya adalah menentukan kombinasi yang menghasilkan performa model terbaik yang dapat dipilih untuk dijadikan model untuk prediksi. Hasil dari *GridSearchCV* untuk algoritma SVM adalah sebagai berikut: <br> 
       ![svm_gridsearch1](https://user-images.githubusercontent.com/92203636/274918519-6574385c-f6b0-4ba3-a812-450bb2eb07fc.png)
       
            |              | precision | recall | f1-score | support |
            |--------------|-----------|--------|----------|---------|
            | 0            | 0.98      | 0.98   | 0.98     | 322     |
            | 1            | 0.97      | 0.97   | 0.97     | 199     |
            | accuracy     |           |        | 0.98     | 521     |
            | macro avg    | 0.98      | 0.98   | 0.98     | 521     |
            | weighted avg | 0.98      | 0.98   | 0.98     | 521     |

       Nilai akurasi mengalami penaikan menjadi 0.98
  
    -  Confusion Matrix <br>
       ![svm_cm](https://user-images.githubusercontent.com/92203636/274918646-b26b1330-ab22-4500-aa25-cee20a3d2145.png)
       <br> Berdasarkan hasil pengujian diatas dapat dilihat bahwa model dengan algoritma *Support Vector Machine* memperoleh nilai akhir sebesar 0.98 dari sebelumnya sebesar 0.79 setelah dilakukan _hyperparameter tuning_.

- **Random Forest**
  <br> Pembuatan model dengan algoritma *Random Forest* menggunakan modul [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) dari library _scikit-learn_. Model akan dilatih dengan data latih lalu kemudian dites dengan data uji untuk melihat performanya. Hasil pengujian menggunakan algoritma *Random Forest* adalah sebagai berikut:
    -  Classification Report
    
            |              | precision | recall | f1-score | support |
            |--------------|-----------|--------|----------|---------|
            | 0            |      1.00 |   0.99 |     0.99 |     322 |
            | 1            |      0.98 |   0.99 |     0.99 |     199 |
            | accuracy     |           |        |     0.99 |     521 |
            | macro avg    |      0.99 |   0.99 |     0.99 |     521 |
            | weighted avg |      0.99 |   0.99 |     0.99 |     521 |
      

    -  Confusion Matrix <br>
       ![rf_cm](https://user-images.githubusercontent.com/92203636/274968973-f9e79048-1592-4d98-ad98-babf9eef11a3.png)
       <br> Berdasarkan hasil pengujian diatas dapat dilihat bahwa model *Random Forest* memperoleh nilai sebesar 0.99

- **Extreme Gradient Boost**
  <br> Pembuatan model dengan algoritma *Random Forest* menggunakan modul [XGBClassifier](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier) dari library _xgboost_. Model akan dilatih dengan data latih lalu kemudian dites dengan data uji untuk melihat performanya. Hasil pengujian menggunakan algoritma *Random Forest* adalah sebagai berikut:
    -  Classification Report
    
            |              | precision | recall | f1-score | support |
            |--------------|-----------|--------|----------|---------|
            | 0            |      1.00 |   0.99 |     0.99 |     322 |
            | 1            |      0.98 |   0.99 |     0.99 |     199 |
            | accuracy     |           |        |     0.99 |     521 |
            | macro avg    |      0.99 |   0.99 |     0.99 |     521 |
            | weighted avg |      0.99 |   0.99 |     0.99 |     521 |


    -  Confusion Matrix
       <br> ![xgb_cm](https://user-images.githubusercontent.com/92203636/274969164-f45c1aa6-66c8-476d-a2d4-b2602966a0e1.png)
       <br> Berdasarkan hasil pengujian diatas dapat dilihat bahwa model *XGBoost* memperoleh nilai sebesar 0.99

## Evaluation
Pada proyek ini, model merupakan kasus dibidang klasifiaksi biner dan metrik evaluasi yang digunakan adalah *F1-score*, *accuracy*, *precision*, dan *recall*. Metrik evaluasi ini didapatkan dari suatu tabel bernama *Confusion Matrix*. Tabel ini digunakan untuk mengukur performa model yang membandingkan nilai prediksi dan nilai aktual atau nilai sebenarnya. Ada empat nilai yang dihasilkan dari tabel tersebut. <br>

   |                   |       |                Keadaan Data Sebenarnya                |                                                                        |
   |:-----------------:|:-----:|:-----------------------------------------------------:|:----------------------------------------------------------------------:|
   |                   |       |                          TRUE                         |                                  FALSE                                 |
   | Hasil<br>Prediksi |  TRUE | TP ( True Positive) <br>disebut juga *correct result* | FP (False Positive) <br>disebut juga *unexpected result*/*false alarm* |
   |                   | FALSE | FN (False Negative) <br>disebut juga *missing result* |         TN (True Negative)<br> disebut juga *correct rejection*        |
   <figcaption>Confusion Matrix: Memperlihatkan performa klasifikasi model.</figcaption>  </figure>

<br>Keempat nilai tersebut akan menghasilkan metrik evaluasi sebagai berikut:

- *Accuracy*
  <br> Akurasi merupakan sebagai tingkat kedekatan antara nilai prediksi dan nilai aktual. Nilai akurasi didapatkan dengan menjumlahkan prediksi positif dan prediksi :

  $$Accuracy = {{TP - TN} \over {TP + TN + FP + FN}}$$ <br>

- *Precision*
  <br> *Precision* menggambarkan jumlah data kategori positif yang diklasifikasikan secara benar dibagi dengan total data yang diklasifikasi positif. *Precision* dapat diperoleh dengan persamaan berikut :

  $$Precision = {{TP} \over {TP + FP}}$$ <br>

- *Recall*
  <br> *Recall* menunjukkan berapa persen data kategori positif yang terklasifikasikan dengan benar oleh sistem. *Recall* dapat diperoleh dengan persamaan berikut :

  $$Recall = {{TP} \over {TP + FN}}$$ <br>

- *F1-Score*
  <br> Pengertian **F1-Score** adalah *harmonic* *mean* dari *precision* dan *recall*. *F1-Score* menggambarkan perbandingan rata-rata *precision* dan *recall* yang dibobotkan. *F1-Score* dapat diperoleh dengan persamaan berikut :

  $$F1 Score = {2 * {precision * recall} \over {precision + recall}}$$ <br>

Pada proyek ini, performa *F-score* model pada tiap algoritma dihitung dengan memanfaatkan modul dari library *scikit-learn*.

Dari hasil pengamatan performa terhadap nilai *accuracy*, *precision*, *recall*, dan *F1-Score* pada kelima model tersebut, diperoleh informasi sebagai berikut:

|      Algoritma      | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) |
|:-------------------:|-------------:|--------------:|-----------:|-------------:|
| Logistic Regression |        76.39 |         73.17 |      60.30 |        66.12 |
| K-Nearest Neighbor  |        99.04 |         98.02 |      99.50 |        98.75 |
| SVM                 |        97.70 |         96.98 |      96.98 |        96.98 |
| Random Forest       |        99.04 |         98.02 |      99.50 |        98.75 |
| XGBoost             |        99.04 |         98.02 |      99.50 |        98.75 |

Kelima algoritma dibandingkan dengan menghitung nilai *accuracy*, *precision*, *recall*, *F1 score*. Terdapat tiga algoritma dengan hasil yang sama-sama terbaik dengan diatas 98% yaitu KNN, *Random Forest*, dan *XGBoost* baik pada Accuract maupun F1 score. Kemudian SVM dengan hasil diatas 96%. Terakhir, algoritma dengan performa hasil yang kurang memuaskan ialah *Logistic Regression* dengan hasil nilai Accuracy 76% dan nilai F1 score 66%.

## Kesimpulan
Dari hasil penilitan yang telah dilakukan, dapat disimpulkan bahwa 4 dari 5 algoritma yang dijadikan model untuk mendeteksi penyakit diabetes menghasilkan performa yang sangat baik. Algoritma yang dijadikan perbandingan ialah *Logistic Regression*, *K-Nearest Neighbor*, *Support Vector Machine*, *Random Forest*, dan *Extreme Gradient Boost*. Tiga algoritma yang memiliki tingkat performa teratas yaitu KNN, *Random Forest*, dan *XGBoost*. Ketiganya sama-sama menghasilkan performa terbaik yang sebagai acuan adalah nilai *accuracy*, *precision*, *recall*, dan *F1-score*. Disusul oleh algoritma SVM yang memiliki nilai performa yang sangat baik. Terakhir, algoritma Logistic Regression memperoleh nilai performa yang kurang baik.


## Referensi
American Diabetes Association. (2022). [Diagnosis and Classification of Diabetes Mellitus.](https://care.diabetesjournals.org/content/42/Supplement_1/S13)

Hasibuan, N.K., Dur, S. dan Husein, I. (2022). [Faktor Penyebab Penyakit Diabetes Melitus dengan Metode Regresi Logistik. _G-Tech: Jurnal Teknologi Terapan_, _6_(2), 257–264.](https://doi.org/10.33379/gtech.v6i2.1696)

Jian Y, Pasquier M, Sagahyroon A, Aloul F. (2021). ["A Machine Learning Approach to Predicting Diabetes Complications" _Healthcare_ 9, no. 12: 1712.](https://doi.org/10.3390/healthcare9121712)

Gretchen A. Youssef. (2019). [Health Care & Education Presidential Address: It’s All About Access!. __Diabetes Care__ 1 January 2021; 44 (1): 1–7](https://doi.org/10.2337/dci20-0050)

Butt, U. M., Letchmunan, S., Ali, M., Hassan, F. H., Baqir, A., & Sherazi, H. H. R. (2021). [Machine Learning Based Diabetes Classification and Prediction for Healthcare Applications. _Journal of healthcare engineering_, _2021_, 9930985.](https://doi.org/10.1155/2021/9930985)

Erlin, Yulvia Nora Marlim, Junadhi, Laili Suryati, & Nova Agustina. (2022). [Early Detection of Diabetes Using Machine Learning with Logistic Regression Algorithm. _Jurnal Nasional Teknik Elektro Dan Teknologi Informasi_, _11_(2), 88-96.](https://doi.org/10.22146/jnteti.v11i2.3586)

Jack Billie Chandra, & Dewi Nasien. (2023). [Penerapan Algoritma Klasifikasi K-Nearest Neighbor pada Penyakit Diabetes. _Prosiding Seminar Nasional Statistika Aktuaria_, _2_(1), 119–126.](https://doi.org/10.1234/snsa.v2i1.344)

Cahyani, A., & Basuki, A. (2019). [Klasifikasi Diabetes Mellitus Menggunakan Support Vector Machine (Studi Kasus: Puskesmas Modopuro, Mojokerto). _Rekayasa, 12_(2), 174-182.](https://doi.org/10.21107/rekayasa.v12i2.19763)

Karo Karo, I. M., & Hendriyana, H. (2022). [Klasifikasi Penderita Diabetes menggunakan Algoritma Machine Learning dan Z-Score. _Jurnal Teknologi Terpadu_, _8_(2), 94–99.](https://doi.org/10.54914/jtt.v8i2.564)

Kartina Diah Kusuma Wardani, & Memen Akbar. (2023). [Diabetes Risk Prediction using Feature Importance Extreme Gradient Boosting (XGBoost). _Jurnal RESTI (Rekayasa Sistem Dan Teknologi Informasi)_, _7_(4), 824 - 831.](https://doi.org/10.29207/resti.v7i4.4651)

Abdurrahman, G., Oktavianto, H., & Sintawati, M. (2022). [Optimasi Algoritma XGBoost Classifier Menggunakan Hyperparameter Gridesearch dan Random Search Pada Klasifikasi Penyakit Diabetes. _INFORMAL: Informatics Journal, 7_(3), 193-198.](https://doi.org/10.19184/isj.v7i3.35441)
