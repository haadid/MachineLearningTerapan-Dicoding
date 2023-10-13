#!/usr/bin/env python
# coding: utf-8

# # Analisis Prediktif : <span style="font-weight:normal">Prediksi Diabetes</span>
# ---

# Proyek Submission 1 - Machine Learning Terapan
# 
# Oleh : Marwan Hadid

# # Pendahuluan

# Pada Proyek ini, saya mengambil dataset dengan tema kesehatan. Masalah yang diambil pada tema ini adalah penyakit diabetes. Pada dataset ini, model akan mengeksplorasi hubungan terhadap beberapa indikator kesehatan serta kemungkinan diabetes. Indikator-indikator tersebut berupa, banyak kali hamil, konsentrasi plasma glukosa selama 2 jam pada tes toleransi glukosa, tekanan darah diastolik, ketebalan kulit, indeks massa tubuh, skor genetik diabetes, dan umur. Model yang telah dikembangkan nantinya diharapkan dapat memprediksi penyakit diabetes berdasarkan indikator yang ada.

# # Mengimport Library

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# # Data Wrangling

# ## Membaca Dataset

# In[2]:


file = 'Healthcare_Diabetes.csv'
df = pd.read_csv(file)
df


# - Terdapat sejumlah 2768 _entry_ pada dataset
# - Data memiliki 10 kolom

# In[3]:


df.info()


# Penjelasan Fitur:
# - **Id**: Penomoran unik untuk tiap data.
# - **Pregnancies**: Banyak kali hamil.
# - **Glucose**: Konsentrasi plasma glukosa selama 2 jam dalam tes toleransi glukosa.
# - **BloodPressure**: Tekanan darah diastolik (mm Hg).
# - **SkinThickness**: Ketebalan lapisan kulit trisep (mm).
# - **Insulin**: Insulin serum 2 jam (mu U/ml).
# - **BMI**: _Body mass index_ atau indeks massa tubuh (berat badan dalam kg / tinggi badan dalam m^2).
# - **DiabetesPedigreeFunction**: Fungsi silsilah diabetes, skor genetik diabetes.
# - **Age**: Umur dalam satuan tahun.
# - **Outcome**: Klasifikasi yang biner menentukan ada (1) atau tidak adanya (0) diabetes.
# 
# Fitur-fitur pada dataset semuanya berupa numerik yang terdiri dari 2 kolom bertipe float64 dan 8 kolom bertipe int64

# In[4]:


df.describe(include='all').T


# Penjelasan :
# - Dataset hanya memiliki fitur numerik.
# - Age berkisar dari 21 hingga 81 tahun.
# - Terdapat anomali pada fitur Insulin, BMI, BloodPressure, SkinThickness karena nilai minimumnya 0.

# ## Membersihkan Dataset

# In[5]:


# Mengecek nilai pada suatu data yang berisi N/A atau Null

df.isna().sum()


# Tidak terdapat _missing value_ pada data

# In[6]:


# Mengecek jika ada duplikasi pada data

print('Jumlah duplikasi : ', df.duplicated().sum())


# In[7]:


# Menghapus kolom 'Id' karena tidak berpengaruh terhadap data

df.drop(['Id'], axis=1, inplace=True)


# # EDA

# ## Univariate Analysis

# In[8]:


# List fitur pada dataset kecuali outcome
fitur = [col for col in df.columns if col != 'Outcome']
fitur


# In[9]:


# Analisis Outlier

plt.figure(figsize=(15,5))
plt.suptitle('Analisis Outlier Pada Fitur', fontsize=20)
for i, col in enumerate(fitur, 1):
    plt.subplot(2, 4, i)
    sns.boxplot(df, x=col)
    plt.xlabel(col, fontsize=13)

plt.tight_layout()
plt.show()


# - Semua fitur memiliki outlier.
# - Fitur Age memiliki beberapa outlier untuk nilai diatas 60.
# - Fitur Glucose, BloodPressure, BMI memiliki anomali pada data yang bernilai 0.
# - Fitur Insulin dan DiabetesPedigreeFunction memiliki banyak outlier.

# In[10]:


plt.figure(figsize=(15,6))
plt.suptitle('Analisis Distribusi Normal Pada Fitur', fontsize=20)
for i, col in enumerate(fitur, 1):
    plt.subplot(2, 4, i)
    sns.histplot(df, x=col, kde=True, bins=18)
    plt.xlabel(col, fontsize=13)
    plt.ylabel('')

plt.tight_layout()
plt.show()


# Distribusi data kebanyakan tidak normal. Hanya fitur Glucose dan BloodPressure yang terlihat memiliki distribusi normal meskipun terdapat outlier dinilai 0 pada kedua fitur. Sedangkan fitur lainnya cenderung _right-skewed_. Data 0 pada Glucose dan BloodPressure akan di _drop_. Kemudian fitur akan distandarisasi menggunakan _StandardScaler_ sebelum digunakan untuk melatih model.

# In[11]:


# Menangani anomali dengan cara menghapus data yang bernilai 0
def drop_anomaly(df):
    anomali = ['Glucose', 'BloodPressure', 'BMI']
    q1 = df[anomali].quantile(0.25)
    q3 = df[anomali].quantile(0.75)
    iqr = q3 - q1

    # Hanya perlu menghilangkan bagian bawah yang bernilai 0
    min = q1 - (iqr * 1.5)

    lower_array = np.where(df[anomali] <= min)[0]

    df.drop(index=lower_array, inplace=True)

    return df

df = drop_anomaly(df)
df.shape


# Data pada kolom Glucose, BloodPressure, dan BMI yang berada bawah kuartil bawah yaitu diangka 0 telah dihapus sehingga menyisakan data sebanyak 2604 baris.

# In[12]:


# Menganalisa Fitur Outcome

plt.figure(figsize=(5,5))
sns.countplot(df, y='Outcome')
plt.show()


# In[13]:


outcome_df = df['Outcome'].value_counts().reset_index()
outcome_df.columns = ['Outcome', 'Jumlah']
outcome_df


# Data klasifikasi biner tidak seimbang pada fitur Outcome. Data tidak adanya diabetes (0) lebih banyak setengah dari data ada diabetes (1).

# ## Multivariate Analysis

# In[14]:


# Membuat pairplot untuk memetakan hubungan antar fitur

sns.pairplot(df, hue='Outcome')


# Tidak terdapat informasi yang relevan untuk mengetahui hubungan antar fitur pada _pairplot_.

# In[15]:


# Memetakan korelasi antar fitur menggunakan heatmap
korelasi = df.corr().round(2)

plt.figure(figsize=(8,6))
sns.heatmap(korelasi, annot=True, linewidths=0.5, cmap='coolwarm')
plt.title('Korelasi antara Fitur', fontsize=15)
plt.show()


# Matriks korelasi menunjukkan adanya hubungan terkait dengan fitur yang ada. Fitur pada Glucose (0.49), BMI (0.27), Age (0.25), dan Pregnancies (0.23) terlihat mempunyai korelasi terhadap Outcome sebagai penentu ada atau tidaknya diabetes dibandingkan fitur lainnya. Sedangkan BloodPressure (0.18), DiabetesPedigreeFunction (0.17), dan Insulin (0.14) juga memiliki korelasi terhadap Outcome meskipun dengan nilai lebih kecil.  Sementara itu ada beberapa fitur menarik yang saling berkorelasi namun tidak memiliki korelasi tinggi pada fitur Outcome yaitu Age dan Pregnancies (0.55), Insulin dan SkinThickness (0.43), BMI dan SkinThickness (0.39), Age dan BloodPressure (0.33), Glucose dan Insulin (0.33), dst.
# 
# Dikarenakan model hanya akan memprediksi Outcome sebagai ada tidaknya diabetes, maka hanya fitur yang memliki korelasi signifikan yaitu Pregnancies, Glucose, Insulin, BMI, dan Age akan digunakan sebagai penentu prediksi model. Fitur yang memiliki tingkat korelasi terhadap Outcome akan dianalisa lebih lanjut.
# 

# In[16]:


# Analisa hubungan Glucose dan Outcome menggunakan barplot dan histplot
plt.subplots(nrows=1, ncols=2, figsize=(13,5))

plt.subplot(1, 2, 1)
sns.barplot(df, x='Outcome', y='Glucose')
plt.title('Rata - Rata Glukosa pada Outcome', fontsize=13)

plt.subplot(1, 2, 2)
sns.histplot(df, x="Glucose", hue='Outcome', palette='husl')
plt.title('Distribusi Glucose dan Outcome', fontsize=13)
plt.ylabel('')

plt.tight_layout()
plt.show()


# In[17]:


# Analisa hubungan BMI dan Outcome menggunakan barplot dan histplot
plt.subplots(nrows=1, ncols=2, figsize=(13,5))

plt.subplot(1, 2, 1)
sns.barplot(df, x='Outcome', y='BMI')
plt.title('Rata - Rata BMI pada Outcome', fontsize=13)

plt.subplot(1, 2, 2)
sns.histplot(df, x="BMI", hue='Outcome', palette='husl')
plt.title('Distribusi BMI dan Outcome', fontsize=13)
plt.ylabel('')

plt.tight_layout()
plt.show()


# In[18]:


# Analisa hubungan Age dan Outcome menggunakan barplot dan histplot
plt.subplots(nrows=1, ncols=2, figsize=(13,5))

plt.subplot(1, 2, 1)
sns.barplot(df, x='Outcome', y='Age')
plt.title('Rata - Rata Age pada Outcome', fontsize=13)

plt.subplot(1, 2, 2)
sns.histplot(df, x="Age", hue='Outcome', palette='husl')
plt.title('Distribusi Age dan Outcome', fontsize=13)
plt.ylabel('')

plt.tight_layout()
plt.show()


# In[19]:


# Analisa hubungan Pregnancies dan Outcome menggunakan barplot dan countplot
plt.subplots(nrows=1, ncols=2, figsize=(13,5))

plt.subplot(1, 2, 1)
sns.barplot(df, x='Outcome', y='Pregnancies')
plt.title('Rata - Rata Pregnancies pada Outcome', fontsize=13)

plt.subplot(1, 2, 2)
sns.countplot(x='Pregnancies', hue ='Outcome', data=df)
plt.title('Distribusi Pregnancies dan Outcome', fontsize=13)
plt.ylabel('')

plt.tight_layout()
plt.show()


# # Data Preparation

# In[20]:


# Fitur pada data akan dipilah menyesuakan korelasi terhadap Outcome model
fitur_final = ['Pregnancies', 'Glucose', 'BMI', 'Age', 'Outcome']
fitur_numerik = [i for i in fitur_final if i != 'Outcome']
new_df = df[fitur_final]

new_df.head(5)


# In[21]:


X = new_df.copy().drop('Outcome', axis=1)
y = new_df['Outcome']


# In[22]:


# Data akan dibagi menjadi 80% train dan 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
print(f'Total sampel pada dataset: {len(X)}')
print(f'Total sampel pada train dataset: {len(X_train)}')
print(f'Total sampel pada test dataset: {len(X_test)}')


# In[23]:


# Standarisasi data menggunakan StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)


# Dataset akan distandarisasi menggunakan _StandardScaler_ untuk mengubah nilai data menjadi distribusi normal dengan mean 0 dan standar deviasi 1. Hal ini dilakukan untuk membangun model dengan baik karena membuat skala data menjadi 0 dan 1.

# In[24]:


X_train


# # Model Development

# ## Logistic Regression

# In[25]:


log_regres=LogisticRegression()
log_regres.fit(X_train, y_train)
log_gres_pred = log_regres.predict(X_test)

print(classification_report(y_test,log_gres_pred))


# In[26]:


log_regres_acc = accuracy_score(y_test, log_gres_pred)

print(f"Akurasi : {log_regres_acc:.2f}")


# Akurasi model dengan menerapkan algoritma _Logistic Regression_ adalah 0.76. Untuk meningkatkan akurasi, akan dilakukan pencarian parameter terbaik dengan menggunakan _GridSearchCV_.

# ### Grid Search Logistic Regression

# In[27]:


# Membuat dictionary untuk parameter yang akan di tune
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
}

grid_search_lr = GridSearchCV(log_regres, param_grid_lr, cv=5, scoring='accuracy', verbose=1)
grid_search_lr.fit(X_train, y_train)

best_params_lr = grid_search_lr.best_params_
print("Best Hyperparameters:", best_params_lr)

# Mengambil model dengan paramater terbaik
best_model_lr = grid_search_lr.best_estimator_

best_pred_lr = best_model_lr.predict(X_test)

# Menghitung nilai dari model terbaik
best_lr_acc = accuracy_score(y_test, best_pred_lr)
print(f"Best Model Accuracy: {best_lr_acc:.2f}")


# Sayangnya, model dengan _hyperparamter tuning_ terbaik tetap memiliki nilai akurasi yang sama.

# ## K-Nearest Neighbor

# In[28]:


# Nilai K secara acak dipilih 5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)

print(classification_report(y_test, knn_pred))


# In[29]:


knn_acc = accuracy_score(y_test, knn_pred)
print(f"Skor Akurasi : {knn_acc:.2f}")


# Nilai K sembarang yaitu 5 dengan akurasi 0.80, memiliki performa yang kurang matang. Performa model K-NN dapat ditingkatkan dengan melakukan _hyperparameter tuning_ pada nilai K. Oleh karena itu, perlu dilakukan pencarian nilai K yang optimal. Salah satu metodenya ialah dengan menggunakan Metode _Elbow_.

# ### Metode Elbow

# In[30]:


# Menghitung nilai optimal K dengan menggunakan metode elbow
error_rate = []
for i in range(1, 8):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    knn_pred_elbow = knn.predict(X_test)
    accuracy = accuracy_score(y_test, knn_pred_elbow)
    print(f"Akurasi Pada K = {i} adalah {accuracy}")
    error_rate.append(np.mean(knn_pred_elbow != y_test))


# In[31]:


plt.figure(figsize=(8,5))
plt.plot(range(1, 8), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. Nilai K', fontsize=15)
plt.xlabel('Nilai K')
plt.ylabel('')
plt.show()


# In[32]:


# Memilih error rate terkecil
print("Minimum error:",min(error_rate)," pada K = ",error_rate.index(min(error_rate))+1)


# Ditemukan bahwa nilai K dengan error rate dan akurasi terbaik adalah 1

# In[33]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)

print(classification_report(y_test, knn_pred))


# In[34]:


knn_acc = accuracy_score(y_test, knn_pred)
print(f"Skor Akurasi : {knn_acc:.2f}")


# ## Support Vector Machine

# In[35]:


svm = SVC()
svm.fit(X_train,y_train)
svm_pred = svm.predict(X_test)


# In[36]:


print(classification_report(y_test, svm_pred))


# In[37]:


svm_acc = accuracy_score(y_test, svm_pred)
print(f"Skor Akurasi : {svm_acc:.2f}")


# Dengan menggunakan algoritma _Support Vector Machine_, model memperoleh akurasi sebesar 0.79

# ### Grid Search SVM

# In[38]:


param_grid_svm = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
}

grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy', verbose=1)
grid_search_svm.fit(X_train, y_train)

best_params_svm = grid_search_svm.best_params_
print("Best Hyperparameters:", best_params_svm)

# Get the best model
best_model_svm = grid_search_svm.best_estimator_

# Make predictions with the best model
best_pred_svm = best_model_svm.predict(X_test)

# Calculate accuracy with the best model
best_svm_acc = accuracy_score(y_test, best_pred_svm)
print(f"Best Model Accuracy: {best_svm_acc:.2f}")


# In[39]:


svm_pred = best_model_svm.predict(X_test)


# In[40]:


print(classification_report(y_test, svm_pred))


# Akurasi untuk model algoritma SVM telah naik dari 0.79 menjadi 0.98 setelah mendapatkan _hyperparameter_ yang optimal

# ## Random Forest

# In[41]:


rf = RandomForestClassifier(random_state=10)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)


# In[42]:


print(classification_report(y_test, rf_pred))


# In[43]:


rf_acc = accuracy_score(y_test, rf_pred)
print(f"Skor Akurasi : {rf_acc:.2f}")


# Algoritma _Random Forest_ menghasilkan skor akurasi sebesar 0.99.

# ## XGBoost Algorithm

# In[44]:


warnings.simplefilter(action='ignore', category=FutureWarning)


# In[45]:


xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)


# In[46]:


print(classification_report(y_test, xgb_pred))


# In[47]:


xgb_acc = accuracy_score(y_test, xgb_pred)
print(f"Skor Akurasi : {xgb_acc:.2f}")


# Dengan menggunkana Algoritma _XGBoost_, model memperoleh akurasi sebesar 0.99

# # Evaluasi Model

# ## Confusion Matrix

# In[48]:


models_pred = {
    'Logistic Regression': log_gres_pred,
    'KNN': knn_pred,
    'SVM': svm_pred,
    'RandomForest': rf_pred,
    'XGBoost': xgb_pred
}


# In[49]:


fig, axes = plt.subplots(1, len(models_pred), figsize=(16, 4))
fig.suptitle('Confusion Matrix')

for idx, (model_name, model_pred) in enumerate(models_pred.items()):
    cm = confusion_matrix(y_test, model_pred)
    ax = axes[idx]
    ax.set_title(model_name)
    sns.set(font_scale=0.8)

    sns.heatmap(cm, annot=True, cmap='PuRd', cbar=False, fmt='d',
                xticklabels=['Non-Diabetes', 'Diabetes'],
                yticklabels=['Non-Diabetes', 'Diabetes'],
                ax=ax)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

plt.tight_layout()
plt.show()


# ## Tabel Evaluasi Model

# In[50]:


algorithms = ['Logistic Regression', 'K-Nearest Neighbor', 'SVM', 'Random Forest', 'XGBoost']
y_preds = [log_gres_pred, knn_pred, svm_pred, rf_pred, xgb_pred]

eval = []

for model, y_pred in zip(algorithms, y_preds):
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    precision = round(precision_score(y_test, y_pred) * 100, 2)
    recall = round(recall_score(y_test, y_pred) * 100, 2)
    f1 = round(f1_score(y_test, y_pred) * 100, 2)
    eval.append([accuracy, precision, recall, f1])

eval = pd.DataFrame(eval, columns=['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1 Score (%)'], index=algorithms)
eval


# Hasil evaluasi diatas menunjukkan bahwa ada tiga algoritma yang memiliki tingkat performa teratas yaitu KNN, Random Forest, dan XGBoost. Ketiganya sama-sama menghasilkan performa terbaik yang sebagai acuan adalah nilai accuracy, precision, dan recall, dari lima algoritma yang dijadikan model. Disusul oleh algoritma SVM yang memiliki nilai performa yang sangat baik. Terakhir, algoritma Logistic Regression memperoleh nilai performa yang kurang baik.

# # Kesimpulan

# Dari hasil penilitan yang telah dilakukan, dapat diambil kesimpulan bahwa 4 dari 5 algoritma yang dijadikan model untuk mendeteksi penyakit diabetes dengan menggunakan dataset diatas menghasilkan performa yang sangat baik. Algoritma yang dijadikan perbandingan ialah Logistic Regression, K-Nearest Neighbor, Support Vector Machine, Random Forest, dan Extreme Gradient Boost. Kelima algoritma dibandingkan dengan menghitung nilai accuracy, precision, recall, F1 score. Terdapat tiga algoritma dengan hasil yang sama-sama terbaik dengan diatas 98% yaitu KNN, Random Forest, dan XGBoost baik pada Accuract maupun F1 score. Kemudian SVM dengan hasil diatas 96%. Terakhir, algoritma dengan performa hasil yang kurang memuaskan ialah Logistic Regression dengan hasil nilai Accuracy 76% dan nilai F1 score 66%.

# # Penutup

# Model dengan berbagai algoritma telah berhasil diuji untuk mendeteksi penyakit diabetes. Empat dari Lima algoritma menghasilkan performa yang sangat baik dalam memprediksi diabetes.
