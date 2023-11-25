#!/usr/bin/env python
# coding: utf-8

# # Sistem Rekomendasi : <span style="font-weight:normal">Rekomendasi Ponsel Seluler</span>
# ---

# Proyek Submission 2 - Machine Learning Terapan
# 
# Oleh : Marwan Hadid

# # Pendahuluan

# Dataset ini merupakan kumpulan data ponsel seluler (cell phone) yang paling populer di Amerika Serikat pada tahun 2022. Dataset ini terdiri dari tiga berkas csv yang nantinya akan dibentuk menjadi sistem rekomendasi.

# # Mengimport Library

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer

import tensorflow as tf


# # Data Understanding

# In[2]:


# Membaca dataset

data_file = 'cellphones data.csv'
rating_file = 'cellphones ratings.csv'
user_file = 'cellphones users.csv'

cellphone_df = pd.read_csv(data_file)
rating_df = pd.read_csv(rating_file)
user_df = pd.read_csv(user_file)


# In[3]:


cellphone_df.head(5)


# In[4]:


rating_df.head(5)


# In[5]:


user_df.head(5)


# In[6]:


print(f'Jumlah data ponsel selular :', cellphone_df.cellphone_id.nunique())
print(f'Jumlah data penilaian pengguna :', len(rating_df))
print(f'Jumlah data pengguna yang melakukan penilaian :', user_df.user_id.nunique())


# In[7]:


cellphone_df.info()


# Penjelasan Fitur:
# - **cellphone_id**: nomor indeks unik pada setiap ponsel.
# - **brand**: merek produsen setiap ponsel.
# - **model**: tipe spesifik dari ponsel.
# - **operating system**: sistem operasi pada ponsel.
# - **internal memory**: ukuran memori internal yang tersedia dalam skala *giga byte* (GB).
# - **RAM**: ukuran RAM pada ponsel dalam skala *giga byte* (GB).
# - **performance**: rating performa ponsel berdasarkan skor *AnTuTu*.
# - **main camera**: resolusi kamera utama (belakang) dalam skala megapiksel (MP).
# - **selfie camera**: resolusi kamera *selfie* (depan) dalam skala megapiksel (MP).
# - **battery size**: kapasitas baterai pada ponsel dalam miliamper perjam (mAh).
# - **screen size**: ukuran ponsel dalam ukuran inci (*inches*).
# - **weight**: berat ponsel dalam gram (g).
# - **price**: harga ponsel dalam mata uang dollar (USD).
# - **release date**: tanggal rilis ponsel
# 
# Jumlah fitur yang ada sebanyak 14, 8 diantaranya adalah numerik int64, 4 diantaranya adalah object, 2 sisanya adalah float64. Fitur **release date** merupakan kolom yang seharusnya bertipe waktu atau datetime.

# In[8]:


print(f'Jumlah duplikasi :' ,cellphone_df.duplicated().sum())
print('Jumlah missing value :' ,cellphone_df.isnull().sum().sum())


# In[9]:


for col in cellphone_df.columns:
    unique_values = cellphone_df[col].unique()
    print(f"Unique values in {col}: {unique_values}")


# Perlu perbaikan pada fitur model karena terdapat kesalahan penamaan pada beberapa model ponsel.

# In[10]:


print(rating_df.nunique())
print('----------------------------------------')
rating_df.info()


# Terdapat 3 kolom pada berkas rating yaitu **user_id** sebagai nomer unik pengguna, **cellphone_id** sebagai nomer unik ponsel yang diberi nilai, dan **rating** sebagai skala nilai ponsel. Semua kolom bertipe numerik. Kolom **rating** memiliki anomali dimana seharusnya terdapat 10 nilai dalam skala 1 hingga 10.

# In[11]:


print(rating_df['rating'].unique())


# In[12]:


rating_df[rating_df['rating']== 18]


# Karena hanya terdapat 1 baris, data ini akan dihapus nantinya.

# In[13]:


print(f'Jumlah duplikasi :' ,rating_df.duplicated().sum())
print('Jumlah missing value :' ,rating_df.isnull().sum().sum())


# In[14]:


print(user_df.nunique())
print('----------------------------------------')
user_df.info()


# Kolom **gender** dan **occupation** memiliki tipe data object sedangkan **age** dan **user_id** bertipe data numerik int64. Kolom **gender** seharusnya hanya memiliki dua nilai unik.

# In[15]:


gender_col = user_df['gender'].unique()
print(gender_col)


# In[16]:


user_df[user_df['gender']== '-Select Gender-']


# In[17]:


print(f'Jumlah duplikasi :' ,user_df.duplicated().sum())
print('Jumlah missing value :' ,user_df.isnull().sum().sum())


# Missing value dan invalid value pada gender akan dihapus pada data preprocessing.

# # Data Preprocessing

# In[18]:


df = pd.merge(rating_df, user_df, on='user_id', how='left')
df = pd.merge(df, cellphone_df, on='cellphone_id', how='left')
df = df.dropna()
df = df[df['gender'] != '-Select Gender-']
df = df[df['rating'] != 18]
df['occupation'] = df['occupation'].str.lower()
df['release_year'] = pd.to_datetime(df['release date'], format='%d/%m/%Y').dt.year
df = df.drop(['release date'], axis=1)


# In[19]:


df['model'] = df['model'].str.replace(r'[\(\)\xa0\d]+', '').str.strip()
df['model'] = df['model'].str.extract(r'([^\(]*)')[0].str.strip()


# In[20]:


print(df['model'].unique())


# In[21]:


df['gender'] = df['gender'].map({'Male' : 0, 'Female' : 1})
#df['operating system'] = df['operating system'].map({'Android' : 0, 'iOS' : 1})


# In[22]:


df.head(5)


# # Data Visualization

# In[23]:


# Jumlah Ponsel Setiap Brand Pada Dataset
brand_model = cellphone_df.drop_duplicates(subset=['brand', 'model'], keep='first')

model_count = brand_model.groupby('brand')['model'].count().reset_index()

plt.figure(figsize=(7, 4.5))
plt.bar(model_count['brand'], model_count['model'], color='blue')
plt.title('Jumlah Ponsel Tiap Brand Pada Dataset', fontsize=14)
plt.tight_layout()
plt.savefig('jumlah_ponsel_tiap_brand.png')
plt.show()


# In[24]:


avg_rating_brand = df.groupby('brand')['rating'].mean().reset_index()

avg_rating_brand = avg_rating_brand.sort_values(by='rating', ascending=True)

plt.figure(figsize=(8, 5))
plt.barh(avg_rating_brand['brand'], avg_rating_brand['rating'], color='magenta')
plt.title('Rata-rata Rating Ponsel Tiap Brand', fontsize=14)
plt.tight_layout()
plt.savefig('rata_rata_rating_tiap_brand.png')
plt.show()


# In[25]:


rating_counts = df['rating'].value_counts().sort_index()

plt.figure(figsize=(7,6))
plt.bar(rating_counts.index, rating_counts.values, color='cyan')
plt.title('Distribusi Rating', fontsize=14)
plt.xticks(rating_counts.index)
plt.tight_layout()
plt.savefig('distribusi_rating.png')
plt.show()


# In[26]:


def wordcl(df, column):
    col = str(column)
    model_text = ' '.join(df[col])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(model_text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


# In[27]:


stopwords = set(["Pro", "Max", "Plus", "Lite", "Ultra"])
model_text = ' '.join([word for word in df['model'] if word not in stopwords])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(model_text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Wordcloud Model Ponsel', fontsize=20)
plt.axis('off')
plt.savefig('wordcloud_model_ponsel.png')
plt.show()


# # Data Preparation

# In[28]:


df = df.sort_values('user_id', ascending=True)
df.head(5)


# In[29]:


# Ponsel teratas berdasarkan rating dan harga
top_cellphones = df.sort_values(by=['rating', 'price'], ascending=[False, True]).drop_duplicates(subset=['brand'], keep='first').reset_index()
print("Top Cellphones by Rating and Price:")
print(top_cellphones[['brand', 'model', 'rating', 'price']].head(10))


# ## Content Based Filtering

# In[30]:


df_prep = df.drop(['user_id','cellphone_id', 'occupation', 'gender', 'age'], axis=1)
df_prep.head(5)


# In[31]:


fitur = ['operating system', 'internal memory', 'RAM', 'main camera']

df_cbf = df_prep.copy()

def combine_features(row):
    return ' '.join([str(row[feature]) for feature in fitur])

df_cbf['combinedFeatures'] = df_cbf.apply(combine_features, axis=1)

print(df_cbf['combinedFeatures'].head())


# In[32]:


df_cbf.duplicated().sum()


# In[33]:


df_cbf.drop_duplicates(subset=['model'], inplace=True)


# In[34]:


cv = CountVectorizer()
cv_matrix = cv.fit_transform(df_cbf['combinedFeatures'])
print(cv_matrix.shape)

similarities_tfidf = cosine_similarity(cv_matrix)
print(similarities_tfidf.shape)


# In[35]:


cosine_sim_df = pd.DataFrame(similarities_tfidf, index=df_cbf['model'], columns=df_cbf['model'])
print('Shape:', cosine_sim_df.shape)

print(cosine_sim_df.sample(4, axis=1).sample(8, axis=0))


# Dataset untuk Content Based Filtering hanya memerlukan fitur-fitur yang berhubungan dengan spesifikasi ponsel.

# ## Collaborative Filtering

# ### Nearest Neighbors

# In[36]:


knn_df = df.copy()

# Menghitung jumlah rating pada setiap ponsel
number_of_ratings = knn_df.groupby('cellphone_id')['rating'].count().reset_index()
number_of_ratings.rename(columns={'rating': 'number of cellphone-rating'}, inplace=True)

# Menggabungkan data dengan number_of_ratings
model_knn_df = knn_df.merge(number_of_ratings, on='cellphone_id')

model_knn_df.head(5)


# In[37]:


model_knn_df_pivot = model_knn_df.pivot_table(columns='user_id', index='cellphone_id', values='rating')
model_knn_df_pivot.fillna(0, inplace=True)

# Menyesuaikan ukuran matriks
model_knn_df_pivot = model_knn_df_pivot.astype(int)

model_knn_df_pivot.head(5)


# Dataset diubah menjadi pivot table. Dataset untuk KNN hanya memerlukan fitur-fitur yang berhubungan dengan rating ponsel.

# ### Deep Learning

# In[38]:


dl_df = df.copy()

# Encode user_id dan cellphone_id
user_ids = dl_df["user_id"].unique().tolist()
user_to_user_encoded  = {x: i for i, x in enumerate(user_ids)}
user_encoded_to_user  = {i: x for i, x in enumerate(user_ids)}

cellphone_ids = dl_df["cellphone_id"].unique().tolist()
cellphone_to_cellphone_encoded  = {x: i for i, x in enumerate(cellphone_ids)}
cellphone_encoded_to_cellphone  = {i: x for i, x in enumerate(cellphone_ids)}

dl_df["user"] = dl_df["user_id"].map(user_to_user_encoded)
dl_df["cellphone"] = dl_df["cellphone_id"].map(cellphone_to_cellphone_encoded)

num_users = len(user_to_user_encoded)
num_cellphones = len(cellphone_encoded_to_cellphone)

dl_df["rating"] = dl_df["rating"].values.astype(np.float32)

min_rating = min(dl_df["rating"])
max_rating = max(dl_df["rating"])

print("Number of users: {}, Number of Books: {}, Min rating: {}, Max rating: {}".format(num_users, num_cellphones, min_rating, max_rating))


# In[39]:


model_dl_df = dl_df.sample(frac=1, random_state=42)
model_dl_df.head(5)


# In[40]:


x = model_dl_df[["user", "cellphone"]].values

# Target variable
y = model_dl_df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

# Split data menjadi train dan test
train_indices = int(0.7 * model_dl_df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:],
)

x_train.shape, x_val.shape, y_train.shape, y_val.shape


# Dataset untuk Deep Learning hanya memerlukan fitur-fitur yang berhubungan dengan rating ponsel. Dataset akan dibagi sebanyak 70% untuk data training dan 30% untuk data testing.

# # Model Development

# ## Content Based Filtering

# In[41]:


def phone_recommendations(model_name, similarity_data=cosine_sim_df, items=df_cbf[fitur + ['model']], k=5):
    index = similarity_data.loc[:, model_name].to_numpy().argpartition(range(-1, -k-1, -1))

    closest = similarity_data.columns.to_numpy()[index[-1:-(k+2):-1]]

    closest = np.ravel(closest)

    closest = [phone for phone in closest if phone != model_name]

    result_df = pd.DataFrame(closest, columns=['model']).merge(items, on='model').head(k)

    similarity_scores = similarity_data.loc[model_name, result_df['model']]
    result_df['score'] = similarity_scores.values

    return result_df[fitur + ['model', 'score']]

model_name = '10T'  # Sesuaikan dengan nama model yang ingin direkomendasikan
num_rec = 5  # Jumlah Rekomendasi
phone_recommendations_df = phone_recommendations(model_name, k=num_rec)

print(f"Top {num_rec} Rekomendasi ponsel berdasarkan model {model_name}:")
print(phone_recommendations_df)


# In[42]:


print(model_name)
print(df_cbf[df_cbf.model.eq(model_name)][fitur])


# Fitur untuk content based filtering dapat ditambah maupun dikurangi pada variabel fitur.

# Hasil dari Content Based Filtering adalah rekomendasi ponsel berdasarkan fitur-fitur yang dimiliki oleh ponsel yang dipilih.

# ## Collaborative Filtering

# ### Nearest Neighbors

# In[43]:


# Inisialisasi model KNN
model_knn_cellphone = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn_cellphone.fit(model_knn_df_pivot)


# In[44]:


cellphone_df_name = pd.DataFrame({'Cellphone': model_knn_df_pivot.index})

cellphone_df_name.duplicated().sum()


# In[45]:


# Fungsi untuk mendapatkan rekomendasi
def recommend_cellphone(cellphone_name):
    cellphone_id = np.where(model_knn_df_pivot.index == cellphone_name)[0][0]
    distances, recommendations = model_knn_cellphone.kneighbors(model_knn_df_pivot.iloc[cellphone_id, :].values.reshape(1, -1), n_neighbors=11)

    cellphone = []
    distance = []

    for i in range(0, len(distances.flatten())):
        if i != 0:
            recommended_cellphone_id = model_knn_df_pivot.index[recommendations.flatten()[i]]
            recommended_cellphone_name = model_knn_df[model_knn_df['cellphone_id'] == recommended_cellphone_id]['model'].values[0]
            cellphone.append(recommended_cellphone_name)
            distance.append(distances.flatten()[i])

    c = pd.Series(cellphone, name='cellphone')
    d = pd.Series(distance, name='distance')
    recommendations_df = pd.concat([c, d], axis=1)
    recommendations_df = recommendations_df.sort_values('distance', ascending=False)

    print(f'10 Rekomendasi untuk ponsel {model_knn_df[model_knn_df["cellphone_id"] == cellphone_name]["model"].values[0]} sebagai berikut :\n')
    for i in range(0, recommendations_df.shape[0]):
        print('{0}: {1}, with distance of {2}'.format(i, recommendations_df["cellphone"].iloc[i], recommendations_df["distance"].iloc[i]))


# In[46]:


cellphone_name = cellphone_df_name.iloc[28][0]
recommend_cellphone(cellphone_name)


# Collaborative Filtering menggunakan metode Nearest Neighbors untuk mendapatkan rekomendasi ponsel berdasarkan rating yang diberikan oleh pengguna.

# ### Deep Learning

# In[47]:


# Recommender system deep learning
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_cellphones, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_cellphones = num_cellphones
        self.embedding_size = embedding_size
        self.user_embedding = tf.keras.layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
        )
        self.user_bias = tf.keras.layers.Embedding(num_users, 1)
        self.cellphone_embedding = tf.keras.layers.Embedding(
            num_cellphones,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
        )
        self.cellphone_bias = tf.keras.layers.Embedding(num_cellphones, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        cellphone_vector = self.cellphone_embedding(inputs[:, 1])
        cellphone_bias = self.cellphone_bias(inputs[:, 1])
        dot_user_cellphone = tf.tensordot(user_vector, cellphone_vector, 2)
        # Add all the components (including bias)
        x = dot_user_cellphone + user_bias + cellphone_bias
        # The sigmoid activation forces the rating to between 0 and 1
        return tf.nn.sigmoid(x)


# In[48]:


model = RecommenderNet(num_users, num_cellphones, 50)

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)


# In[49]:


history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=8,
    epochs=100,
    verbose=1,
    validation_data=(x_val, y_val),
)


# In[50]:


# Pembuatan dataframe untuk ponsel
phone_df = pd.DataFrame({
    'cellphone_id': dl_df['cellphone_id'],
    'Model': dl_df['model'],
    'Brand': dl_df['brand'],
})

phone_df.duplicated().sum()


# In[51]:


phone_df.drop_duplicates(inplace=True)


# In[52]:


phone_df.head(5)


# In[53]:


# Top rekomendasi ponsel untuk random user
user_id = dl_df.user_id.sample(10).iloc[0]
cellphones_rated_by_user = dl_df[dl_df.user_id == user_id]
cellphones_not_rated = phone_df[~phone_df['cellphone_id'].isin(cellphones_rated_by_user.cellphone_id.values)]['cellphone_id']
cellphones_not_rated = list(
    set(cellphones_not_rated).intersection(set(dl_df['cellphone_id']))
)

cellphones_not_rated = [[cellphone_to_cellphone_encoded.get(x)] for x in cellphones_not_rated]

user_encoder = user_to_user_encoded.get(user_id)

user_cellphone_array = np.hstack(
    ([[user_encoder]] * len(cellphones_not_rated), cellphones_not_rated)
)


# In[54]:


ratings = model.predict(user_cellphone_array).flatten()


# In[55]:


top_ratings_indices = ratings.argsort()[-10:][::-1]

recommended_cellphone_ids = [
    cellphone_encoded_to_cellphone.get(cellphones_not_rated[x][0]) for x in top_ratings_indices
]

print("Showing recommendations for user: {}".format(user_id))
print("====" * 9)
print("Cellphones with high ratings from user")
print("----" * 8)

top_cellphones_user = (
    cellphones_rated_by_user.sort_values(
        by='rating',
        ascending=False)
    .head(5)
    .cellphone_id.values
)

cellphone_df_rows_user = phone_df[phone_df["cellphone_id"].isin(top_cellphones_user)]
for row in cellphone_df_rows_user.itertuples():
    print(row.Model, ":", row.Brand)

print("----" * 8)
print("Top 10 Cellphones recommendations")
print("----" * 8)

recommended_cellphones = phone_df[phone_df["cellphone_id"].isin(recommended_cellphone_ids)]
for row in recommended_cellphones.itertuples():
    print(row.Model, ":", row.Brand)


# # Model Evaluation

# ## Collaborative Filtering Deep Learning

# In[56]:


plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('Model Metrics', fontsize=15)
plt.ylabel('Root Mean Squared Error')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('figure.png')
plt.show()


# In[57]:


model.evaluate(x,y)


# Pada pembuatan proyek ini metrik evaluasi yang digunakan yaitu root mean squared error (RMSE). Dari proses ini, kita memperoleh nilai error akhir sebesar sekitar 0.2040 dan error pada data validasi sebesar 0.2944. Nilai tersebut cukup bagus untuk sistem rekomendasi.
# 
# Setelah dilakukan evaluasi menggunakan seluruh data memperoleh nilai error sebesar 0.3281.
