import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# 1. Veri Kümesini Yükleme ve Hazırlama
file_path = r"C:\Users\User\.cache\kagglehub\datasets\rmisra\news-category-dataset\versions\3\News_Category_Dataset_v3.json"

print("Veri yükleniyor...")
df = pd.read_json(file_path, lines=True, nrows=10000)

# Haber başlığı ve açıklamasını birleştiriyoruz
df['combined_text'] = df['headline'] + " " + df['short_description']

print(f"Toplam {len(df)} nesne başarıyla yüklendi.\n")

# 2. Vektör Temsiline Dönüştürme
print("Metinler TF-IDF vektörlerine dönüştürülüyor...")
start_time = time.time()

vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

print(f"Vektörleştirme tamamlandı. Geçen süre: {time.time() - start_time:.2f} saniye\n")

# 3. Kosinüs Benzerliği Hesaplama
# Not: Her bir döngü adımında manuel kosinüs hesabı yapmak Python'da günlerce sürebilir.
# Bu nedenle vektör çarpımlarını (benzerlik matrisini) toplu hesaplayıp,
# "Brute Force" mantığını ARAMA UZAYINI (12.5 milyon çift) tararken uygulayacağız.
print("Benzerlik matrisi hesaplanıyor...")
cosine_sim = cosine_similarity(tfidf_matrix)

# 4. Brute Force ile En Çok Benzeyen 10 Çifti Bulma
print("Brute Force algoritması ile tüm çiftler karşılaştırılıyor...")
search_start_time = time.time()

top_10_pairs = []  # (skor, id1, id2) formatında tutacağız
num_items = len(df)

# Brute Force: İç içe döngülerle tüm (i, j) kombinasyonlarını tarama
# i < j koşulu, (A, B) ve (B, A) tekrarlarını ve (A, A) kendisiyle karşılaştırmayı önler
for i in range(num_items):
    for j in range(i + 1, num_items):
        score = cosine_sim[i][j]
        
        # Eğer listemiz henüz 10 elemana ulaşmadıysa direkt ekle ve sırala
        if len(top_10_pairs) < 10:
            top_10_pairs.append((score, i, j))
            # Skora göre büyükten küçüğe sırala
            top_10_pairs.sort(key=lambda x: x[0], reverse=True)
        else:
            # Eğer mevcut skor, listemizdeki en düşük skordan (sonuncu elemandan) büyükse
            if score > top_10_pairs[-1][0]:
                top_10_pairs[-1] = (score, i, j)  # Son elemanı güncelle
                top_10_pairs.sort(key=lambda x: x[0], reverse=True)  # Yeniden sırala

print(f"Brute Force arama tamamlandı. Geçen süre: {time.time() - search_start_time:.2f} saniye\n")

# 5. Sonuçları Yazdırma
print("="*60)
print("EN ÇOK BENZEYEN 10 NESNE ÇİFTİ (BRUTE FORCE)")
print("="*60)

for rank, (score, i, j) in enumerate(top_10_pairs, 1):
    print(f"\n{rank}. Çift | Benzerlik Skoru: {score:.4f}")
    print(f"  [ID: {i} - Kategori: {df['category'].iloc[i]}]")
    print(f"  Başlık 1: {df['headline'].iloc[i]}")
    print(f"  Açıklama 1: {df['short_description'].iloc[i]}\n")
    
    print(f"  [ID: {j} - Kategori: {df['category'].iloc[j]}]")
    print(f"  Başlık 2: {df['headline'].iloc[j]}")
    print(f"  Açıklama 2: {df['short_description'].iloc[j]}")
    print("-" * 60)