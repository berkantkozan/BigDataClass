import pandas as pd
import json
import time
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- AYARLAR VE VERİ YÜKLEME ---
PATH = r"C:\Users\User\.cache\kagglehub\datasets\rmisra\news-category-dataset\versions\3\News_Category_Dataset_v3.json"
LIMIT = 10000  # Ödev gereği 5000 nesne

print(f"{LIMIT} kayıt yükleniyor...")
data = []
with open(PATH, 'r') as f:
    for i, line in enumerate(f):
        if i >= LIMIT: break
        data.append(json.loads(line))

df = pd.DataFrame(data)
df['full_text'] = df['headline'] + " " + df['short_description']
texts = df['full_text'].tolist()

# --- BÖLÜM 2: COSINE BENZERLİĞİ (KESİN) ---
print("\n--- Bölüm 2: Cosine Benzerliği Hesaplanıyor ---")
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(texts)

start_cos = time.time()
cos_sim_matrix = cosine_similarity(tfidf_matrix)
indices = np.triu_indices(LIMIT, k=1)
similarities = cos_sim_matrix[indices]
top_10_idx = np.argsort(similarities)[-10:][::-1]

print("En Benzer 10 Çift (Cosine):")
for i in top_10_idx:
    u, v = indices[0][i], indices[1][i]
    print(f"Benzerlik: {similarities[i]:.4f} | Index {u} ve {v}")
print(f"Cosine İşlem Süresi: {time.time() - start_cos:.4f} saniye")

# --- BÖLÜM 3: MINHASH VE JACCARD ---
print("\n--- Bölüm 3: MinHash Tahmini Başlatılıyor ---")

def get_shingles(text, k=5):
    return set([text[i:i+k] for i in range(len(text) - k + 1)])

all_shingles = [get_shingles(t) for t in texts]

def generate_minhash_signatures(shingle_sets, num_hashes):
    flat_shingles = list(set().union(*shingle_sets))
    shingle_map = {s: i for i, s in enumerate(flat_shingles)}
    max_val = len(flat_shingles)
    
    a = random.sample(range(1, max_val), num_hashes)
    b = random.sample(range(1, max_val), num_hashes)
    c = 2**31 - 1 
    
    signatures = np.full((num_hashes, len(shingle_sets)), np.inf)
    for doc_idx, s_set in enumerate(shingle_sets):
        for shingle in s_set:
            shingle_id = shingle_map[shingle]
            for i in range(num_hashes):
                hash_val = (a[i] * shingle_id + b[i]) % c
                if hash_val < signatures[i, doc_idx]:
                    signatures[i, doc_idx] = hash_val
    return signatures

# MinHash Tahmini (100 Hash Fonksiyonu ile)
num_h = 100
start_mh = time.time()
sigs = generate_minhash_signatures(all_shingles, num_hashes=num_h)
mh_time = time.time() - start_mh

# Karşılaştırma için ilk 100 çifti örnekleyelim (Hız için)
print(f"MinHash ({num_h} hash) Süresi: {mh_time:.4f} saniye")

# Örnek bir doğruluk kontrolü
doc1, doc2 = indices[0][top_10_idx[0]], indices[1][top_10_idx[0]]
real_jaccard = len(all_shingles[doc1] & all_shingles[doc2]) / len(all_shingles[doc1] | all_shingles[doc2])
estimated_jaccard = np.mean(sigs[:, doc1] == sigs[:, doc2])

print(f"\nÖrnek Çift Analizi ({doc1} ve {doc2}):")
print(f"Gerçek Jaccard: {real_jaccard:.4f}")
print(f"MinHash Tahmini: {estimated_jaccard:.4f}")
print(f"Hata Payı: {abs(real_jaccard - estimated_jaccard):.4f}")