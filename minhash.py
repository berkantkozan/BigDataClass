import pandas as pd
import numpy as np
import time
import binascii
import random

# 1. Veri Kümesini Yükleme (Performans testi için 1000 belge alıyoruz)
file_path = r"C:\Users\User\.cache\kagglehub\datasets\rmisra\news-category-dataset\versions\3\News_Category_Dataset_v3.json"
df = pd.read_json(file_path, lines=True, nrows=10000)
df['combined_text'] = df['headline'] + " " + df['short_description']
documents = df['combined_text'].tolist()

# 2. Shingle (K-gram) Çıkarımı
# Metinleri karakter bazlı 3-gram'lara (shingle) bölüp 32-bit integer hash'lere dönüştürüyoruz
def get_shingles(text, k=3):
    shingles = set()
    # Metni kelimelere bölmek yerine karakter bazlı k-gram daha ayırt edicidir
    text = text.lower().replace(" ", "") 
    for i in range(len(text) - k + 1):
        shingle = text[i:i+k]
        # Shingle'ı 32-bit integer'a dönüştür (hafıza optimizasyonu)
        crc = binascii.crc32(shingle.encode('utf-8')) & 0xffffffff
        shingles.add(crc)
    return shingles

print("Belgeler shingle'lara dönüştürülüyor...")
shingle_sets = [get_shingles(doc, k=5) for doc in documents]

# 3. Kesin Jaccard Benzerliği Hesaplama
def exact_jaccard(set1, set2):
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

# 4. MinHash İmzalarını Oluşturma Sınıfı
class MinHash:
    def __init__(self, num_hashes):
        self.num_hashes = num_hashes
        self.max_shingle_id = 2**32 - 1
        # Asal sayı (c), hash çakışmalarını önlemek için büyük seçilir
        self.next_prime = 4294967311 
        
        # h(x) = (a*x + b) % c formülü için rastgele 'a' ve 'b' katsayıları
        self.coeff_a = random.sample(range(1, self.max_shingle_id), num_hashes)
        self.coeff_b = random.sample(range(0, self.max_shingle_id), num_hashes)
        
    def generate_signature(self, shingle_set):
        signature = []
        for i in range(self.num_hashes):
            min_hash_val = float('inf')
            for shingle in shingle_set:
                # Evrensel hash fonksiyonu: h(x) = (a*x + b) % c
                hash_val = (self.coeff_a[i] * shingle + self.coeff_b[i]) % self.next_prime
                if hash_val < min_hash_val:
                    min_hash_val = hash_val
            signature.append(min_hash_val)
        return signature

def minhash_similarity(sig1, sig2):
    # İki imza arasındaki aynı olan elemanların oranı Jaccard benzerliğini tahmin eder
    return np.mean(np.array(sig1) == np.array(sig2))

# 5. Deney ve Karşılaştırma Ortamı
# Farklı hash fonksiyonu sayıları ile analizi gerçekleştireceğiz
hash_counts = [20, 50, 100, 200]
num_pairs_to_test = 10000 # Test edilecek rastgele belge çifti sayısı

# Karşılaştırma için rastgele çiftler seçiyoruz
random.seed(42)
test_pairs = [(random.randint(0, len(documents)-1), random.randint(0, len(documents)-1)) 
              for _ in range(num_pairs_to_test)]

print("\n--- ANALİZ BAŞLIYOR ---\n")

# A. Kesin Jaccard Testi
start_time = time.time()
exact_sims = [exact_jaccard(shingle_sets[i], shingle_sets[j]) for i, j in test_pairs]
exact_time = time.time() - start_time
print(f"Kesin Jaccard Çalışma Süresi ({num_pairs_to_test} çift): {exact_time:.4f} saniye")

# B. Farklı Hash Sayıları ile MinHash Testi
for num_hashes in hash_counts:
    minhash_model = MinHash(num_hashes)
    
    # İmzaları oluşturma
    sig_start = time.time()
    signatures = [minhash_model.generate_signature(s) for s in shingle_sets]
    sig_time = time.time() - sig_start
    
    # Benzerlikleri hesaplama (Tahmin)
    comp_start = time.time()
    approx_sims = [minhash_similarity(signatures[i], signatures[j]) for i, j in test_pairs]
    comp_time = time.time() - comp_start
    
    # Doğruluk Analizi (Mean Absolute Error - Ortalama Mutlak Hata)
    mae = np.mean(np.abs(np.array(exact_sims) - np.array(approx_sims)))
    
    print(f"\nHash Fonksiyonu Sayısı: {num_hashes}")
    print(f"  -> İmza Oluşturma Süresi: {sig_time:.4f} saniye")
    print(f"  -> MinHash Karşılaştırma Süresi: {comp_time:.4f} saniye")
    print(f"  -> Toplam MinHash Süresi: {sig_time + comp_time:.4f} saniye")
    print(f"  -> Ortalama Mutlak Hata (MAE): {mae:.4f}")