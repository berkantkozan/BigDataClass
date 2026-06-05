import json
import re
import zlib
import random
import time
import pickle
import os

def save_cache(signature_matrix, metadata, filename="minhash_cache.pkl"):
    """Matrisi ve metadata sözlüğünü binary olarak diske kaydeder."""
    print(f"\nVeriler '{filename}' dosyasına kaydediliyor...")
    with open(filename, 'wb') as f:
        # İki objeyi tek bir tuple olarak kaydediyoruz
        pickle.dump((signature_matrix, metadata), f)
    print("Kaydetme işlemi başarılı!")

def load_cache(filename="minhash_cache.pkl"):
    """Diskteki binary dosyadan matrisi ve metadata'yı okur."""
    if os.path.exists(filename):
        print(f"Önbellek '{filename}' dosyasından yükleniyor...")
        with open(filename, 'rb') as f:
            signature_matrix, metadata = pickle.load(f)
        print("Yükleme başarılı!")
        return signature_matrix, metadata
    else:
        print("Önbellek dosyası bulunamadı.")
        return None, None

# ==========================================
# 1. Shingling ve Ön İşleme Modülü
# ==========================================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_shingles(text, k=3, level='word'):
    text = preprocess_text(text)
    if level == 'word':
        tokens = text.split()
        if len(tokens) < k:
            return {" ".join(tokens)} if tokens else set()
        return {" ".join(tokens[i:i+k]) for i in range(len(tokens) - k + 1)}
    elif level == 'char':
        if len(text) < k:
            return {text} if text else set()
        return {text[i:i+k] for i in range(len(text) - k + 1)}

# ==========================================
# 2. MinHash Modülü
# ==========================================
class MinHashGenerator:
    def __init__(self, num_hashes=100, seed=42):
        self.num_hashes = num_hashes
        self.c = 4294967311 # Büyük bir asal sayı
        random.seed(seed)
        self.a_coeffs = [random.randint(1, self.c - 1) for _ in range(num_hashes)]
        self.b_coeffs = [random.randint(0, self.c - 1) for _ in range(num_hashes)]

    def _string_to_int(self, string_val):
        return zlib.crc32(string_val.encode('utf-8')) & 0xffffffff

    def generate_signature(self, shingles):
        signature = [float('inf')] * self.num_hashes
        for shingle in shingles:
            x = self._string_to_int(shingle)
            for i in range(self.num_hashes):
                hash_val = (self.a_coeffs[i] * x + self.b_coeffs[i]) % self.c
                if hash_val < signature[i]:
                    signature[i] = hash_val
        return signature

# ==========================================
# 3. Dosya Okuma ve Entegrasyon
# ==========================================
def build_signature_matrix(file_path, k=3, level='word', num_hashes=100):
    """
    JSON dosyasını okur, shingle'ları çıkarır ve tüm veri seti için imza matrisini oluşturur.
    """
    minhash_gen = MinHashGenerator(num_hashes=num_hashes)
    
    signature_matrix = [] # Her satır bir belgenin MinHash imzası olacak
    doc_metadata = {}     # ID -> URL eşleştirmesini tutacağımız sözlük
    
    start_time = time.time()
    
    print(f"'{file_path}' dosyası işleniyor...\n" + "-"*40)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for doc_id, line in enumerate(f):
                if not line.strip():
                    continue
                    
                record = json.loads(line)
                headline = record.get("headline", "")
                short_desc = record.get("short_description", "")
                
                # Metin birleştirme ve shingling
                merged_text = f"{headline} {short_desc}"
                shingles = get_shingles(merged_text, k=k, level=level)
                
                # Boş shingle setlerini atla (metin olmayan satırlar için)
                if not shingles:
                    signature_matrix.append([0] * num_hashes) # Boş vektör
                    doc_metadata[doc_id] = record.get("link", "Bilinmeyen Link")
                    continue
                
                # MinHash imzasını hesapla ve matrise ekle
                signature = minhash_gen.generate_signature(shingles)
                signature_matrix.append(signature)
                
                # Orijinal içeriğe geri dönmek için linki sakla
                doc_metadata[doc_id] = record.get("link", "Bilinmeyen Link")
                
                # Her 10.000 belgede bir ilerleme durumunu yazdır
                if (doc_id + 1) % 10000 == 0:
                    elapsed = time.time() - start_time
                    print(f"İşlenen belge sayısı: {doc_id + 1:,} - Geçen süre: {elapsed:.2f} saniye")
                    
    except FileNotFoundError:
        print(f"Hata: '{file_path}' dosyası bulunamadı. Dizini kontrol edin.")
        return None, None
        
    total_time = time.time() - start_time
    print("-" * 40)
    print(f"İşlem tamamlandı! Toplam {len(signature_matrix):,} belge işlendi.")
    print(f"Toplam süre: {total_time:.2f} saniye")
    
    return signature_matrix, doc_metadata

# ==========================================
# Çalıştırma
# ==========================================
if __name__ == "__main__":
    FILE_NAME = "News_Category_Dataset_v3.json"
    CACHE_FILE = "minhash_cache.pkl"
    K_GRAM = 3
    NUM_HASHES = 100

    # 1. Önce diski kontrol et, önbellek varsa oradan yükle
    sig_matrix, metadata = load_cache(CACHE_FILE)
    
    # 2. Önbellek yoksa (ilk çalışma) baştan hesapla ve kaydet
    if not sig_matrix:
        sig_matrix, metadata = build_signature_matrix(
            file_path=FILE_NAME, 
            k=K_GRAM, 
            level='word', 
            num_hashes=NUM_HASHES
        )
        if sig_matrix:
            save_cache(sig_matrix, metadata, CACHE_FILE)
            
    if sig_matrix:
        print(f"\nKullanıma Hazır Matris Boyutu: {len(sig_matrix)} x {len(sig_matrix[0])}")