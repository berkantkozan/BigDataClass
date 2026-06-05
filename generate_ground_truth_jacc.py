import json
import time
import re
from itertools import combinations, islice
import pickle
import multiprocessing as mp
from tqdm import tqdm  # İlerleme çubuğu kütüphanesi

# --- Ön İşleme ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def get_shingles(text, k=3):
    text = preprocess_text(text)
    tokens = text.split()
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i+k]) for i in range(len(tokens) - k + 1)}

# --- Bellek Optimizasyonu (Çekirdekler Arası RAM Paylaşımı) ---
# 2-4 GB'lık shingle sözlüğünü her çekirdeğe kopyalamak RAM'i tüketir.
# Bu yüzden sözlüğü global bir değişken olarak paylaşıyoruz.
global_shingles = {}

def init_worker(shared_dict):
    """Her işçi (worker) başlatıldığında sözlüğü sadece okuma amaçlı RAM'ine alır."""
    global global_shingles
    global_shingles = shared_dict

def process_chunk(chunk_pairs):
    """Her çekirdeğin çalıştıracağı alt iş paketi."""
    THRESHOLD = 0.80
    found = set()
    
    for doc1, doc2 in chunk_pairs:
        s1 = global_shingles.get(doc1)
        s2 = global_shingles.get(doc2)
        
        if not s1 or not s2: 
            continue
            
        intersection = len(s1.intersection(s2))
        union = len(s1.union(s2))
        score = intersection / union if union > 0 else 0.0
        
        if score >= THRESHOLD:
            found.add((doc1, doc2))
            
    return found

def chunked_iterable(iterable, size):
    """22 milyar veriyi RAM'de tutmak yerine, işlemcilere parça parça yediren jeneratör."""
    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, size))
        if not chunk:
            break
        yield chunk

# --- Ana Akış ---
if __name__ == "__main__":
    DATASET_FILE = "News_Category_Dataset_v3.json"
    GROUND_TRUTH_FILE = "ground_truth_pairs.pkl"
    NUM_CORES = 20 # Bilgisayarınızın çekirdek sayısına göre ayarlayabilirsiniz
    
    print("1. Shingle'lar RAM'e yükleniyor...")
    shingles_dict = {}
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        for doc_id, line in enumerate(f):
            if not line.strip(): continue
            record = json.loads(line)
            text = f"{record.get('headline', '')} {record.get('short_description', '')}"
            shingles_dict[doc_id] = get_shingles(text, k=3)
            
    total_docs = len(shingles_dict)
    total_combinations = (total_docs * (total_docs - 1)) // 2
    
    # İş paketlerinin boyutu: Her çekirdek tek seferde 5 milyon kombinasyon işlesin
    CHUNK_SIZE = 5_000_000 
    total_chunks = (total_combinations // CHUNK_SIZE) + 1
    
    print(f"\n2. Kaba Kuvvet (Brute-Force) Başlıyor...")
    print(f"Toplam Belge: {total_docs:,} | Toplam İşlem: {total_combinations:,}")
    print(f"Çekirdek Sayısı: {NUM_CORES} | Takip Çubuğu Aktif\n")
    
    ground_truth_pairs = set()
    start_time = time.time()
    
    # Tüm kombinasyonları anlık üreten iteratör
    pair_generator = combinations(range(total_docs), 2)
    chunk_generator = chunked_iterable(pair_generator, CHUNK_SIZE)
    
    # Paralel Havuz (Pool) Başlatılıyor
    with mp.Pool(processes=NUM_CORES, initializer=init_worker, initargs=(shingles_dict,)) as pool:
        
        # imap_unordered ve tqdm kombinasyonu sayesinde anlık takip!
        for result in tqdm(pool.imap_unordered(process_chunk, chunk_generator), 
                           total=total_chunks, 
                           desc="Hesaplanıyor", 
                           unit="paket"):
            
            # Gelen kopyaları ana havuza ekle
            ground_truth_pairs.update(result)

    elapsed = time.time() - start_time
    print(f"\nİşlem Tamamlandı! Süre: {elapsed / 3600:.2f} saat")
    print(f"Toplam Gerçek Kopya Bulundu: {len(ground_truth_pairs):,}")
    
    print("3. Diske kaydediliyor...")
    with open(GROUND_TRUTH_FILE, 'wb') as f:
        pickle.dump(ground_truth_pairs, f)
        
    print("Kayıt Başarılı!")