import json
import time
import re
from itertools import combinations, islice
import csv
import multiprocessing as mp
from tqdm import tqdm
import pickle

# --- Ön İşleme ---
def preprocess_text(text):
    text = text.lower()
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', text)).strip()

def get_shingles(text, k=3):
    tokens = preprocess_text(text).split()
    return {" ".join(tokens[i:i+k]) for i in range(len(tokens) - k + 1)} if len(tokens) >= k else ({" ".join(tokens)} if tokens else set())

# --- Bellek Paylaşımı ve İşçi (Worker) ---
global_shingles = {}
def init_worker(shared_dict):
    global global_shingles
    global_shingles = shared_dict

def process_chunk(chunk_pairs):
    THRESHOLD = 0.80
    found = set()
    for doc1, doc2 in chunk_pairs:
        s1, s2 = global_shingles.get(doc1), global_shingles.get(doc2)
        if not s1 or not s2: continue
        union = len(s1.union(s2))
        if union > 0 and (len(s1.intersection(s2)) / union) >= THRESHOLD:
            found.add((doc1, doc2))
    return found

def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, size))
        if not chunk: break
        yield chunk

if __name__ == "__main__":
    DATASET_FILE = "News_Category_Dataset_v3.json"
    LSH_REPORT_FILE = "koordineli_icerik_raporu.csv"
    GROUND_TRUTH_FILE = "ground_truth_FULL.pkl" # Yeni isimle kaydediyoruz
    NUM_CORES = 12 
    
    print("1. TÜM VERİ SETİ (209.527 Belge) RAM'e Yükleniyor...")
    shingles_dict = {}
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        for doc_id, line in enumerate(f):
            if not line.strip(): continue
            record = json.loads(line)
            text = f"{record.get('headline', '')} {record.get('short_description', '')}"
            shingles_dict[doc_id] = get_shingles(text, k=3)

    total_docs = len(shingles_dict)
    total_combinations = (total_docs * (total_docs - 1)) // 2
    
    print(f"\n2. KESİN GERÇEKLİK (BRUTE-FORCE) BAŞLIYOR...")
    print(f"Toplam İşlem: {total_combinations:,} (Yaklaşık 22 Milyar)")
    print(f"Uyarı: Bu işlem yaklaşık 2.5 - 3 saat sürebilir.")
    
    CHUNK_SIZE = 2_000_000 # Hızlı ilerleme için chunk boyutunu büyüttük
    total_chunks = (total_combinations // CHUNK_SIZE) + 1
    
    ground_truth_pairs = set()
    start_time = time.time()
    
    # Random yok, doğrudan tüm iterasyon
    pair_generator = combinations(range(total_docs), 2)
    chunk_generator = chunked_iterable(pair_generator, CHUNK_SIZE)
    
    with mp.Pool(processes=NUM_CORES, initializer=init_worker, initargs=(shingles_dict,)) as pool:
        for result in tqdm(pool.imap_unordered(process_chunk, chunk_generator), 
                           total=total_chunks, desc="Devasa Hesaplama"):
            ground_truth_pairs.update(result)

    print(f"\nİşlem Tamamlandı! Süre: {(time.time() - start_time)/3600:.2f} Saat")
    
    # 3. Diske Kayıt
    print(f"\n3. Kesin kopyalar '{GROUND_TRUTH_FILE}' dosyasına kaydediliyor...")
    with open(GROUND_TRUTH_FILE, "wb") as f:
        pickle.dump(ground_truth_pairs, f)
        
    # 4. Filtresiz Tam Metrik Hesabı
    print("\n4. Tam Veri Seti Metrikleri Hesaplanıyor...")
    lsh_found_pairs = set()
    with open(LSH_REPORT_FILE, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader) 
        for row in reader:
            # Artık filtre yok, LSH'in bulduğu HER ŞEYİ alıyoruz
            lsh_found_pairs.add(tuple(sorted([int(row[0]), int(row[1])])))

    tp = len(lsh_found_pairs.intersection(ground_truth_pairs))
    fp = len(lsh_found_pairs - ground_truth_pairs)
    fn = len(ground_truth_pairs - lsh_found_pairs)
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print("\n" + "="*40)
    print("      KUSURSUZ (FULL DATASET) METRİKLERİ")
    print("="*40)
    print(f"Doğru Pozitif (TP): {tp:,}")
    print(f"Yanlış Negatif (Kaçanlar - FN): {fn:,}")
    print(f"Yanlış Pozitif (Hatalı LSH - FP): {fp:,}")
    print("-" * 40)
    print(f"Recall (Duyarlılık): %{recall*100:.2f}")
    print(f"Precision (Kesinlik): %{precision*100:.2f}")
    print(f"F1-Score: %{f1_score*100:.2f}")
    print("="*40)