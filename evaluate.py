import pickle
import csv

def evaluate_metrics(ground_truth_file, lsh_report_csv):
    print("Veriler yükleniyor...\n" + "-"*30)
    
    # 1. Kaba Kuvvet Sonuçlarını (Ground Truth) Yükle
    try:
        with open(ground_truth_file, 'rb') as f:
            ground_truth_pairs = pickle.load(f)
        print(f"Kesin Gerçeklik (Ground Truth) Yüklendi: {len(ground_truth_pairs):,} kopya")
    except FileNotFoundError:
        print(f"Hata: '{ground_truth_file}' bulunamadı. Önce kaba kuvvet hesaplamasını yapın.")
        return

    # 2. LSH'in Bulduğu Sonuçları Yükle
    lsh_found_pairs = set()
    try:
        with open(lsh_report_csv, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader) # Başlığı atla
            for row in reader:
                pair = tuple(sorted([int(row[0]), int(row[1])]))
                lsh_found_pairs.add(pair)
        print(f"LSH Raporu Yüklendi: {len(lsh_found_pairs):,} kopya")
    except FileNotFoundError:
        print(f"Hata: '{lsh_report_csv}' bulunamadı.")
        return

    # 3. Kesişim (Set Intersection) İşlemleri (Çok Hızlıdır)
    tp = len(lsh_found_pairs.intersection(ground_truth_pairs)) # İkisinin de buldukları
    fp = len(lsh_found_pairs - ground_truth_pairs)             # LSH'in uydurdukları
    fn = len(ground_truth_pairs - lsh_found_pairs)             # LSH'in kaçırdıkları
    
    # 4. Matematiksel Formüller
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 5. Sonuç Raporu
    print("\n" + "="*40)
    print("        PERFORMANS METRİKLERİ")
    print("="*40)
    print(f"Gerçekte Var Olan Toplam Kopya: {len(ground_truth_pairs):,}")
    print(f"Doğru Pozitif (True Positive): {tp:,}")
    print(f"Yanlış Negatif (Kaçanlar - FN): {fn:,}")
    print(f"Yanlış Pozitif (Hatalı LSH - FP): {fp:,}")
    print("-" * 40)
    print(f"Recall (Duyarlılık): %{recall*100:.2f}")
    print(f"Precision (Kesinlik): %{precision*100:.2f}")
    print(f"F1-Score: %{f1_score*100:.2f}")
    print("="*40)

if __name__ == "__main__":
    GROUND_TRUTH_FILE = "ground_truth_pairs.pkl"
    LSH_REPORT_FILE = "koordineli_icerik_raporu.csv"
    
    evaluate_metrics(GROUND_TRUTH_FILE, LSH_REPORT_FILE)