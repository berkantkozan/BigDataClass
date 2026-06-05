import time
import pickle
import csv

def export_report_to_csv(duplicates, metadata, filename="koordineli_icerik_raporu.csv"):
    """
    Kesinleşen kopya/koordineli içerikleri CSV formatında kaydeder.
    """
    print(f"\nRapor '{filename}' dosyasına dışa aktarılıyor...")
    
    # CSV dosyasını yazma modunda aç
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Başlık (Header) satırını yaz
        writer.writerow(["Belge_1_ID", "Belge_2_ID", "Benzerlik_Skoru", "Belge_1_URL", "Belge_2_URL"])
        
        # Tüm onaylanmış kopyaları satır satır yaz
        for doc1, doc2, score in duplicates:
            url1 = metadata.get(doc1, "Link Yok")
            url2 = metadata.get(doc2, "Link Yok")
            
            # Skoru yüzde formatında daha okunaklı hale getir
            formatted_score = f"%{score * 100:.1f}"
            
            writer.writerow([doc1, doc2, formatted_score, url1, url2])
            
    print(f"Dışa aktarma başarılı! Toplam {len(duplicates)} satır kaydedildi.")

def calculate_approximate_jaccard(candidate_pairs, signature_matrix, threshold=0.80):
    print(f"\n{len(candidate_pairs):,} aday çift için Jaccard benzerliği hesaplanıyor...")
    start_time = time.time()
    
    K = len(signature_matrix[0]) 
    confirmed_duplicates = []
    
    for doc1_id, doc2_id in candidate_pairs:
        sig1 = signature_matrix[doc1_id]
        sig2 = signature_matrix[doc2_id]
        
        matches = sum(1 for i in range(K) if sig1[i] == sig2[i])
        similarity_score = matches / K
        
        if similarity_score >= threshold:
            confirmed_duplicates.append((doc1_id, doc2_id, similarity_score))
            
    confirmed_duplicates.sort(key=lambda x: x[2], reverse=True)
    
    elapsed_time = time.time() - start_time
    print(f"Hesaplama Tamamlandı! Süre: {elapsed_time:.2f} saniye")
    print(f"Eşik Değerini (>= {threshold}) Aşan Kopya/Koordineli İçerik Sayısı: {len(confirmed_duplicates):,}")
    
    return confirmed_duplicates

# ==========================================
# Ana Akış
# ==========================================
if __name__ == "__main__":
    SIMILARITY_THRESHOLD = 0.80 
    
    # 1. İmza matrisi ve Metadata'yı yükle
    print("İmza matrisi yükleniyor...")
    with open("minhash_cache.pkl", "rb") as f:
        sig_matrix, metadata = pickle.load(f)
        
    # 2. LSH'in ürettiği Aday Çiftleri yükle
    print("Aday çiftler yükleniyor...")
    with open("lsh_candidates.pkl", "rb") as f:
        candidates = pickle.load(f)
        
    # 3. Jaccard Hesaplamasını Çalıştır
    final_duplicates = calculate_approximate_jaccard(
        candidates, 
        sig_matrix, 
        threshold=SIMILARITY_THRESHOLD
    )
    
    # 4. Sonuçları İncele
    print("-" * 50)
    print("Tespit Edilen En Yüksek Benzerlikli 5 Çift:")
    for i, (doc1, doc2, score) in enumerate(final_duplicates[:5]):
        print(f"\n[{i+1}] Benzerlik Skoru: %{score * 100:.1f}")
        print(f"  Belge 1 ID: {doc1} | URL: {metadata[doc1]}")
        print(f"  Belge 2 ID: {doc2} | URL: {metadata[doc2]}")
    # 5. RAPORU KAYDET (YENİ EKLENEN KISIM)
    export_report_to_csv(final_duplicates, metadata)