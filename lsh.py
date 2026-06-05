from itertools import combinations
import pickle
import time
from main import load_cache

class LSH:
    def __init__(self, b=20, r=5):
        """
        LSH Banding mimarisi.
        b: Bant sayısı (Bands)
        r: Her banttaki satır sayısı (Rows)
        Not: b * r = K (Toplam MinHash boyutu) olmalıdır.
        """
        self.b = b
        self.r = r
        
        # Olası eşik değerini (threshold) bilgi amaçlı hesaplıyoruz
        self.threshold = (1.0 / b) ** (1.0 / r)

    def get_candidate_pairs(self, signature_matrix):
        """
        İmza matrisini bantlara bölerek aday çiftleri (candidate pairs) bulur.
        """
        n_docs = len(signature_matrix)
        candidate_pairs = set() # Tekrar eden çiftleri önlemek için set kullanıyoruz
        
        print(f"LSH İşlemi Başladı (Bant: {self.b}, Satır: {self.r}, Yaklaşık Eşik: {self.threshold:.3f})")
        start_time = time.time()

        # Her bir bant için işlem yapıyoruz
        for band_idx in range(self.b):
            # Bu bant için hash kovalarını (buckets) tutacağımız sözlük
            buckets = {}
            
            start_row = band_idx * self.r
            end_row = start_row + self.r
            
            # Tüm belgeleri dön ve bu banta ait alt imzayı (sub-signature) çıkar
            for doc_id in range(n_docs):
                # Alt imzayı tuple'a çeviriyoruz ki sözlükte anahtar (key) olabilsin
                sub_signature = tuple(signature_matrix[doc_id][start_row:end_row])
                
                if sub_signature in buckets:
                    buckets[sub_signature].append(doc_id)
                else:
                    buckets[sub_signature] = [doc_id]
            
            # Bu banttaki kovaları incele
            for bucket_id, docs_in_bucket in buckets.items():
                # Eğer bir kovada 1'den fazla belge varsa, bunlar aday çifttir
                if len(docs_in_bucket) > 1:
                    # Kova içindeki belgelerin tüm ikili kombinasyonlarını oluştur
                    for pair in combinations(docs_in_bucket, 2):
                        # (1, 5) ile (5, 1) aynı şeydir, bu yüzden küçük ID'yi başa yazarak sıralıyoruz
                        sorted_pair = tuple(sorted(pair))
                        candidate_pairs.add(sorted_pair)

        elapsed_time = time.time() - start_time
        print(f"LSH İşlemi Tamamlandı! Süre: {elapsed_time:.2f} saniye")
        print(f"Bulunan Aday Çift Sayısı: {len(candidate_pairs):,}")
        
        return candidate_pairs

# ==========================================
# Test ve Simülasyon
# ==========================================
if __name__ == "__main__":
    CACHE_FILE = "minhash_cache.pkl"
    
    # 1. Adım: İmza matrisini ve URL sözlüğünü diskten (.pkl) yükle
    # (Bir önceki adımda yazdığımız load_cache fonksiyonunu çağırıyoruz)
    sig_matrix, metadata = load_cache(CACHE_FILE)
    
    if sig_matrix:
        print(f"\nYüklenen Matris Boyutu: {len(sig_matrix)} belge, her biri {len(sig_matrix[0])} boyutlu imza.")
        print("-" * 50)
        
        # 2. Adım: LSH Nesnesini Başlat
        # K = 100 olduğu için b=20 ve r=5 seçiyoruz. (20 * 5 = 100)
        # Jaccard benzerlik eşiği (threshold) yaklaşık (1/20)^(1/5) ≈ %54.9 olacaktır.
        lsh = LSH(b=20, r=5)
        
        # 3. Adım: Yüklenen .pkl verisini LSH filtresine sok ve Aday Çiftleri bul
        candidates = lsh.get_candidate_pairs(sig_matrix)
        
        # Sonuçları inceleme (İlk 5 aday çifti göster)
        print("\nÖrnek Aday Çiftler (Belge ID'leri):")
        for i, pair in enumerate(list(candidates)[:5]):
            doc1_id, doc2_id = pair
            print(f"Çift {i+1}: Belge {doc1_id} ve Belge {doc2_id}")
            
            # Eğer URL'leri görmek isterseniz metadata sözlüğünü kullanabilirsiniz:
            # print(f"  Link 1: {metadata[doc1_id]}")
            # print(f"  Link 2: {metadata[doc2_id]}")
    else:
        print("Lütfen önce imza matrisini oluşturup .pkl olarak kaydedin.")
# Aday çiftleri diske kaydet
with open("lsh_candidates.pkl", "wb") as f:
    pickle.dump(candidates, f)
print("\nAday çiftler 'lsh_candidates.pkl' dosyasına başarıyla kaydedildi!")