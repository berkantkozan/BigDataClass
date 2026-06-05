import pickle
import csv
import matplotlib.pyplot as plt
import seaborn as sns

def load_full_metrics():
    # Yeni oluşturduğumuz KUSURSUZ dosyamız
    ground_truth_file = "ground_truth_FULL.pkl" 
    lsh_report_file = "koordineli_icerik_raporu.csv"
    
    print("Veriler yükleniyor...")
    try:
        with open(ground_truth_file, 'rb') as f:
            ground_truth_pairs = pickle.load(f)
    except FileNotFoundError:
        print("Hata: Ground Truth dosyası bulunamadı.")
        return None

    lsh_found_pairs = set()
    try:
        with open(lsh_report_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader) 
            for row in reader:
                lsh_found_pairs.add(tuple(sorted([int(row[0]), int(row[1])])))
    except FileNotFoundError:
        print("Hata: LSH raporu CSV dosyası bulunamadı.")
        return None

    # Tam Veri Seti Metrikleri
    tp = len(lsh_found_pairs.intersection(ground_truth_pairs))
    fp = len(lsh_found_pairs - ground_truth_pairs)
    fn = len(ground_truth_pairs - lsh_found_pairs)
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "recall": recall, "precision": precision, "f1_score": f1_score
    }

def create_final_visualizations(metrics):
    if not metrics: return

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('Tam Veri Seti LSH Performans Analizi (210.000 Belge)', fontsize=16, fontweight='bold')

    # 1. Bar Chart
    percentages = [metrics['recall'] * 100, metrics['precision'] * 100, metrics['f1_score'] * 100]
    labels = ['Duyarlılık\n(Recall)', 'Kesinlik\n(Precision)', 'F1-Skoru']
    colors = ['#3498db', '#2ecc71', '#9b59b6']

    bars = axes[0].bar(labels, percentages, color=colors, width=0.6)
    axes[0].set_ylim(0, 110)
    axes[0].set_ylabel('Yüzde (%)', fontweight='bold')
    axes[0].set_title('Sınıflandırma Başarısı', fontsize=14)

    for bar in bars:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, yval + 1.5, f'%{yval:.2f}', 
                     ha='center', va='bottom', fontweight='bold', fontsize=12)

    # 2. Donut Chart
    sizes = [metrics['tp'], metrics['fn'], metrics['fp']]
    labels_pie = [f"Doğru Tespit (TP)\n{metrics['tp']:,}", 
                  f"Kaçırılan (FN)\n{metrics['fn']:,}", 
                  f"Hatalı Eşleşme (FP)\n{metrics['fp']:,}"]
    colors_pie = ['#2ecc71', '#e74c3c', '#f1c40f']

    axes[1].pie(sizes, explode=(0.05, 0.05, 0.05), labels=labels_pie, colors=colors_pie, 
                autopct='%1.1f%%', shadow=False, startangle=140, 
                textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    axes[1].add_artist(centre_circle)
    axes[1].set_title('Evrensel Tahmin Dağılımı', fontsize=14)

    plt.tight_layout()
    plt.savefig('tam_veriseti_lsh_analizi.png', dpi=300, bbox_inches='tight')
    print("Grafik başarıyla oluşturuldu: 'tam_veriseti_lsh_analizi.png'")
    plt.show()

if __name__ == "__main__":
    metrics_data = load_full_metrics()
    create_final_visualizations(metrics_data)