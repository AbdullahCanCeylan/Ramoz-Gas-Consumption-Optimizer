import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt

print("1. Veri yükleniyor ve model eğitiliyor...")
# 10 dakikalık periyotlarla pürüzsüzleştirilmiş verimizi okuyoruz
df = pd.read_parquet("D:/Staj/adım 2/10dk_toplulastirilmis_sarfiyat.parquet")

# Hedef ve özellikleri (features) ayırıyoruz
y = df['SpesifikGazSarfiyati']
X = df.drop(columns=[
    'GasConsumption', 
    'ElectricConsumption', 
    'MeterProduced', 
    'SpesifikGazSarfiyati'
], errors='ignore')

# Modeli tüm veriyle eğitiyoruz
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

print("2. SHAP (Açıklanabilir Yapay Zeka) analizi başlatılıyor...")
print("Bu işlem veri boyutuna göre 15-30 saniye sürebilir, lütfen bekleyin...")

# TreeExplainer, ağaç tabanlı algoritmalar (Random Forest, XGBoost vb.) için optimize edilmiştir
explainer = shap.TreeExplainer(model)

# SHAP değerlerini hesaplıyoruz (Her bir satırın tahmine olan pozitif/negatif katkısı)
shap_values = explainer.shap_values(X)

print("3. SHAP Özeti (Summary Plot) çizdiriliyor...")
# SHAP Summary Plot çizimi
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X, show=False)

# Grafiğe başlık ekleyip ekranda gösteriyoruz
plt.title("SHAP Analizi: Makine Ayarlarının Gaz Sarfiyatına Yönsel Etkileri", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()