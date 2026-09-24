import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore') # SciPy'ın gereksiz uyarılarını gizlemek için

print("1. Veri yükleniyor ve Dijital İkiz (Model) hazırlanıyor...")
# Bir önceki adımda kaydettiğimiz toplulaştırılmış veriyi okuyalım
df = pd.read_parquet("D:/Staj/adım 2/10dk_toplulastirilmis_sarfiyat.parquet")

# Hedef ve özellikleri belirleyip modelimizi tekrar eğitiyoruz
y = df['SpesifikGazSarfiyati']
X = df.drop(columns=['GasConsumption', 'ElectricConsumption', 'MeterProduced', 'SpesifikGazSarfiyati'], errors='ignore')

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

print("2. Fabrikadan verimsiz bir an (kötü bir senaryo) seçiliyor...")
# Spesifik gaz sarfiyatının ortalamanın üzerinde olduğu (verimsiz) bir anı bulalım
verimsiz_indeks = df[df['SpesifikGazSarfiyati'] > df['SpesifikGazSarfiyati'].mean()].index[0]
ornek_veri = X.loc[verimsiz_indeks].copy()
gercek_sarfiyat = y.loc[verimsiz_indeks]

print("3. Müdahale Edilecek Parametreler (Sıcaklık ve Nem) Belirleniyor...")
# Optimizasyon motorunun dokunmasına (değiştirmesine) izin verdiğimiz ayarlar
optimize_edilecek_kolonlar = [
    'Cabinet1TemperatureAct', 
    'Cabinet6TemperatureAct', 
    'HumidityExhaustActual'
]

# Algoritmanın fiziksel sınırların dışına (örneğin -50 dereceye) çıkmasını engellemek için
# veri setindeki minimum ve maksimum çalışma değerlerini sınır (bounds) olarak veriyoruz
sinirlar = []
for col in optimize_edilecek_kolonlar:
    sinirlar.append((X[col].min(), X[col].max()))

print("4. SciPy Optimizasyon Algoritması çalıştırılıyor (Yüzlerce senaryo simüle ediliyor)...")
# Amaç Fonksiyonu: Algoritmanın "minimize" etmeye (en aza indirmeye) çalışacağı hedef
def amac_fonksiyonu(opt_degerler):
    simule_veri = ornek_veri.copy()
    
    # SciPy'ın denediği yeni ayarları, bizim simüle verimize yazıyoruz
    for i, col in enumerate(optimize_edilecek_kolonlar):
        simule_veri[col] = opt_degerler[i]
        
    # Modelden (Dijital İkizden) bu yeni ayarlarla gaz sarfiyatını tahmin etmesini istiyoruz
    tahmin_edilen_sarfiyat = model.predict(simule_veri.values.reshape(1, -1))[0]
    return tahmin_edilen_sarfiyat

# Başlangıç noktası olarak makinenin o anki gerçek ayarlarını veriyoruz
baslangic_degerleri = [ornek_veri[col] for col in optimize_edilecek_kolonlar]

# Powell algoritması, Ağaç (Random Forest) modellerindeki kesintili yüzeyler için en iyi optimizasyon algoritmalarından biridir
sonuc = minimize(amac_fonksiyonu, baslangic_degerleri, method='Powell', bounds=sinirlar)

print("\n=======================================================")
print("             YAPAY ZEKA OPTİMİZASYON RAPORU            ")
print("=======================================================")
print(f"Mevcut (Gerçek) Spesifik Sarfiyat : {gercek_sarfiyat:.4f} m3/metre")
print(f"Yapay Zekanın Bulduğu Min Sarfiyat: {sonuc.fun:.4f} m3/metre")
print(f"-------------------------------------------------------")
print(f"DİJİTAL İKİZ İLE BULUNAN TASARRUF : % {((gercek_sarfiyat - sonuc.fun) / gercek_sarfiyat) * 100:.2f}")
print("=======================================================\n")

print("--- OLMASI GEREKEN EN İYİ MAKİNE AYARLARI (REÇETE) ---")
for i, col in enumerate(optimize_edilecek_kolonlar):
    print(f"{col}:")
    print(f"  -> Eski Operatör Ayarı: {baslangic_degerleri[i]:.2f}")
    print(f"  -> Algoritmanın Önerisi: {sonuc.x[i]:.2f}\n")