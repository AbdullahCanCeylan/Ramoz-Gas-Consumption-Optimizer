import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("1. Veri yükleniyor ve temizleniyor...")
# Ham veriyi okuyoruz (toplulaştırılmamış orijinal veri)
df = pd.read_parquet("D:/Staj/makine_verileri_temiz.parquet")
df.columns = df.columns.str.replace('\ufeff', '')

print("2. Makinenin çalıştığı aktif durumlar filtreleniyor...")
# Sadece makinenin hızının 0'dan büyük olduğu ve egzoz fanlarının devrede olduğu anları alıyoruz
df_active = df[(df['MachineSpeedActual'] > 0) & 
               (df['Exhaust1SpeedAct'] > 0) & 
               (df['Exhaust2SpeedAct'] > 0) & 
               (df['HumidityExhaustActual'] > 0)].copy()

# Analiz edeceğimiz 3 temel sütunu seçiyoruz
cols_of_interest = ['Exhaust1SpeedAct', 'Exhaust2SpeedAct', 'HumidityExhaustActual']
df_exhaust = df_active[cols_of_interest]

print("\n--- EGZOZ FANLARI VE NEM ARASINDAKİ KORELASYON MATRİSİ ---")
# Korelasyon matrisini hesapla
correlation = df_exhaust.corr()
print(correlation)

print("\n3. Grafikler hazırlanıyor...")
# İkili (Dual) grafik ekranı oluşturuyoruz
plt.figure(figsize=(14, 6))

# 1. Grafik: Egzoz 1 ve Nem İlişkisi
plt.subplot(1, 2, 1)
# regplot: Hem noktaları (scatter) çizer hem de aralarındaki matematiksel eğilimi (regresyon çizgisi) gösterir
sns.regplot(x='Exhaust1SpeedAct', y='HumidityExhaustActual', data=df_exhaust, 
            scatter_kws={'alpha':0.2, 'color':'blue'}, line_kws={'color':'red', 'lw':2})
plt.title('1. Egzoz Fan Devri vs Kabin İçi Nem', fontsize=12, fontweight='bold')
plt.xlabel('Egzoz 1 Hızı (Act)', fontsize=10)
plt.ylabel('Egzoz Nemi (Actual)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

# 2. Grafik: Egzoz 2 ve Nem İlişkisi
plt.subplot(1, 2, 2)
sns.regplot(x='Exhaust2SpeedAct', y='HumidityExhaustActual', data=df_exhaust, 
            scatter_kws={'alpha':0.2, 'color':'green'}, line_kws={'color':'red', 'lw':2})
plt.title('2. Egzoz Fan Devri vs Kabin İçi Nem', fontsize=12, fontweight='bold')
plt.xlabel('Egzoz 2 Hızı (Act)', fontsize=10)
plt.ylabel('Egzoz Nemi (Actual)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()