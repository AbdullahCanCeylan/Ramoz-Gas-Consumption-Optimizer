import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("1. Veri yükleniyor ve zaman serisine çevriliyor...")
df = pd.read_parquet("makine_verileri_temiz.parquet")
df['CreateDate'] = pd.to_datetime(df['CreateDate'])
df = df.sort_values('CreateDate').reset_index(drop=True)

print("2. Veri temizliği yapılıyor (BOM karakteri)...")
df.columns = df.columns.str.replace('\ufeff', '')

print("3. Tüketim ve üretim farkları (delta) hesaplanıyor...")
df['GasConsumption'] = df['MachineCounterGas'].diff()
df['ElectricConsumption'] = df['MachineCounterElectric'].diff()
df['MeterProduced'] = df['MachineCounterMeter'].diff()

# Sayaçların anlık düşüşleri veya sıfırlanmaları nedeniyle oluşan eksi (negatif) diferansiyel değerleri temizliyoruz
df = df.dropna()
df = df[(df['GasConsumption'] >= 0) & (df['MeterProduced'] >= 0) & (df['MachineSpeedActual'] > 0)]

print("4. Veri 10 dakikalık periyotlar halinde toplulaştırılıyor (Resampling)...")
# Gereksiz kümülatif sütunları düşürüyoruz
drop_cols = ['ID', 'MachineCounterMeter', 'MachineCounterElectric', 'MachineCounterGas', 'vardiya']
df_features = df.drop(columns=drop_cols, errors='ignore')

# Zaman damgasını index yapıyoruz (Toplulaştırma işlemi için zorunlu)
df_features.set_index('CreateDate', inplace=True)

# Hangi sütunların nasıl toplulaştırılacağını belirliyoruz
# Tüm sensör verileri için 10 dakikalık 'ortalama' (mean) alıyoruz
agg_dict = {col: 'mean' for col in df_features.columns}
# Ancak tüketim ve üretim miktarları için 10 dakikalık 'toplam' (sum) almalıyız
agg_dict['GasConsumption'] = 'sum'
agg_dict['ElectricConsumption'] = 'sum'
agg_dict['MeterProduced'] = 'sum'

# 10 dakikalık ('10min') bloklar halinde veriyi grupla
df_resampled = df_features.resample('10min').agg(agg_dict).dropna()

# Standart sapması 0 olan (tamamen sabit değer içeren) sütunları düşür
df_resampled = df_resampled.loc[:, df_resampled.std() > 0]

print("5. Spesifik Gaz Sarfiyatı (Hedef Değişken) hesaplanıyor...")
# Sadece o 10 dakikalık periyotta gerçekten üretim yapılan anları alıyoruz
df_resampled = df_resampled[df_resampled['MeterProduced'] > 0].copy()
df_resampled['SpesifikGazSarfiyati'] = df_resampled['GasConsumption'] / df_resampled['MeterProduced']

print("6. Aykırı değerler (Outliers) IQR yöntemiyle temizleniyor...")
Q1 = df_resampled['SpesifikGazSarfiyati'].quantile(0.25)
Q3 = df_resampled['SpesifikGazSarfiyati'].quantile(0.75)
IQR = Q3 - Q1
alt_sinir = Q1 - 1.5 * IQR
ust_sinir = Q3 + 1.5 * IQR

df_resampled = df_resampled[(df_resampled['SpesifikGazSarfiyati'] >= alt_sinir) & 
                            (df_resampled['SpesifikGazSarfiyati'] <= ust_sinir)]

print("7. Veri sızıntısını (Data Leakage) önlemek için kolonlar ayıklanıyor...")
y = df_resampled['SpesifikGazSarfiyati']
X = df_resampled.drop(columns=[
    'GasConsumption', 
    'ElectricConsumption', 
    'MeterProduced', 
    'SpesifikGazSarfiyati'
], errors='ignore')

print("8. Makine Öğrenmesi (Random Forest) modeli eğitiliyor...")
# Veriyi %80 Eğitim, %20 Test olarak ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("9. Özellik önem dereceleri hesaplanıyor...\n")
importance_df = pd.DataFrame({
    'Oznitelik': X.columns,
    'Onem_Derecesi': model.feature_importances_
}).sort_values(by='Onem_Derecesi', ascending=False)

print("--- 10 DAKİKALIK PERİYOTLARDA GAZ SARFİYATINA EN ÇOK ETKİ EDEN AYARLAR ---")
print(importance_df.head(15))

print("\n10. Model Doğruluğu Test Ediliyor...")
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- MODEL DOĞRULUK METRİKLERİ ---")
print(f"R-Kare (R²): {r2:.4f}")
print(f"Ortalama Mutlak Hata (MAE): {mae:.4f}")
print(f"Kök Ortalama Kare Hata (RMSE): {rmse:.4f}")

# Güncel tabloyu kaydet
df_resampled.to_parquet("10dk_toplulastirilmis_sarfiyat.parquet", engine='pyarrow')

# Görselleştirme
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='darkgreen', edgecolors='white', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Gerçek Spesifik Gaz Sarfiyatı (10 Dk. Ort.)')
plt.ylabel('Modelin Tahmin Ettiği Gaz Sarfiyatı')
plt.title('Gerçek vs Tahmin Edilen Değerler (Zaman Serisi Toplulaştırması Sonrası)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()