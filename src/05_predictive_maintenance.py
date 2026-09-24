import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("1. Veri yükleniyor...")
# 10 dakikalık toplulaştırılmış (gürültüden arındırılmış) verimizi okuyoruz
df = pd.read_parquet("D:/Staj/adım 2/10dk_toplulastirilmis_sarfiyat.parquet")

print("2. Kabinlerin Termal Performans ve Arıza Riskleri Hesaplanıyor...")
kabin_hata_oranlari = {}

# Ramöz makinesindeki 10 kabin için döngü oluşturuyoruz
for i in range(1, 11):
    set_col = f'Cabinet{i}TemperatureSet'
    act_col = f'Cabinet{i}TemperatureAct'
    status_col = f'Cabinet{i}TemperatureStatus'
    
    # İlgili kabinin sütunları veri setinde var mı kontrol edelim
    if set_col in df.columns and act_col in df.columns and status_col in df.columns:
        
        # SADECE BRÜLÖRÜN AKTİF OLDUĞU ANLARI FİLTRELİYORUZ
        # Toplulaştırılmış veride Status 0-1 arası ortalama bir değerdir. 
        # > 0.5 olması, o 10 dakikalık periyodun en az yarısında brülörün ateşleme yaptığını gösterir.
        df_aktif = df[df[status_col] > 0.5].copy()
        
        if len(df_aktif) > 0:
            # Steady-State Error (Kalıcı Durum Hatası) hesaplaması: Hedef - Gerçekleşen
            df_aktif['Temp_Error'] = df_aktif[set_col] - df_aktif[act_col]
            
            # Anomali Kriteri: Brülör çalışmasına rağmen sıcaklık hedefin 5°C veya daha fazla altındaysa
            anomali_sayisi = len(df_aktif[df_aktif['Temp_Error'] > 5])
            toplam_durum = len(df_aktif)
            
            # Zorlanma / Isı Kaçağı oranını yüzde olarak hesapla
            hata_orani = (anomali_sayisi / toplam_durum) * 100
            kabin_hata_oranlari[f'Kabin {i}'] = hata_orani

print("3. Analiz sonuçları sıralanıyor ve görselleştiriliyor...")
# Sonuçları bir DataFrame'e çevirip en sorunlu kabinden en sağlam olana doğru sıralayalım
df_hata = pd.DataFrame(list(kabin_hata_oranlari.items()), columns=['Kabin', 'Zorlanma_Orani_Yuzde'])
df_hata = df_hata.sort_values(by='Zorlanma_Orani_Yuzde', ascending=False)

print("\n--- KABİN ISI KAÇAĞI VE KESTİRİMCİ BAKIM RAPORU ---")
print(df_hata.to_string(index=False))

# Bar grafiği ile görselleştirme
plt.figure(figsize=(12, 6))
# Kırmızı tonları kullanarak tehlike seviyesini vurguluyoruz (Koyu kırmızı = Yüksek Risk)
sns.barplot(x='Zorlanma_Orani_Yuzde', y='Kabin', data=df_hata, palette='Reds_r')

plt.title('Kestirimci Bakım: Kabinlerin Hedef Sıcaklığa Ulaşamama (Termal Zorlanma) Oranları', fontsize=14, fontweight='bold')
plt.xlabel('Brülör Açıkken Hedefin 5°C Altında Kalma Oranı (%)', fontsize=12)
plt.ylabel('Kabinler', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()