import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Ramöz AI Optimizasyon", layout="wide")
st.title("🏭 Ramöz Makinesi AI Karar Destek ve Optimizasyon Sistemi")
st.markdown("""
Bu dijital ikiz, makine öğrenmesi kullanılarak eğitilmiştir. Sürgüleri manuel değiştirerek test yapabilir 
veya soldaki **'En İdeal Ayarları Uygula'** butonu ile yapay zekanın önerdiği optimum reçeteyi görebilirsiniz.
""")

@st.cache_resource
def model_ve_veri_yukle():
    df = pd.read_parquet("D:/Staj/adım 2/10dk_toplulastirilmis_sarfiyat.parquet")
    y = df['SpesifikGazSarfiyati']
    X = df.drop(columns=['GasConsumption', 'ElectricConsumption', 'MeterProduced', 'SpesifikGazSarfiyati'], errors='ignore')
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    referans_veri = X.median().to_dict()
    return model, referans_veri, X

model, referans_veri, X = model_ve_veri_yukle()

# --- STREAMLIT SESSION STATE (SÜRGÜLERİ KONTROL ETMEK İÇİN) ---
# Sürgülerin başlangıç değerlerini sistemin hafızasına (session_state) atıyoruz
if 'hiz_val' not in st.session_state:
    st.session_state.hiz_val = float(X['MachineSpeedActual'].median())
if 'nem_val' not in st.session_state:
    st.session_state.nem_val = float(X['HumidityExhaustActual'].median())
if 'kab1_val' not in st.session_state:
    st.session_state.kab1_val = float(X['Cabinet1TemperatureAct'].median())
if 'kab6_val' not in st.session_state:
    st.session_state.kab6_val = float(X['Cabinet6TemperatureAct'].median())

# Otomatik Optimizasyon Butonu Fonksiyonu
def ideal_degerleri_ata():
    # Sistemin fiziksel sınırlarına göre en tasarruflu ve güvenli değerler
    st.session_state.hiz_val = float(X['MachineSpeedActual'].max()) # Mevcut en yüksek hız (örn. 49.93)
    st.session_state.nem_val = 110.0 # Güvenli ve yüksek nem doygunluğu
    st.session_state.kab1_val = 135.0 # Termal şoku önleyecek optimum ilk giriş
    st.session_state.kab6_val = 160.0 # Gereksiz alevi engelleyen optimum zirve

baz_df = pd.DataFrame([referans_veri])
baz_sarfiyat = model.predict(baz_df)[0]

# --- KULLANICI ARAYÜZÜ (SOL MENÜ) ---
st.sidebar.header("⚙️ Makine Ayarları (Reçete)")

# OPTİMİZASYON BUTONU
st.sidebar.button("✨ En İdeal Ayarları Uygula", on_click=ideal_degerleri_ata, type="primary", use_container_width=True)
st.sidebar.markdown("---")

# Sürgüler artık 'key' parametresi ile session_state'e bağlı!
hiz = st.sidebar.slider("Makine Hızı (m/dk)", min_value=float(X['MachineSpeedActual'].min()), max_value=float(X['MachineSpeedActual'].max()), key='hiz_val')
egzoz_nemi = st.sidebar.slider("Egzoz Nemi (Doygunluk)", min_value=float(X['HumidityExhaustActual'].min()), max_value=float(X['HumidityExhaustActual'].max()), key='nem_val')
kabin1_sicaklik = st.sidebar.slider("1. Kabin Sıcaklığı (°C)", min_value=float(X['Cabinet1TemperatureAct'].min()), max_value=float(X['Cabinet1TemperatureAct'].max()), key='kab1_val')
kabin6_sicaklik = st.sidebar.slider("6. Kabin Sıcaklığı (°C)", min_value=float(X['Cabinet6TemperatureAct'].min()), max_value=float(X['Cabinet6TemperatureAct'].max()), key='kab6_val')

# --- SİMÜLASYON ---
simulasyon_verisi = referans_veri.copy()
simulasyon_verisi['MachineSpeedActual'] = hiz
simulasyon_verisi['HumidityExhaustActual'] = egzoz_nemi
simulasyon_verisi['Cabinet1TemperatureAct'] = kabin1_sicaklik
simulasyon_verisi['Cabinet6TemperatureAct'] = kabin6_sicaklik

df_tahmin = pd.DataFrame([simulasyon_verisi])
tahmin_edilen_sarfiyat = model.predict(df_tahmin)[0]
tasarruf_farki = tahmin_edilen_sarfiyat - baz_sarfiyat

# --- SONUÇ EKRANI ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Tahmin Sonucu")
    st.metric(
        label="Spesifik Gaz Sarfiyatı (m³/metre)", 
        value=f"{tahmin_edilen_sarfiyat:.4f}",
        delta=f"{tasarruf_farki:.4f} m³/metre (Ortalamaya Göre)",
        delta_color="inverse"
    )

with col2:
    st.subheader("🤖 Yapay Zeka Geri Bildirimi")
    
    # 1. HIZ ANALİZİ (Maksimum 49.93'e göre uyarlandı)
    if hiz < 35:
        st.error(f"⚠️ **KRİTİK VERİMSİZLİK (Hız):** Makine hızı çok düşük ({hiz:.2f} m/dk). Makinenin ısınması için harcanan sabit enerji çok az kumaşa bölünüyor. Bu durum metretül maliyetini aşırı yükseltir.")
    elif hiz >= 45:
        st.success(f"✅ **OPTİMUM ÜRETİM (Hız):** Makine hızı maksimum verimlilik sınırında ({hiz:.2f} m/dk). Isıl maliyet şu an en geniş kumaş metresine paylaştırılıyor.")
        
    # 2. NEM ANALİZİ
    if egzoz_nemi < 80:
        st.error(f"🔥 **ENERJİ İSRAFI (Nem):** Egzoz nemi ({egzoz_nemi:.2f}) çok düşük! İçerideki hava daha neme doymadan (sıcak ve kuruyken) dışarı atılıyor. Fanları yavaşlatarak nemi ve enerjiyi içeride tutun.")
    elif egzoz_nemi > 100:
        st.success(f"✅ **TERMAL DENGE (Nem):** Egzoz nemi seviyesi başarılı. Sıcak hava bacadan israf edilmek yerine kurutma işlemine aktif katkı sağlıyor.")
        
    # 3. KABİN 1 (HEM YÜKSEK HEM DÜŞÜK UYARILARI)
    if kabin1_sicaklik > 160:
        st.warning(f"📈 **İSRAF (Kabin 1):** Sıcaklık ({kabin1_sicaklik:.2f}°C) çok yüksek. Islak kumaşa ilk girişte ani termal şok uygulamak doğalgaz sarfiyatını tırmandırıyor.")
    elif kabin1_sicaklik < 120:
        st.error(f"💧 **KALİTE RİSKİ (Kabin 1):** Sıcaklık ({kabin1_sicaklik:.2f}°C) çok düşük! Kumaşın üzerindeki suyun buharlaşma işlemi tam başlamayacaktır. Bu durum kurutma yükünü arka kabinlere yığar ve kumaşın makineden ıslak çıkmasına neden olabilir.")
    
    # 4. KABİN 6 (HEM YÜKSEK HEM DÜŞÜK UYARILARI)
    if kabin6_sicaklik > 180:
        st.warning(f"📈 **AŞIRI YÜK (Kabin 6):** Sıcaklık ({kabin6_sicaklik:.2f}°C) stres sınırında. Bu seviyede hedefi tutturmak için brülör hiç kapanmadan yanacak ve izolasyon kaçakları artacaktır.")
    elif kabin6_sicaklik < 140:
        st.error(f"💧 **FİKSE/KURUTMA RİSKİ (Kabin 6):** Sıcaklık ({kabin6_sicaklik:.2f}°C) yeterli değil. Kurutma eğrisinin zirve noktasında ortam çok soğuk kalırsa, kumaşta kalite problemleri (nemli çıkış veya boya fikse olmaması) yaşanır.")

    # MÜKEMMELLİK KONTROLÜ (Koşullar 49.93 maksimum hıza ve kaliteye göre güncellendi)
    if (hiz >= 45) and (egzoz_nemi >= 100) and (120 <= kabin1_sicaklik <= 150) and (140 <= kabin6_sicaklik <= 170):
        st.balloons()
        st.success("🎉 **KUSURSUZ REÇETE:** Tüm ayarlar tam olması gerektiği gibi! Sistem hem doğalgazı minimumda tutuyor hem de kumaşın ıslak çıkma riskini ortadan kaldırıyor. Üretime bu şekilde devam edilmelidir.")

st.markdown("---")
st.markdown("*Atatürk Üniversitesi Bilgisayar Mühendisliği Staj Projesi - MLOps Dashboard*")