# 🏭 Ramöz Makinesi AI Karar Destek ve Optimizasyon Sistemi

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-orange?style=for-the-badge&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-MLOps-red?style=for-the-badge&logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-Data_Engineering-green?style=for-the-badge&logo=pandas)

Tekstil endüstrisinde en yoğun enerji tüketen ekipmanlardan biri olan Ramöz (Stenter) makineleri için geliştirilmiş, **Endüstri 4.0** tabanlı uçtan uca bir Makine Öğrenmesi (MLOps) ve Kuralcı Analitik projesidir. SCADA sisteminden alınan 120'den fazla sensör verisi işlenerek makinenin "Dijital İkizi" oluşturulmuş ve fiziksel donanıma müdahale edilmeden doğalgaz tüketiminde **%4.96 net tasarruf** sağlayan algoritmik bir Karar Destek Sistemi canlıya alınmıştır.

---

## 🌟 Öne Çıkan Özellikler

*   📊 **Dijital İkiz (Digital Twin):** Random Forest Regressor kullanılarak, makinenin spesifik doğalgaz sarfiyatı **%84.18 (R²)** doğruluk oranıyla modellenmiştir.
*   🧠 **Kuralcı Analitik (Prescriptive Analytics):** SciPy (Powell Optimizesi) kullanılarak, üretim hızını düşürmeden enerji tüketimini minimize eden "En İdeal Makine Reçetesi" dinamik olarak hesaplanmaktadır.
*   🛠️ **Kestirimci Bakım (Predictive Maintenance):** Kalıcı Durum Hatası (Steady-State Error) prensibiyle, 10 farklı ısıtma kabinindeki donanımsal izolasyon kaçakları ve arıza riskleri sensör verileri üzerinden tespit edilmektedir.
*   📈 **Açıklanabilir Yapay Zeka (XAI):** SHAP entegrasyonu ile makine ayarlarının (fan devirleri, sıcaklıklar) fatura üzerindeki yönsel etkileri "Karakutu" olmaktan çıkarılıp matematiksel olarak ispatlanmıştır.
*   💻 **MLOps Dashboard:** Fabrika operatörlerinin anlık kâr/zarar simülasyonları yapabilmesi ve kalite kontrol uyarılarını görebilmesi için Streamlit ile Session State mimarisine sahip interaktif bir arayüz geliştirilmiştir.

---

## 🏗️ Proje Mimarisi ve Veri Boru Hattı (Pipeline)

Proje, yazılım mühendisliği standartlarına uygun olarak 5 ana fazda tasarlanmıştır:
1. **Veri Mühendisliği:** Ham `.rpt` logları Regex ile temizlenmiş, BOM karakterlerinden arındırılmış ve bellek optimizasyonu için yüksek performanslı `.parquet` formatına dönüştürülmüştür.
2. **Sinyal İşleme:** Sensör gecikmelerini (latensi) sönümlemek için Pandas ile 10 dakikalık zaman serisi toplulaştırması (Resampling) uygulanmıştır.
3. **Aykırı Değer (Outlier) Temizliği:** İstatistiksel IQR (Çeyrekler Açıklığı) yöntemi kullanılarak gürültülü üretim rejimleri filtrelenmiştir.
4. **Model Eğitimi:** Scikit-learn ile Karar Ağacı tabanlı topluluk modelleri eğitilmiş ve hiperparametre optimizasyonları yapılmıştır.
5. **Ürünleştirme:** Streamlit Caching ve Two-Way Binding mimarileri kullanılarak model son kullanıcıyla buluşturulmuştur.

---

## 📂 Klasör Yapısı

```text
Ramoz-AI-Optimizer/
├── app/
│   └── app.py                         # Streamlit MLOps Karar Destek Arayüzü
├── assets/                            # Grafikler, SHAP analizleri ve ekran görüntüleri
├── data/                              # .parquet formatındaki temizlenmiş veri setleri
├── src/                               # Makine Öğrenmesi Pipeline Kodları
│   ├── 01_data_preprocessing.py       # Veri temizleme ve Parquet dönüşümü
│   ├── 02_feature_engineering.py      # Resampling ve IQR filtrelemeleri
│   ├── 03_correlation_analysis.py     # Pearson korelasyon ve regresyon analizleri
│   ├── 04_model_training.py           # Random Forest modelinin eğitilmesi
│   ├── 05_shap_analysis.py            # Açıklanabilir Yapay Zeka (XAI) analizleri
│   ├── 06_predictive_maintenance.py   # Termal zorlanma ve arıza tespit algoritması
│   └── 07_optimization_engine.py      # SciPy Powell optimizasyon motoru
├── requirements.txt                   # Proje bağımlılıkları ve kütüphane sürümleri
└── README.md