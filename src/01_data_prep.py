import pandas as pd
import re

def rpt_to_parquet(rpt_dosya_yolu, parquet_dosya_yolu):
    print(f"'{rpt_dosya_yolu}' dosyası okunuyor...")
    
    # 1. Dosyanın okunması
    with open(rpt_dosya_yolu, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    if len(lines) < 3:
        print("Hata: Dosya geçerli sensör verisi içermiyor.")
        return

    header_line = lines[0]
    dash_line = lines[1]
    data_lines = lines[2:]

    # 2. Düzenli ifadeler (Regex) ile sütun genişliklerinin tespiti
    col_spans = [(m.start(), m.end()) for m in re.finditer(r'-+', dash_line)]
    
    # 3. Başlıkların (Features) dinamik olarak çıkarılması
    headers = [header_line[start:end].strip() for start, end in col_spans]

    # 4. Sabit genişlikli verilerin ayrıştırılması ve NULL kontrolü
    parsed_data = []
    for line in data_lines:
        if line.strip() == '':
            continue
            
        row = [line[start:end].strip() if len(line) >= start else '' for start, end in col_spans]
        row = [None if val == 'NULL' or val == '' else val for val in row]
        parsed_data.append(row)

    # 5. Pandas DataFrame nesnesinin oluşturulması
    df = pd.DataFrame(parsed_data, columns=headers)
    print(f"Toplam {len(df)} satır ve {len(df.columns)} sütun ayrıştırıldı.")

    # 6. Veri Tiplerinin Temizlenmesi (Casting & Cleaning)
    print("Sensör verileri sayısal tiplere dönüştürülüyor...")
    for col in df.columns:
        if col == "CreateDate":
            df[col] = pd.to_datetime(df[col], errors='coerce')
        else:
            if df[col].dtype == object:
                # Virgüllü ondalıkları noktaya çevir, boşlukları temizle
                df[col] = df[col].astype(str).str.strip().str.replace(',', '.')
                df[col] = df[col].replace('None', None)
            
            # Değerleri makine öğrenmesi için float64 tipine zorla
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 7. Sıkıştırılmış Parquet formatında dışa aktarım
    df.to_parquet(parquet_dosya_yolu, engine='pyarrow')
    print(f"\nİşlem başarılı! Yapılandırılmış veriler '{parquet_dosya_yolu}' konumuna kaydedildi.")

# --- Ana Çalıştırma Bloğu ---
if __name__ == "__main__":
    GIRDI_DOSYASI = "D:/Staj/2026 yılı 2.makine.rpt"
    CIKTI_DOSYASI = "D:/Staj/makine_verileri_temiz.parquet"
    
    rpt_to_parquet(GIRDI_DOSYASI, CIKTI_DOSYASI)