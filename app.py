import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# Tarayıcı otomatik çevirisini tamamen engelleyen HTML üstbilgisi
st.set_page_config(page_title="2. El Araç Piyasası Zeka Platformu", layout="wide")
st.markdown(
    """
    <html translate="no" class="notranslate">
    <head>
    <meta name="google" content="notranslate">
    <meta http-equiv="Content-Language" content="tr">
    </head>
    </html>
    <script>
        document.documentElement.setAttribute('translate', 'no');
    </script>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 translate="no">🚗 İkinci El Araç Piyasası Profesyonel Analiz Platformu</h1>', unsafe_allow_html=True)

# 1. VERİYİ YÜKLE VE TEMİZLE
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv(r'C:\Users\USER\Desktop\arac_projesi\arac_verisi.csv')
    df.columns = df.columns.str.strip().str.lower()
    
    df = df.dropna(subset=['fiyat', 'marka', 'seri', 'yil', 'kilometre'])
    
    for col in ['fiyat', 'kilometre', 'yil', 'motor_gucu', 'motor_hacmi', 'tramer', 'degisen', 'boyali']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df = df.dropna()
    df = df[(df['fiyat'] > 50000) & (df['kilometre'] >= 0) & (df['yil'] >= 1995)]
    return df

df = load_and_clean_data()

# 2. YAN MENÜ FİLTRELERİ
st.sidebar.header("🔍 Araç Filtreleme Paneli")

markalar = sorted(df['marka'].unique())
secilen_marka = st.sidebar.selectbox("Marka Seçin", markalar)

seriler = sorted(df[df['marka'] == secilen_marka]['seri'].unique())
secilen_seri = st.sidebar.selectbox("Seri Seçin", seriler)

temp_df = df[(df['marka'] == secilen_marka) & (df['seri'] == secilen_seri)]

modeller = sorted(temp_df['model'].unique())
secilen_model = st.sidebar.selectbox("Model / Paket Seçin (İsteğe Bağlı)", ['Tümü'] + list(modeller))

vites_tipleri = ['Tümü'] + sorted(list(df['vites_tipi'].unique()))
secilen_vites = st.sidebar.selectbox("Vites Tipi", vites_tipleri)

yakit_tipleri = ['Tümü'] + sorted(list(df['yakit_tipi'].unique()))
secilen_yakit = st.sidebar.selectbox("Yakıt Tipi", yakit_tipleri)

# Akıllı Filtreleme Mantığı
filtreli_df = temp_df.copy()

if secilen_model != 'Tümü':
    model_filtreli = filtreli_df[filtreli_df['model'] == secilen_model]
    if not model_filtreli.empty:
        filtreli_df = model_filtreli

if secilen_vites != 'Tümü':
    vites_filtreli = filtreli_df[filtreli_df['vites_tipi'] == secilen_vites]
    if not vites_filtreli.empty:
        filtreli_df = vites_filtreli

if secilen_yakit != 'Tümü':
    yakit_filtreli = filtreli_df[filtreli_df['yakit_tipi'] == secilen_yakit]
    if not yakit_filtreli.empty:
        filtreli_df = yakit_filtreli

# 3. ÖZET METRİK KARTLARI (KPI)
st.markdown(f"### 📊 Seçim Özeti: {secilen_marka} {secilen_seri} {f'({secilen_model})' if secilen_model != 'Tümü' else ''}")

if not filtreli_df.empty:
    toplam_ilan = len(filtreli_df)
    ortalama_fiyat = filtreli_df['fiyat'].mean()
    ortalama_km = filtreli_df['kilometre'].mean()
    medyan_fiyat = filtreli_df['fiyat'].median()
    ortalama_tramer = filtreli_df['tramer'].mean() if 'tramer' in filtreli_df.columns else 0
    
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    col_m1.metric("Toplam İlan", f"{toplam_ilan} Adet")
    col_m2.metric("Ortalama Fiyat", f"{ortalama_fiyat:,.0f} TL")
    col_m3.metric("Medyan Fiyat", f"{medyan_fiyat:,.0f} TL")
    col_m4.metric("Ortalama KM", f"{ortalama_km:,.0f} KM")
    col_m5.metric("Ort. Tramer", f"{ortalama_tramer:,.0f} TL")

    st.markdown("---")

    # 4. TEKNİK ÖZELLİKLER
    st.subheader("⚙️ Veri Setine Dayalı Ortalama Teknik Özellikler")
    t_c1, t_c2, t_c3, t_c4, t_c5 = st.columns(5)
    t_c1.metric("Ort. Motor Gücü", f"{filtreli_df['motor_gucu'].mean():.0f} HP")
    t_c2.metric("Ort. Motor Hacmi", f"{filtreli_df['motor_hacmi'].mean():.0f} cc")
    t_c3.metric("Ort. Yakıt Tüketimi", f"{filtreli_df['ortalama_yakit_tuketimi'].mean():.1f} Lt / 100km")
    t_c4.metric("Ort. Değişen Parça", f"{filtreli_df['degisen'].mean():.1f} Adet")
    t_c5.metric("Ort. Boyalı Parça", f"{filtreli_df['boyali'].mean():.1f} Adet")
    st.markdown("---")

    # 5. FIRSAT İLANLARI
    st.subheader("💡 Akıllı Fırsat Önerileri (Fiyat/Performans)")
    if toplam_ilan > 1:
        firsatlar = filtreli_df[
            (filtreli_df['fiyat'] < ortalama_fiyat) & 
            (filtreli_df['kilometre'] < ortalama_km)
        ].sort_values(by='fiyat', ascending=True).head(3)
        
        if not firsatlar.empty:
            for idx, row in firsatlar.iterrows():
                st.info(f"🔥 Fırsat İlanı: {row['model']} | {int(row['yil'])} Model | {row['kilometre']:,.0f} KM | Vites: {row['vites_tipi']} | Fiyat: {row['fiyat']:,.0f} TL")
        else:
            st.info("Bu kriterlerde şu an öne çıkan net bir fırsat ilanı bulunamadı.")
    else:
        st.info("Bu filtre kombinasyonunda tek bir ilan olduğu için fırsat kıyaslaması yapılamıyor.")

    st.markdown("---")

    # 6. GRAFİKLER
    col1, col2 = st.columns(2)

    with col1:
        fig_scatter = px.scatter(filtreli_df, x='kilometre', y='fiyat', color='yil', 
                                 title="Kilometre ve Fiyat Dağılım İlişkisi")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        yil_fiyat = filtreli_df.groupby('yil')['fiyat'].mean().reset_index()
        fig_bar = px.bar(yil_fiyat, x='yil', y='fiyat', 
                         title="Yıllara Göre Ortalama Fiyat Trendi")
        st.plotly_chart(fig_bar, use_container_width=True)

    # 7. VİTES VE YAKIT DAĞILIMI
    col3, col4 = st.columns(2)

    with col3:
        vites_dagilim = filtreli_df['vites_tipi'].value_counts().reset_index()
        vites_dagilim.columns = ['Vites Tipi', 'İlan Sayısı']
        fig_vites = px.pie(vites_dagilim, names='Vites Tipi', values='İlan Sayısı', 
                           title="Vites Türü Dağılımı", hole=0.4)
        st.plotly_chart(fig_vites, use_container_width=True)

    with col4:
        yakit_dagilim = filtreli_df['yakit_tipi'].value_counts().reset_index()
        yakit_dagilim.columns = ['Yakıt Tipi', 'İlan Sayısı']
        fig_yakit = px.bar(yakit_dagilim, x='Yakıt Tipi', y='İlan Sayısı', 
                           title="Yakıt Türü Dağılımı", color='Yakıt Tipi')
        st.plotly_chart(fig_yakit, use_container_width=True)

    # 8. YAPAY ZEKA TAHMİN MODELİ
    st.markdown("---")
    st.subheader("🤖 Gelişmiş Yapay Zeka Fiyat Tahmin Simülasyonu")

    if len(filtreli_df) > 3:
        X = filtreli_df[['kilometre', 'yil', 'motor_gucu']]
        y = filtreli_df['fiyat']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        t_col1, t_col2, t_col3 = st.columns(3)
        with t_col1:
            girilen_km = st.number_input("Kilometre", min_value=0, max_value=500000, value=int(ortalama_km), step=5000)
        with t_col2:
            girilen_yil = st.number_input("Üretim Yılı", min_value=1995, max_value=2026, value=int(filtreli_df['yil'].median()), step=1)
        with t_col3:
            girilen_guc = st.number_input("Motor Gücü (HP)", min_value=50, max_value=500, value=int(filtreli_df['motor_gucu'].mean()), step=5)
            
        if st.button("Piyasa Değerini Hesapla"):
            tahmin_input = pd.DataFrame([[girilen_km, girilen_yil, girilen_guc]], columns=['kilometre', 'yil', 'motor_gucu'])
            tahmin_fiyat = model.predict(tahmin_input)[0]
            
            min_pazar = filtreli_df['fiyat'].min()
            max_pazar = filtreli_df['fiyat'].max()
            tahmin_fiyat = max(min_pazar, min(tahmin_fiyat, max_pazar))
            
            st.success(f"🏷️ Yapay Zeka Tahmini: **{tahmin_fiyat:,.0f} TL**")
    else:
        st.info("Yapay zeka modelini eğitmek için bu filtrede en az 4 ilan gereklidir.")

    # 9. HAM VERİ TABLOSU VE İNDİRME
    st.markdown("---")
    st.subheader("📋 İlan Veri Detayları (Ham Tablo)")
    st.dataframe(filtreli_df, use_container_width=True)

    st.markdown("---")
    st.subheader("📥 Veri Dışa Aktarımı")
    csv_data = filtreli_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Filtrelenmiş İlanları CSV Olarak İndir",
        data=csv_data,
        file_name=f'{secilen_marka}_{secilen_seri}_analiz.csv',
        mime='text/csv',
    )
else:
    st.warning("Seçtiğiniz kriterlere uygun ilan bulunamadı. Lütfen sol menüden model veya vites/yakıt filtrelerini 'Tümü' olarak seçerek tekrar deneyin.")