# VitalStream Kullanıcı Kılavuzu

## İçindekiler

1. [Giriş](#giriş)
2. [Sistem Gereksinimleri](#sistem-gereksinimleri)
3. [Başlangıç](#başlangıç)
4. [Hasta İzleme](#hasta-izleme)
5. [Alarm Yönetimi](#alarm-yönetimi)
6. [Rapor Oluşturma](#rapor-oluşturma)
7. [Ayarlar](#ayarlar)
8. [Sorun Giderme](#sorun-giderme)

---

## Giriş

VitalStream, hastanelerde gerçek zamanlı hasta izleme için tasarlanmış profesyonel bir klinik izleme sistemidir. Bu kılavuz, sistemin temel özelliklerini ve kullanımını açıklamaktadır.

### Temel Özellikler

- **Gerçek Zamanlı İzleme**: EKG, SpO2, kan basıncı ve diğer vital bulgular
- **Çoklu Hasta İzleme**: Aynı anda 16+ hasta izleme
- **Akıllı Alarm Sistemi**: Önceliklendirilmiş alarm yönetimi
- **Gelişmiş Analiz**: AI/ML tabanlı aritmı tespiti
- **Kapsamlı Raporlama**: PDF ve CSV formatında raporlar

---

## Sistem Gereksinimleri

### Donanım Gereksinimleri

- **İşlemci**: Intel Core i5 veya üzeri
- **RAM**: Minimum 8 GB (16 GB önerilir)
- **Ekran**: 1920x1080 çözünürlük (Full HD)
- **Ağ**: 100 Mbps Ethernet bağlantısı

### Yazılım Gereksinimleri

- **İşletim Sistemi**: Windows 10/11, macOS 11+, Linux (Ubuntu 20.04+)
- **Tarayıcı**: Chrome 90+, Firefox 88+, Safari 14+
- **Ağ**: HTTPS desteği

---

## Başlangıç

### Giriş Yapma

1. Tarayıcınızda `https://vitalstream.hospital.com` adresine gidin
2. Kullanıcı adı ve şifrenizi girin
3. İki faktörlü kimlik doğrulama kodunu girin (etkinse)
4. "Giriş Yap" butonuna tıklayın

### Ana Ekran

Giriş yaptıktan sonra ana kontrol panelini göreceksiniz:

- **Sol Panel**: Hasta listesi
- **Orta Panel**: Seçili hastanın vital bulguları
- **Sağ Panel**: Alarm bildirimleri
- **Üst Menü**: Ayarlar, raporlar, yardım

---

## Hasta İzleme

### Hasta Seçme

1. Sol paneldeki hasta listesinden bir hasta seçin
2. Hasta kartına tıklayın
3. Hastanın detaylı bilgileri orta panelde görüntülenir

### Vital Bulgular

Her hasta için aşağıdaki vital bulgular izlenir:

- **Kalp Hızı (HR)**: Dakikadaki atım sayısı
- **SpO2**: Oksijen satürasyonu (%)
- **Kan Basıncı**: Sistolik/Diastolik (mmHg)
- **Solunum Hızı**: Dakikadaki solunum sayısı
- **Vücut Sıcaklığı**: Derece (°C)

### Dalga Formları

Gerçek zamanlı dalga formlarını görüntülemek için:

1. Hasta detay ekranında "Dalga Formları" sekmesine tıklayın
2. EKG, SpO2 pletismografi ve diğer dalga formları görüntülenir
3. Yakınlaştırma için fare tekerleğini kullanın
4. Dondurma için "Freeze" butonuna basın

### Trend Grafikleri

1. "Trendler" sekmesine tıklayın
2. Zaman aralığını seçin (1 saat, 6 saat, 24 saat, 72 saat)
3. Görüntülemek istediğin vital bulguları seçin
4. Grafik otomatik olarak güncellenir

---

## Alarm Yönetimi

### Alarm Seviyeleri

- **🔴 KRİTİK**: Acil müdahale gerektirir
- **🟠 YÜKSEK**: Hızlı değerlendirme gerektirir
- **🟡 ORTA**: Dikkat gerektirir
- **🟢 DÜŞÜK**: Bilgilendirme amaçlı

### Alarm Onaylama

1. Alarm bildirimi göründüğünde
2. "Onayla" butonuna tıklayın
3. Gerekirse not ekleyin
4. "Kaydet" butonuna basın

### Alarm Ayarları

Hasta bazında alarm limitlerini ayarlamak için:

1. Hasta detay ekranında "Ayarlar" ikonuna tıklayın
2. "Alarm Limitleri" sekmesini seçin
3. Her vital bulgu için min/max değerleri girin
4. "Kaydet" butonuna basın

---

## Rapor Oluşturma

### PDF Raporu

1. Hasta detay ekranında "Rapor" butonuna tıklayın
2. Rapor türünü seçin:
   - 24 Saatlik Özet
   - Haftalık Rapor
   - Özel Tarih Aralığı
3. "PDF Oluştur" butonuna basın
4. Rapor indirilir

### Veri Dışa Aktarma

1. "Dışa Aktar" butonuna tıklayın
2. Format seçin (CSV, Excel, JSON)
3. Tarih aralığını belirleyin
4. "İndir" butonuna basın

---

## Ayarlar

### Kullanıcı Tercihleri

- **Tema**: Açık/Koyu mod
- **Dil**: Türkçe/İngilizce
- **Bildirimler**: Sesli/Sessiz
- **Ekran Düzeni**: Grid/Liste görünümü

### Sistem Ayarları (Yönetici)

- **Kullanıcı Yönetimi**: Kullanıcı ekleme/çıkarma
- **Cihaz Yapılandırması**: Monitör bağlantıları
- **Yedekleme**: Otomatik yedekleme ayarları
- **Güvenlik**: Şifre politikaları

---

## Sorun Giderme

### Sık Karşılaşılan Sorunlar

#### Bağlantı Kopması

**Sorun**: Sistem bağlantısı kesildi

**Çözüm**:
1. İnternet bağlantınızı kontrol edin
2. Sayfayı yenileyin (F5)
3. Sorun devam ederse IT desteğine başvurun

#### Dalga Formu Görünmüyor

**Sorun**: EKG dalga formu ekranda görünmüyor

**Çözüm**:
1. Cihaz bağlantısını kontrol edin
2. Elektrotların doğru takıldığından emin olun
3. "Yenile" butonuna basın

#### Alarm Çalmıyor

**Sorun**: Alarmlar sesli çalmıyor

**Çözüm**:
1. Ses ayarlarını kontrol edin
2. Tarayıcı bildirim izinlerini kontrol edin
3. Sistem sesinin açık olduğundan emin olun

### Destek İletişim

- **Teknik Destek**: support@vitalstream.com
- **Telefon**: +90 (212) 555-0100
- **Acil Durum**: 7/24 destek hattı

---

## Ek Kaynaklar

- [Video Eğitimler](https://vitalstream.com/training)
- [SSS](https://vitalstream.com/faq)
- [API Dokümantasyonu](https://vitalstream.com/api-docs)
- [Topluluk Forumu](https://community.vitalstream.com)

---

**Versiyon**: 1.0  
**Son Güncelleme**: 3 Ocak 2026  
**Telif Hakkı**: © 2026 VitalStream. Tüm hakları saklıdır.
