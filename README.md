# Log-Mel Tabanlı Ses Analizi ve CNN-BiLSTM-CTC ile Metne Dönüştürme
## KTÜ Bilgisayar Mühendisliği Dönem Projesi

Bu proje; ham ses sinyallerini **Sayısal Sinyal İşleme (DSP)** teknikleriyle analiz ederek **80-Kanal Log-Mel Spektrogramına** dönüştüren ve **2D-CNN + BiLSTM + CTC (Connectionist Temporal Classification)** derin öğrenme mimarisi ile Türkçe konuşmayı metne dönüştüren uçtan uca bir Konuşma Tanıma (ASR) sistemidir.

---

## 📚 Detaylı Öğrenim ve Kod Dokümantasyonu

Projenin her bir parçasını derinlemesine öğrenmek ve kodları satır satır incelemek için hazırlanan özel alt rehberler:

1. **[Bölüm 1: Ses Sinyalleri ve Sayısal Sinyal İşleme (DSP)](speech_project/docs/01_SES_VE_DSP.md)**
   - Ham ses dalgası, örnekleme (16kHz), Pre-emphasis filtresi, STFT ve 80-Kanal Log-Mel spektrogramı.
2. **[Bölüm 2: Yapay Zeka Mimarisi (CNN-BiLSTM-CTC)](speech_project/docs/02_YAPAY_ZEKA_MIMARISI.md)**
   - 2D-CNN, zaman boyutunu koruyan MaxPool, Doğrusal Projeksiyon, BiLSTM ve CTC sınıflandırıcı katmanları.
3. **[Bölüm 3: CTC Algoritması ve Türkçe Tokenizer](speech_project/docs/03_CTC_VE_TOKENIZER.md)**
   - Zaman uyumsuzluğu problemi, `<BLANK>` token'ının mantığı, Greedy Kod Çözücü, WER ve CER metrikleri.
4. **[Bölüm 4: Model Eğitimi, Veri Boru Hattı ve Kütüphaneler](speech_project/docs/04_EGITIM_VE_VERI_BORU_HATTI.md)**
   - Dinamik dolgulama (Collate Function), PyTorch CUDA eğitim döngüsü, donanım ve kütüphane altyapısı.
5. **[Bölüm 5: Özgünlük Kriterleri ve Deney Rehberi (30 Puan)](speech_project/docs/05_OZGUNLUK_VE_DENEYLER.md)**
   - Kendi ses kayıtlarımız, beyaz gürültü ve LTI oda yankısı simülasyonları, Mel parametre analizi.
6. **[Akademik Ön Rapor Taslağı (KTÜ Formatı)](speech_project/reports/ON_RAPOR.md)**
   - Use-case, component çizenekleri, proje takvimi ve IEEE referanslarıyla tam metin rapor.

---

## 📁 Proje Dizin Yapısı

```bash
Sinyaller/
├── README.md                          # Proje genel rehberi ve dokümantasyonu
└── speech_project/
    ├── requirements.txt               # Gerekli Python kütüphaneleri
    ├── venv/                          # PyTorch + CUDA destekli sanal ortam
    ├── models/
    │   └── best_model.pt              # Eğitilmiş PyTorch model ağırlıkları
    ├── reports/
    │   ├── ON_RAPOR.md                # KTÜ formatına tam uyumlu Ön Rapor taslağı
    │   └── figures/                   # Akademik rapor analiz grafikleri
    │       ├── dsp_analysis.png       # 4 panelli kapsamlı DSP analiz grafiği
    │       ├── noise_comparison.png   # Temiz vs. 10dB Gürültü vs. Yankı karşılaştırması
    │       ├── mel_param_analysis.png # Mel filtre sayısı ve Pre-emphasis analizi
    │       └── training_loss.png      # CTC Loss ve WER eğitim eğrileri
    ├── data/
    │   ├── sample_data/               # Hızlı doğrulama sentetik ses veri seti
    │   └── own_recordings/            # Öğrencinin mikrofondan kaydettiği sesler
    └── src/
        ├── dsp/
        │   ├── audio_processor.py     # Pre-emphasis, STFT, Log-Mel spektrogramı
        │   └── visualizer.py          # Dalga formu, FFT ve spektrogram görselleştirici
        ├── dataset/
        │   ├── tokenizer.py           # 31 sınıflı Türkçe karakter tokenizer'ı
        │   └── speech_dataset.py      # PyTorch Dataset ve dinamik padding (collate_fn)
        ├── models/
        │   └── speech_model.py        # 2D-CNN + Projeksiyon + BiLSTM + CTC Mimarisi
        ├── decoder/
        │   └── ctc_decoder.py         # Greedy CTC Decoder ve WER/CER metrikleri
        ├── utils/
        │   ├── noise_simulator.py     # Beyaz gürültü (SNR) ve LTI Yankı simülatörü
        │   ├── record_audio.py        # Mikrofondan ses kaydetme ve etiketleme aracı
        │   └── generate_mock_dataset.py
        ├── experiments/
        │   └── param_sensitivity.py  # Mel ve Pre-emphasis duyarlılık deneyleri
        ├── train.py                   # PyTorch CUDA model eğitim döngüsü
        └── evaluate.py                # Ses dosyasından metin tahmini (Inference)
```

---

## 🚀 Hızlı Başlangıç ve Kullanım

### 1. Sanal Ortamı Aktif Etme
```bash
source speech_project/venv/bin/activate
```

### 2. DSP ve Sinyal Analiz Grafiğini Üretme
Ham ses sinyali, Pre-emphasis filtresi, FFT spektrumu ve 80-Kanal Log-Mel spektrogramını oluşturmak için:
```bash
PYTHONPATH=. python speech_project/src/dsp/visualizer.py
```
*Çıktı grafiği:* `speech_project/reports/figures/dsp_analysis.png`

### 3. Akustik Gürültü ve Yankı Karşılaştırması
Sessiz oda, 10 dB kafe gürültüsü ve kapalı oda yankısı simülasyonu için:
```bash
PYTHONPATH=. python speech_project/src/utils/noise_simulator.py
```
*Çıktı grafiği:* `speech_project/reports/figures/noise_comparison.png`

### 4. Log-Mel Parametre Duyarlılık Deneyi (Özgünlük Analizi)
Farklı Mel filtre sayıları (40, 80, 128) ve Pre-emphasis etkisini karşılaştırmak için:
```bash
PYTHONPATH=. python speech_project/src/experiments/param_sensitivity.py
```
*Çıktı grafiği:* `speech_project/reports/figures/mel_param_analysis.png`

### 5. Mikrofondan Kendi Sesini Kaydetme
Özgünlük kriteri için kendi sesinizi kaydetmek:
```bash
# Temiz ortam kaydı
python speech_project/src/utils/record_audio.py --text "merhaba sinyaller ve sistemler" --condition clean --id 1

# Gürültülü ortam kaydı
python speech_project/src/utils/record_audio.py --text "merhaba sinyaller ve sistemler" --condition noisy --id 2
```

### 6. Modeli Eğitme (CUDA GPU Destekli)
```bash
PYTHONPATH=. python speech_project/src/train.py
```
*Eğitim bittiğinde model ağırlıkları `speech_project/models/best_model.pt` ve kayıp grafiği `speech_project/reports/figures/training_loss.png` olarak kaydedilir.*

### 7. Ses Dosyasını Metne Dönüştürme (Inference)
```bash
PYTHONPATH=. python speech_project/src/evaluate.py --audio "speech_project/data/sample_data/wavs/sample_001.wav" --ref "merhaba dunya"
```

---

## 📊 Teorik Temeller

### Pre-emphasis Filtresi
İnsan sesinde yüksek frekanslardaki zayıflamayı dengelemek için 1. dereceden yüksek geçiren FIR filtre kullanılır:
$$y[n] = x[n] - 0.97 \cdot x[n-1]$$

### Log-Mel Spektrogramı
1. **STFT**: 25 ms pencere uzunluğu ($win\_length=400$), 10 ms atlama ($hop\_length=160$) ile pencerelenip Fourier dönüşümü alınır.
2. **Mel Skalası**: Doğrusal frekanslar insan kulağının logaritmik duyma eşiğine uygun 80 üçgen filtreye izdüşürülür.
3. **Log Dönüşümü**: Enerji desibel (dB) ölçeğine dönüştürülüp ortalaması 0, standart sapması 1 olacak şekilde normalize edilir.

### CTC (Connectionist Temporal Classification)
Ses sinyalinin uzunluğu ile metin uzunluğu arasındaki zaman farkını otomatik çözer. Model her zaman adımında 31 sınıftan (29 harf + boşluk + `<BLANK>`) birini tahmin eder. Ardışık yinelenenler tekilleştirilip blank token'lar silinerek nihai metin elde edilir.

---

## 📄 Akademik Ön Rapor
KTÜ yönergesine uygun olarak hazırlanan tam metin ön rapor taslağına **[ON_RAPOR.md](file:///home/nos/Masaüstü/DERSLER/Sinyaller/speech_project/reports/ON_RAPOR.md)** dosyasından ulaşabilirsiniz.
