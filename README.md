# Türkçe Konuşmayı Metne Dönüştürme (Log-Mel + CNN-BiLSTM-CTC)

Bu projede mikrofondan alınan Türkçe konuşmayı yazıya dökmeye çalışıyorum. Önce sesi
sinyal işleme adımlarından geçirip Log-Mel spektrogramına çeviriyorum, sonra bu
spektrogramı CNN-BiLSTM tabanlı bir modele verip CTC ile harf dizisine dönüştürüyorum.

## Nasıl çalışıyor?

1. **Ses okuma:** WAV dosyası 16 kHz mono olarak okunuyor.
2. **Pre-emphasis:** `y[n] = x[n] - 0.97·x[n-1]` ile s, ş, f gibi kısık ünsüzleri biraz güçlendiriyorum.
3. **Log-Mel spektrogram:** 25 ms pencere, 10 ms kaydırma, 80 Mel filtresi. Sonuçta ses 80 satırlık bir "resme" dönüşüyor.
4. **Model:** 2 katmanlı CNN spektrogramdaki desenleri yakalıyor, 2 katmanlı BiLSTM zaman içindeki bağlamı öğreniyor. Toplamda yaklaşık 3M parametre var.
5. **CTC:** Model her adımda 31 sınıftan (29 harf + boşluk + blank) birini tahmin ediyor. Tekrarlar birleştirilip blank'ler silinince metin ortaya çıkıyor.
6. **Değerlendirme:** WER ve CER ile ölçüyorum. Türkçe eklemeli bir dil olduğu için CER bence daha anlamlı bir ölçüt.

## Klasörler

```
src/
  dsp/           -> ses okuma, pre-emphasis, Log-Mel, grafikler
  dataset/       -> Türkçe tokenizer ve PyTorch dataset
  models/        -> CNN-BiLSTM-CTC modeli
  decoder/       -> greedy CTC decoder, WER/CER
  utils/         -> ses kaydı, gürültü/yankı simülasyonu
  experiments/   -> Mel parametre denemeleri
  train.py       -> eğitim
  evaluate.py    -> ses dosyasından tahmin
data/            -> örnek veri ve kendi kayıtlarım
reports/         -> ön rapor ve grafikler
models/          -> eğitilmiş model
```

## Kurulum

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Kullanım

Tüm komutları proje klasöründen çalıştırıyorum:

```bash
# Eğitim
python src/train.py

# Bir ses dosyasını metne çevirme
python src/evaluate.py --audio data/sample_data/wavs/sample_001.wav --ref "merhaba dunya"

# Kendi sesimi kaydetme (clean / noisy / reverb)
python src/utils/record_audio.py --text "merhaba sinyaller ve sistemler" --condition clean --id 1

# Grafikler
python src/dsp/visualizer.py
python src/utils/noise_simulator.py
python src/experiments/param_sensitivity.py
```

## Denemeler

- **Ortam karşılaştırması:** Aynı sesi temiz, 10 dB gürültülü ve yankılı hallerde karşılaştırdım. Yankıyı gecikmeli kopyaları toplayarak, yani evrişimle simüle ettim.
- **Mel filtre sayısı:** 40, 80 ve 128'i denedim. 128'de bazı filtreler boş kaldı. 80 en dengeli sonucu verdi.
- **Pre-emphasis:** Açık ve kapalı hallerini karşılaştırdım. Açıkken yüksek frekanslar spektrogramda belirgin şekilde daha görünür oluyor.

Grafikler `reports/figures/` klasöründe.
