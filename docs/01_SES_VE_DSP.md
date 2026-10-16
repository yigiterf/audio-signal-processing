# BÖLÜM 1: SES SİNYALLERİ VE SAYISAL SİNYAL İŞLEME (DSP)

Bu belge, konuşma tanıma sistemimizin ilk adımı olan **Sayısal Sinyal İşleme (DSP - Digital Signal Processing)** aşamasını hem fiziksel/teorik temelleriyle hem de Python kodlarıyla açıklar.

---

## 1. Ses Sinyali Nedir? Bilgisayar Sesi Nasıl Görür?

Ses, havadaki moleküllerin titreşimiyle yayılan bir basınç dalgasıdır. Mikrofonun zarı bu titreşimleri elektrik gerilimine (voltaj) dönüştürür. 

Bilgisayar sürekli (analog) bir sinyali doğrudan saklayamaz. Bu yüzden sinyali belirli aralıklarla ölçüp sayısallaştırırız (**Örnekleme - Sampling**):
* **Örnekleme Frekansı ($f_s = 16.000\text{ Hz}$):** Saniyede 16.000 adet genlik (amplitüd) ölçümü alınır. Nyquist kuralına göre bu sayede $8.000\text{ Hz}$'e kadar olan tüm insan sesi frekanslarını kusursuz yakalayabiliriz.
* **Zaman Domeni:** 2 saniyelik bir ses dosyası, Python'da uzunluğu $16.000 \times 2 = 32.000$ olan bir sayı dizisidir (1B Tensör).

> ⚠️ **Problem:** Zamana bağlı bu sayılar ("dalga formu") bize sesin şiddetini gösterir ama sesin içinde o an hangi harfin (hangi frekansların) olduğunu doğrudan söyleyemez. Bu yüzden frekans analizine geçeriz.

---

## 2. Pre-emphasis Filtresi (Yüksek Frekans Dengeleme)

İnsan sesi üretilirken ses tellerinden çıkan hava akımı (glottal darbe) nedeniyle **yüksek frekanslar her oktavda yaklaşık 6 dB zayıflar**.
* "a", "o" gibi ünlü harfler düşük frekanslıdır ve çok güçlüdür.
* "s", "ş", "f", "t" gibi sürtünmeli ünsüzler ise yüksek frekanslıdır (4000–8000 Hz) ancak enerjileri çok düşüktür.

Eğer ham sesi doğrudan yapay zekaya verirsek, model güçlü ünlüleri çok iyi öğrenirken fısıltılı ünsüzleri gözden kaçırır.

### Matematiksel Modeli (1. Dereceden Yüksek Geçiren FIR Filtre)
$$y[n] = x[n] - \alpha \cdot x[n-1]$$
* Projemizde $\alpha = 0.97$ seçilmiştir.
* **Fiziksel Anlamı:** Bir önceki örnek ile şimdiki örnek arasındaki *farkı* alır. Yavaş değişen (düşük frekanslı) sesler birbirini götürürken, hızla değişen (yüksek frekanslı) sesler güçlenir (+15–20 dB kazanç sağlanır).

```python
# audio_processor.py içerisindeki kod:
def apply_preemphasis(self, waveform: torch.Tensor) -> torch.Tensor:
    if self.preemphasis_coeff <= 0:
        return waveform
    filtered = torch.cat([
        waveform[:, :1],
        waveform[:, 1:] - self.preemphasis_coeff * waveform[:, :-1]
    ], dim=-1)
    return filtered
```

---

## 3. STFT (Kısa Zamanlı Fourier Dönüşümü)

Fourier Dönüşümü (FFT), bir sinyalin hangi frekanslardan oluştuğunu bulur. Ancak klasik FFT tüm sinyali tek bir frekans grafiğine döker; bu durumda **"Hangi ses NE ZAMAN söylendi?"** bilgisi kaybolur.

Konuşma zamanla değiştiği için **STFT (Kısa Zamanlı Fourier Dönüşümü)** uygularız:
1. Sesi **25 milisaniyelik (400 örnek)** küçük pencerelere (dilimlere) böleriz (`win_length = 400`).
2. Pencere kenarlarındaki ani kesintileri engellemek için **Hann Penceresi** ile çarparız.
3. Her pencereyi **10 milisaniye (160 örnek)** sağa kaydırırız (`hop_length = 160`). Pencereler birbirinin üzerine %60 biner (overlap).
4. Her minik dilime FFT uygulayarak o anki frekans gücünü hesaplarız ($|STFT|^2$).

---

## 4. Mel Skalası ve 80-Kanal Mel Filtre Bankası

İnsan kulağının duyma sistemi doğrusal (lineer) değildir:
* $100\text{ Hz}$ ile $200\text{ Hz}$ arasındaki ses farkını çok net ayırt ederiz.
* Fakat $7.000\text{ Hz}$ ile $7.100\text{ Hz}$ arasındaki $100\text{ Hz}$'lik farkı neredeyse hiç duyamayız.

Bu nedenle doğrusal Hertz (Hz) frekanslarını, insan kulağının algısına göre esneten **Mel Skalasına** dönüştürürüz:
$$m = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

### Mel Filtre Bankası Nedir?
Frekans eksenini düşük frekanslarda sık, yüksek frekanslarda ise geniş aralıklı **80 adet üçgen filtreye** böleriz. Bu filtreler toplanarak her zaman adımı için 80 adet enerji değeri üretir.

---

## 5. Log Dönüşümü ve Normalizasyon

İnsan kulağı ses şiddetini de doğrusal değil, logaritmik (desibel / dB) olarak algılar.
1. **Log Dönüşümü:**  
   $$\text{Log-Mel} = \log(\text{Mel} + 10^{-6})$$  
   *(Sıfıra bölünmeyi ve $\log(0) = -\infty$ hatasını engellemek için küçük bir $10^{-6}$ epsilon sayısı eklenir.)*
2. **Spektral Normalizasyon:**  
   Matrisin ortalamasını 0, standart sapmasını 1 yaparız:
   $$\text{Log-Mel}_{norm} = \frac{\text{Log-Mel} - \mu}{\sigma + 10^{-6}}$$  
   Bu sayede mikrofona çok yakın veya uzaktan konuşulmasından kaynaklanan ses seviyesi farkları dengelenir; yapay zeka sadece sesin formuna odaklanır.

---

## 6. Kodun Çıktısı: Tensör Boyutu Ne Anlama Geliyor?

`AudioProcessor` sınıfımız 2 saniyelik bir sesi işlediğinde:
* **Girdi:** `[1, 32000]` (1 kanal, 32.000 zaman örneği)
* **Çıktı (Log-Mel Tensörü):** `[1, 80, 201]`
  * **1:** Kanal sayısı (Tek kanal - Mono)
  * **80:** Frekans Mel filtre sayısı (Resmin dikey ekseni / yükseklik)
  * **201:** Zaman çerçevesi sayısı (Resmin yatay ekseni / genişlik)

Ses artık derin öğrenme modellerinin çok iyi anladığı **2 boyutlu bir akustik görüntüye (Spectrogram Heatmap)** dönüşmüştür!
