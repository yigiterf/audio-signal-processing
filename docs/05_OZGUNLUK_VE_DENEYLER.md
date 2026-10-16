# BÖLÜM 5: ÖZGÜNLÜK VE DENEYLER REHBERİ (30 PUANLIK BÖLÜM)

Bu belge, proje föyünün en yüksek puanlı bölümlerinden biri olan **Özgünlük (30 Puan)** kriterlerinin (a, b, c, d, e) projede nasıl uygulandığını ve hocaya nasıl sunulacağını açıklar.

---

## 1. Özgünlük Kriteri (a): Öğrencinin Kendi Ses Kayıtları

Hoca, projenin sadece hazır internet verileriyle değil, öğrencinin bizzat kendi sesiyle test edilmesini şart koşmuştur.

### Nasıl Kullanılır? ([record_audio.py](file:///home/nos/Masaüstü/DERSLER/Sinyaller/speech_project/src/utils/record_audio.py))
Terminalden tek bir komutla kendi mikrofonundan ses kaydı alabilirsin:
```bash
python speech_project/src/utils/record_audio.py --text "merhaba sinyaller ve sistemler" --condition clean --id 1
```

Bu script:
1. `arecord` aracıyla 16 kHz mono formatında kaydı alır.
2. `speech_project/data/own_recordings/clean_001.wav` olarak kaydeder.
3. Otomatik olarak `manifest.csv` dosyasına cümle transkripti ve ortam etiketiyle yazar.

---

## 2. Özgünlük Kriteri (b): Farklı Ses, Ortam ve Gürültü Koşulları

Gerçek dünyada kullanıcılar sessiz bir stüdyoda değil; kafede, sokakta veya yankılı amfilerde konuşurlar. Sistemimizin gürültüye dayanıklılığını test etmek için `src/utils/noise_simulator.py` geliştirilmiştir:

1. **Temiz Ortam (Sessiz Oda):** Referans temel sinyalimizdir.
2. **Beyaz Gürültülü Ortam (Kafe/Cadde Koşulu):**  
   Belirlenen Sinyal-Gürültü Oranında ($SNR = 10\text{ dB}$) Gauss beyaz gürültüsü eklenir.  
   $$P_{\text{gürültü}} = \frac{P_{\text{sinyal}}}{10^{SNR / 10}}$$
3. **Yankılı Ortam (Oda Akustiği / Reverberation):**  
   Ses dalgaları odadaki duvarlara çarpıp gecikmeli olarak mikrofona gelir. Bu fiziksel olay, **Sinyaller ve Sistemler dersinin en temel konusu olan LTI Sistem Dürtü Yanıtı (Impulse Response) Evrişimi (Convolution)** ile modellenmiştir:
   $$y[n] = x[n] + 0.4 \cdot x[n - d_1] + 0.25 \cdot x[n - d_2] + 0.15 \cdot x[n - d_3]$$

> 📊 **Üretilen Rapor Grafiği:** [noise_comparison.png](file:///home/nos/Masaüstü/DERSLER/Sinyaller/speech_project/reports/figures/noise_comparison.png)  
> *(Temiz, 10 dB gürültülü ve yankılı ses dalgalarının zaman domenindeki farkları açıkça görülür.)*

---

## 3. Özgünlük Kriteri (c): Log-Mel Parametre Duyarlılık Analizi

Hoca, spektrogramı yalnızca ezbere çıkartmayı değil, parametrelerin etkisini incelemeyi istemiştir. `src/experiments/param_sensitivity.py` modülü bu incelemeyi yapar:

### Deney 1: Mel Filtre Sayısı ($N_{mels} = 40$ vs $80$ vs $128$)
* **40 Mel Filtresi:** Hesaplama açısından hızlıdır ancak ince formant frekanslarını kaçırabilir.
* **80 Mel Filtresi (Bizim Seçimimiz):** Konuşma tanıma literatürünün altın standardıdır. Formantları ve ses ayrıntılarını en dengeli biçimde yakalar.
* **128 Mel Filtresi:** 16 kHz örnekleme frekansı ve 512 FFT boyutunda ($N_{freqs} = 257$), yüksek frekanslarda üçgen filtrelerin tabanı FFT çözünürlüğünden daha dar kalır. Bu durum bazı filtrelerin sıfır değer almasına ve modelde aşırı öğrenme (overfitting) riskine yol açar.  
  *Bu matematiksel analiz, 80 filtrenin neden en doğru tercih olduğunu kanıtlar.*

### Deney 2: Pre-emphasis Katsayısı ($\alpha = 0.0$ vs $\alpha = 0.97$)
* $\alpha = 0.0$ (Filtresiz): Spektrogramın üst bantları karanlık ve sönüktür. "s", "f" gibi harfler kaybolur.
* $\alpha = 0.97$ (Filtreli): Üst bantlardaki enerji dengelenir, sürtünmeli ünsüzler parlar ve model harfleri kolayca ayırt eder.

> 📊 **Üretilen Rapor Grafiği:** [mel_param_analysis.png](file:///home/nos/Masaüstü/DERSLER/Sinyaller/speech_project/reports/figures/mel_param_analysis.png)

---

## 4. Özgünlük Kriteri (d & e): Model Sonuçlarının Yorumlanması ve Çıkarım

Eğittiğimiz model ile ses dosyalarını metne dökmek için:
```bash
python speech_project/src/evaluate.py --audio "ses_dosyasi.wav" --ref "dogru_metin"
```

Çıktı olarak hem tahmin edilen Türkçe metin ekrana yazılır, hem de `WER` (Kelime Hatası) ve `CER` (Karakter Hatası) oranları hesaplanır. Böylece modelin gürültülü ortamda başarısının ne kadar düştüğü sayısal olarak raporlanabilir.
