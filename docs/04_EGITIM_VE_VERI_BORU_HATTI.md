# BÖLÜM 4: EĞİTİM, VERİ BORU HATTI VE KÜTÜPHANELER

Bu belge; ses verilerinin modele nasıl beslendiğini, değişken uzunluklu seslerin nasıl işlendiğini, PyTorch eğitim döngüsünün adımlarını ve projede kullanılan donanım/kütüphane altyapısını açıklar.

---

## 1. Ses Kayıtlarını Neden Yüklüyoruz? Ne Kadar Veri Gerekir?

### Ses Kayıtlarının Amacı Nedir?
Derin öğrenme modelleri "boş bir beyin" gibidir. Başlangıçta Türkçe harflerin veya ses frekanslarının hiçbir anlamını bilmezler.  
Modeli eğitmek için ona binlerce çift örnek veririz:
$$\text{Ses Dalgası (WAV)} \longleftrightarrow \text{Doğru Metin (Transkript)}$$

Model, bu çiftleri defalarca inceleyerek hangi frekans kombinasyonunun "m", hangisinin "e", hangisinin "r" harfine karşılık geldiğini öğrenir.

### Ne Kadar Ses Kaydına İhtiyacımız Var?
Konuşma tanımada veri miktarı ihtiyacı hedefe göre belirlenir:
1. **Endüstriyel Genel ASR (Siri, Google Asistan, Whisper):**  
   Herhangi birinin herhangi bir konuda söylediği her şeyi anlamak için **500 - 10.000+ saat** konuşma verisi gerekir.
2. **Dönem Projesi / Belirli Komut Seti (Bizim Senaryomuz):**  
   Amacımız mimarinin çalıştığını, ses işleme (DSP) mantığını, gürültü ve ortam dayanıklılığını hocaya göstermektir:
   * **50 - 200 adet ses kaydı** (toplam 10–20 dakikalık konuşma, 20-30 farklı cümle kalıbı) modelin belirli cümleleri ve harf dizilimlerini kusursuz kavraması için yeterlidir.
   * Kendi sesimizden ekleyeceğimiz **15-20 kayıt** ise projenin özgünlük puanını (30 puan) garantiye alır.

---

## 2. Kullanılan Kütüphaneler ve Donanım Birimleri

Eğitimi ve tüm sistemi çalıştıran teknoloji yığını şu şekildedir:

### A. Donanım Birimi (Hardware Unit)
* **NVIDIA GPU (CUDA):** Bilgisayarında bulunan NVIDIA ekran kartı devreye sokulmuştur (`CUDA available: True`).
* *Neden CPU değil de GPU?*  
  Modeldeki 3 milyon parametrenin matris çarpımları CPU üzerinde saatler sürerken, GPU'nun binlerce paralel çekirdeği sayesinde eğitim **saniyeler içinde** tamamlanır.

### B. Yazılım Kütüphaneleri
| Kütüphane | Rolü ve Görevi |
|---|---|
| **PyTorch (`torch`)** | Model katmanlarını (Conv2d, BiLSTM, Linear), tensör tensör matris işlemlerini ve geri yayılımı (backpropagation) yöneten ana yapay zeka çatısı. |
| **`torch.nn.CTCLoss`** | Ses ve metin arasındaki zaman hizalamasını dinamik programlama ile çözen özel kayıp fonksiyonu. |
| **`soundfile`** | Ses dosyalarını (WAV) diske bağımsız ve en yüksek hızda okuyan/yazan ses motoru. |
| **`torch.optim.AdamW`** | Model ağırlıklarını güncelleyen, ağırlık sönümlemeli (weight decay) modern optimizasyon algoritması. |
| **`jiwer`** | Modelin yazıya döktüğü metin ile asıl metin arasındaki Levenshtein mesafesini hesaplayıp WER ve CER hata oranlarını çıkaran metrik kütüphanesi. |
| **`matplotlib`** | Akademik rapor grafikleri (spektrogram, dalga formu, kayıp eğrileri) üreten görselleştirme aracı. |

---

## 3. Değişken Uzunluklu Sesler ve Dinamik Dolgulama (Collate Function)

Konuşma verisindeki en kritik teknik zorluk şudur:
* 1. ses dosyası 1.2 saniyedir ($T_1 = 120$ kare).
* 2. ses dosyası 3.5 saniyedir ($T_2 = 350$ kare).
* Hedef cümlelerin de harf sayıları birbirinden tamamen farklıdır.

PyTorch'ta modeller verileri kümeler (**Batch**, örneğin 4'erli grup) halinde işler. Ancak bir kümedeki tüm tensörlerin matris boyutlarının eşit olması şarttır!

### Çözüm: `speech_collate_fn` ([speech_dataset.py](file:///home/nos/Masaüstü/DERSLER/Sinyaller/speech_project/src/dataset/speech_dataset.py))
1. Kümedeki en uzun sesi ($T_{max}$) ve en uzun metni ($L_{max}$) bulur.
2. Diğer kısa seslerin sonuna sıfırlar ekler (**Padding**).
3. Kısa metinlerin sonuna `<BLANK>` (0) token'ları ekler.
4. Her bir sesin gerçek orijinal uzunluğunu (`input_lengths` ve `target_lengths`) CTC Loss fonksiyonuna bildirmek üzere kaydeder.

```python
# speech_dataset.py içerisindeki mantık:
padded_mels = torch.zeros(batch_size, 1, 80, max_time)
padded_targets = torch.zeros(batch_size, max_target_len, dtype=torch.long)
```

---

## 4. PyTorch Eğitim Döngüsü Adım Adım ([train.py](file:///home/nos/Masaüstü/DERSLER/Sinyaller/speech_project/src/train.py))

Her bir eğitim turunda (Epoch) arka planda gerçekleşen 6 temel adım:

```python
# 1. Gradyanları Sıfırla (Önceki adımdan kalan türevleri temizler)
optimizer.zero_grad()

# 2. İleri Besleme (Forward Pass): Model ses tensörünü alır, harf olasılıklarını üretir
log_probs = model(mels)

# 3. Gerçek Zaman Uzunluklarını Hesapla (CNN zamanı yarıya indirir)
output_lengths = model.get_output_lengths(input_lengths)

# 4. CTC Kaybını (Loss) Hesapla
loss = criterion(log_probs.permute(1, 0, 2), targets, output_lengths, target_lengths)

# 5. Geri Yayılım (Backpropagation): Loss'un tüm parametrelere göre türevini al
loss.backward()

# Gradyan Patlamasını Önle (Gradient Clipping - LSTM kararlılığı için zorunludur)
nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

# 6. Ağırlıkları Güncelle (Model biraz daha öğrenir)
optimizer.step()
```

Eğitim bittiğinde en iyi ağırlıklar `speech_project/models/best_model.pt` dosyasına kaydedilir ve artık model tek başına sesleri metne dökmeye hazır hale gelir!
