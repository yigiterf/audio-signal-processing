# BÖLÜM 2: YAPAY ZEKA MİMARİSİ (CNN-BiLSTM-CTC)

Bu belge, Log-Mel spektrogramını alıp Türkçe harf olasılıklarına dönüştüren **2.99 Milyon Parametreli** hibrit yapay zeka mimarimizin (`SpeechRecognitionModel`) iç mekanizmasını açıklar.

---

## 1. Mimarinin Kuşbakışı Özeti

Modelimiz 4 ana bloğun birleşimidir:

```
[Girdi: Log-Mel Spektrogramı] (Boyut: [Batch, 1, 80 Mel, Zaman T])
             │
             ▼
[1. Blok: 2D-Evrişimsel Sinir Ağı (2D-CNN)]
   -> Akustik biçimleri, formantları ve yerel ses desenlerini süzme
             │
             ▼
[2. Blok: Doğrusal Projeksiyon (Linear Projection)]
   -> 1280 boyutlu CNN özniteliğini 256 boyuta sıkıştırma
             │
             ▼
[3. Blok: Çift Yönlü LSTM (BiLSTM)]
   -> Zaman boyunca konuşmanın geçmiş ve gelecek bağlamını öğrenme
             │
             ▼
[4. Blok: CTC Sınıflandırıcı Kafa (Log-Softmax)]
   -> Her zaman adımında 31 Türkçe sınıfın olasılık dağılımı
```

---

## 2. Blok 1: 2D-Evrişimsel Sinir Ağı (CNN)

Spektrogram iki boyutlu bir ısı haritası (resim) olduğu için görüntü işlemede kullanılan 2B Evrişim katmanları ses için de mükemmel çalışır.

### Katman Detayları:
1. **1. Conv Katmanı:**  
   - Girdi: 1 kanal $\rightarrow$ Çıktı: 32 kanal. $3 \times 3$ filtrelerle temel akustik çizgileri yakalar.
   - **BatchNorm2d + ReLU:** Eğitimi hızlandırır ve doğrusal olmayan özellikleri öğrenir.
   - **MaxPool2d(kernel=2, stride=2):** Frekansı 80'den 40'a, zamanı $T$'den $T/2$'ye indirir. Arka plan gürültüsünü süzer.

2. **2. Conv Katmanı:**  
   - Girdi: 32 kanal $\rightarrow$ Çıktı: 64 kanal. $3 \times 3$ filtrelerle karmaşık ses formasyonlarını yakalar.
   - **Kritik Tasarım Kararı - MaxPool2d(kernel=(2, 1), stride=(2, 1)):**  
     Buradaki havuzlama frekans eksenini 40'tan 20'ye indirirken, **zaman eksenine hiç dokunmaz (stride=1)**!  
     *Neden?* Çünkü konuşma tanımada zaman çözünürlüğü aşırı küçültülürse kısa süren harfler (örneğin patlamalı "p", "t", "b" harfleri) tamamen kaybolur.

```python
# speech_model.py içerisindeki kod:
final_mel_dim = n_mels // 4                       # 80 / 4 = 20
cnn_flat_features = cnn_channels[1] * final_mel_dim  # 64 * 20 = 1280
```

---

## 3. Blok 2: Doğrusal Projeksiyon (Linear Layer)

CNN'den çıkan tensörün boyutu: `[Batch, 64 Kanal, 20 Mel, Zaman/2]` şeklindedir.
1. Tensörü zaman ekseni başa gelecek şekilde yeniden dizeriz (`permute`): `[Batch, Zaman/2, 64, 20]`.
2. Kanallar ile frekansı birleştiririz: $64 \times 20 = 1280$ boyutlu bir öznitelik vektörü elde ederiz.
3. **Linear(1280, 256) + Dropout(0.3):** Bu 1280 boyutu 256'ya indirir. Bu projeksiyon, LSTM'in öğrenmesi gereken parametre yükünü hafifletir ve aşırı öğrenmeyi (overfitting) engeller.

---

## 4. Blok 3: Çift Yönlü LSTM (BiLSTM)

İnsan konuşması anlık değil, bağlamsal bir süreçtir. Türkçede seslerin telaffuzu kendisinden önce ve sonra gelen harflerden etkilenir (**Eşsöyleyiş - Coarticulation**).  
Örneğin: "gel" derken çıkan "g" sesi ile "gül" derken çıkan "g" sesinin frekansları farklıdır.

### Neden BiLSTM (Bidirectional)?
* **İleri Yönlü (Forward LSTM):** Konuşmanın başından sonuna doğru okur ("Geçmişte ne söylendi?").
* **Geri Yönlü (Backward LSTM):** Konuşmanın sonundan başına doğru okur ("Gelecekte hangi ses gelecek?").
* İki yönün çıktıları birleştirilir: $256 + 256 = 512$ boyutlu zengin bir bağlam vektörü oluşur.

```python
self.bilstm = nn.LSTM(
    input_size=projection_dim,  # 256
    hidden_size=rnn_hidden,     # 256
    num_layers=rnn_layers,      # 2 katmanlı
    batch_first=True,
    bidirectional=True,         # Çift yönlü
    dropout=0.3
)
```

---

## 5. Blok 4: CTC Çıkış Katmanı (Classifier)

BiLSTM'in her zaman adımında ürettiği 512 boyutlu vektör, tam bağlantılı bir katmana verilir:
* **Linear(512, 31):** 31 adet sınıfımızın (29 Türkçe harf + boşluk + `<BLANK>`) ham skorlarını (logit) üretir.
* **Log-Softmax:** Logit değerlerini olasılıkların logaritmasına çevirir:
  $$\log(\text{Softmax}(z_i)) = \log\left(\frac{e^{z_i}}{\sum_j e^{z_j}}\right)$$

CTC Loss fonksiyonu, sayısal kararlılık (taşmaları önlemek) için doğrudan Log-Softmax değerleriyle çalışır.

---

## 6. Model Parametre İstatistiği

Modelimizde toplam **2,992,992 (yaklaşık 3 Milyon)** eğitilebilir parametre vardır:
* **2D-CNN Katmanları:** ~20 Bin parametre
* **Projeksiyon Katmanı:** ~330 Bin parametre
* **2-Katmanlı BiLSTM:** ~2.6 Milyon parametre (Modelin ana beyni)
* **CTC Sınıflandırıcı:** ~16 Bin parametre

Bu boyut, bilgisayarındaki NVIDIA GPU üzerinde saniyeler içinde eğitilebilecek kadar hafif, ancak Türkçe ses yapısını mükemmel kavrayacak kadar güçlüdür.
