# BÖLÜM 3: CTC (CONNECTIONIST TEMPORAL CLASSIFICATION) VE TOKENIZER

Bu belge, ses tanıma literatürünün en büyük devrimlerinden biri olan **CTC Algoritmasını**, Türkçe diline özel tasarlanmış **Tokenizer** sınıfımızı ve **Greedy Kod Çözücüyü (Decoder)** açıklar.

---

## 1. Konuşma Tanımanın En Büyük Problemi: Zaman Uyumsuzluğu

Geleneksel sınıflandırma problemlerinde (örneğin kedi/köpek resimleri):
* 1 resim verirsiniz $\rightarrow$ 1 etiket ("kedi") alırsınız.

Ancak seste durum çok farklıdır:
* 2 saniyelik bir ses dosyasında modelimiz **100 zaman adımı** üretir.
* Bu 2 saniyede söylenen kelime ise "merhaba"dır (sadece **7 harf**!).
* Kimisi "merhaba" derken "e" harfini 0.1 saniyede söyler, kimisi "meeeerhaba" diyerek 0.8 saniye uzatır.
* Her harfin ses dosyasının tam hangi milisaniyesinde başladığını ve bittiğini elle etiketlemek (alignment) imkansızdır.

> ❓ **Soru:** 100 zaman adımı veren bir model çıktısı ile 7 harflik bir metin nasıl eğitilir ve eşleştirilir?  
> 👉 **Cevap:** **CTC (Connectionist Temporal Classification)** algoritması ile!

---

## 2. CTC Algoritması ve `<BLANK>` (Boşluk) Token'ının Büyüsü

Alex Graves tarafından 2006 yılında geliştirilen CTC, alfabe setimize özel bir **Boşluk / Sessizlik (`<BLANK>`)** sembolü ekler (indeksi daima 0'dır).

### CTC Sıkıştırma Kuralları (Collapsing Rule):
1. **Ardışık Tekrarları Sıkıştır:** Model aynı harfi art arda üretiyorsa bunu tek harfe indirger.  
   `[m, m, m, e, e, r]` $\rightarrow$ `[m, e, r]`
2. **`<BLANK>` Token'ları Temizle:** İndeksi 0 olan tüm blank sembollerini atar.  
   `[<BLANK>, m, <BLANK>, e]` $\rightarrow$ `[m, e]`
3. **Peki Yan Yana İki Aynı Harf Nasıl Yazılır? (Örn: "elli", "saat"):**  
   İşte `<BLANK>` token'ının asıl dehası buradadır! Eğer iki aynı harf arasına bir `<BLANK>` girerse, sıkıştırma kuralı bozulmaz ve her iki harf de korunur:  
   * `[l, l, l]` $\rightarrow$ Tekilleşir: `[l]`
   * `[l, l, <BLANK>, l, l]` $\rightarrow$ `[l, <BLANK>, l]` $\rightarrow$ Blank silinir: `[l, l]` ("elli" yazılabilir!)

---

## 3. Türkçe Tokenizer ([tokenizer.py](file:///home/nos/Masaüstü/DERSLER/Sinyaller/speech_project/src/dataset/tokenizer.py))

Model harfleri doğrudan anlayamaz; sadece sayısal indeksleri (tensörleri) anlayabilir. `TurkishTokenizer` sınıfımız harfler ile sayılar arasındaki köprüdür.

### Alfabemiz (31 Sınıf):
| İndeks | Karakter | Anlamı |
|---|---|---|
| **0** | `<BLANK>` | CTC Hizalama ve Boşluk Token'ı |
| **1** | `' '` | Kelimeler Arası Boşluk Karakteri |
| **2 - 30** | `a, b, c, ç, ..., z` | 29 Küçük Türkçe Harf |

### Türkçe Dilinin Özel Durumu (İ / I Problemi):
Python'daki standart `.lower()` fonksiyonu İngilizce kurallara göre çalıştığı için büyük `I` harfini `i` yapar. Bu durum "Isparta" kelimesinin "isparta" olmasına yol açarak Türkçeyi bozar.

Tokenizer'ımız bu sorunu özel olarak çözer:
```python
# tokenizer.py içerisindeki temizleme kuralı:
def clean_text(self, text: str) -> str:
    text = text.replace("İ", "i").replace("I", "ı")
    text = text.lower()
    # Noktalama işaretlerini ve sayıları temizle, sadece alfabe karakterlerini koru
    ...
```

---

## 4. CTC Greedy Decoder ([ctc_decoder.py](file:///home/nos/Masaüstü/DERSLER/Sinyaller/speech_project/src/decoder/ctc_decoder.py))

Model ileri besleme yaptığında bize her zaman adımında 31 sınıfın log-olasılıklarını verir:
`[Batch, Zaman_Adımları, 31]`

Greedy (Açgözlü) Kod Çözücü şu 4 adımla metni çıkarır:
1. **Argmax:** Her zaman adımında en yüksek olasılığa sahip indeksi seç: $\hat{y}_t = \arg\max P(y_t)$.
2. **Tekilleştirme:** Ardışık yinelenen indeksleri tekilleştir (`[17, 17, 17] -> [17]`).
3. **Blank Eleme:** İndeksi 0 olan `<BLANK>` token'larını diziden çıkar.
4. **Harfe Çevirme:** Kalan sayıları `TurkishTokenizer.indices_to_text()` ile harflere dönüştür.

---

## 5. Başarım Ölçüm Metrikleri (WER ve CER)

Akademik konuşma tanımada başarı "Doğruluk Yüzdesi (Accuracy)" ile değil, **Hata Oranları** ile ölçülür:

### 1. WER (Word Error Rate - Kelime Hata Oranı)
$$WER = \frac{S + D + I}{N}$$
* **S (Substitution):** Yanlış yazılan kelime sayısı (örn: "ev" yerine "el")
* **D (Deletion):** Modelin atladığı/yazmadığı kelime sayısı
* **I (Insertion):** Modelin fazladan uydurduğu kelime sayısı
* **N:** Asıl referans cümledeki toplam kelime sayısı

### 2. CER (Character Error Rate - Karakter Hata Oranı)
Türkçe sondan eklemeli bir dildir. Örneğin referans "kitaplarımızdan" iken model "kitaplarımızda" yazarsa, WER bunu %100 hatalı kelime sayar. Ancak harf düzeyinde 15 harften 14'ü doğrudur.  
**CER**, karakterler arasındaki Levenshtein mesafesini hesaplayarak modelin fonetik başarısını çok daha hassas gösterir.
