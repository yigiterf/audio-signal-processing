# src/models/speech_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SpeechRecognitionModel(nn.Module):
    """
    Log-Mel spektrogramını metne dönüştüren CNN-BiLSTM-CTC Konuşma Tanıma Modeli.
    
    Boru Hattı:
    1. 2B-CNN: Spektrogramdaki yerel akustik desenleri (formantlar, patlamalar) yakalar.
    2. Projeksiyon: Frekans eksenini düzleştirip boyutunu küçültür.
    3. BiLSTM: Zaman boyunca iki yönlü bağlamı (dil dizilimini) öğrenir.
    4. Sınıflandırıcı: CTC için her zaman adımında harf olasılıklarını üretir.
    """
    def __init__(
        self,
        num_classes: int,           # Karakter sayısı (29 harf + boşluk + CTC Blank)
        n_mels: int = 80,           # Girdi frekans filtre sayısı
        cnn_channels: Tuple[int, int] = (32, 64),
        projection_dim: int = 256,  # CNN sonrası özellik boyutu
        rnn_hidden: int = 256,      # LSTM gizli durum boyutu
        rnn_layers: int = 2,        # LSTM katman sayısı
        dropout: float = 0.3
    ):
        super(SpeechRecognitionModel, self).__init__()

        self.num_classes = num_classes
        self.n_mels = n_mels

        # ==========================================
        # 1. EVRİŞİMSEL SİNİR AĞI (2D-CNN) BLOĞU
        # ==========================================
        # Girdi boyutu: [Batch, 1, 80 (Frekans), Zaman]
        self.conv_block = nn.Sequential(
            # 1. Conv Katmanı: 1 kanal -> 32 kanal
            nn.Conv2d(in_channels=1, out_channels=cnn_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(),
            # Frekansı 80'den 40'a indirir, zamanı yarıya düşürür (gürültüyü süzer)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 2. Conv Katmanı: 32 kanal -> 64 kanal
            nn.Conv2d(in_channels=cnn_channels[0], out_channels=cnn_channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels[1]),
            nn.ReLU(),
            # Frekansı 40'tan 20'ye indirir; ancak zaman eksenini KORUR (stride=(2, 1))
            # Zaman çözünürlüğünü korumak CTC hizalaması için çok kritiktir.
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )

        # CNN çıkışındaki toplam frekans ve kanal çarpımı:
        # 80 Mel -> 1. MaxPool sonrası 40 -> 2. MaxPool sonrası 20 frekans satırı kalır.
        final_mel_dim = n_mels // 4  # 80 / 4 = 20
        cnn_flat_features = cnn_channels[1] * final_mel_dim  # 64 * 20 = 1280

        # ==========================================
        # 2. DOĞRUSAL PROJEKSİYON (LINEAR PROJECTION)
        # ==========================================
        # 1280 boyutlu öznitelik vektörünü BiLSTM için 256 boyuta sıkıştırır
        self.projection = nn.Sequential(
            nn.Linear(cnn_flat_features, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ==========================================
        # 3. ÇİFT YÖNLÜ LSTM (BiLSTM) BLOĞU
        # ==========================================
        self.bilstm = nn.LSTM(
            input_size=projection_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,  # Çift yönlü: Hem geçmişe hem geleceğe bakar
            dropout=dropout if rnn_layers > 1 else 0.0
        )

        # ==========================================
        # 4. ÇIKIŞ SINIFLANDIRICI (CTC HEAD)
        # ==========================================
        # BiLSTM çift yönlü olduğu için çıktı boyutu: rnn_hidden * 2 (256 * 2 = 512)
        self.classifier = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        İleri besleme adımı.
        
        Parametre:
            x: Log-Mel tensörü [Batch, 1, Frekans (80), Zaman]
            
        Dönüş:
            log_probs: [Batch, Zaman_Adımı, Sınıf_Sayısı] formatında log-olasılıklar
        """
        # 1. CNN Katmanları
        # x: [Batch, 1, 80, Zaman] -> [Batch, 64, 20, Zaman // 2]
        x = self.conv_block(x)

        # 2. BiLSTM İçin Şekillendirme
        # Tensörü zaman eksenini öne alacak şekilde yeniden düzenleriz:
        # [Batch, Kanallar (64), Frekans (20), Zaman] -> [Batch, Zaman, Kanallar (64), Frekans (20)]
        x = x.permute(0, 3, 1, 2)
        batch_size, time_steps, channels, freqs = x.size()

        # Kanalları ve frekansları tek bir öznitelik boyutunda birleştir:
        # [Batch, Zaman, 64 * 20 = 1280]
        x = x.contiguous().view(batch_size, time_steps, channels * freqs)

        # 3. Projeksiyon Katmanı
        # [Batch, Zaman, 1280] -> [Batch, Zaman, 256]
        x = self.projection(x)

        # 4. BiLSTM Katmanı
        # [Batch, Zaman, 256] -> [Batch, Zaman, 512]
        lstm_out, _ = self.bilstm(x)

        # 5. Karar Katmanı ve LogSoftmax (CTC gereksinimi)
        # [Batch, Zaman, 512] -> [Batch, Zaman, num_classes]
        logits = self.classifier(lstm_out)
        
        # CTC Loss sayısal kararlılık için Log-Softmax bekler
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs

    def get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        CNN'deki MaxPool katmanları zaman eksenini yarıya indirdiği için,
        CTC Loss fonksiyonuna verilecek olan çıkış zaman adımı uzunluklarını hesaplar.
        """
        # 1. MaxPool: kernel=2, stride=2 -> T // 2
        # 2. MaxPool: kernel=(2,1), stride=(2,1) -> Zaman değişmez
        return torch.div(input_lengths, 2, rounding_mode="floor")


if __name__ == "__main__":
    # Test ve Tensör Boyut Doğrulaması
    num_classes = 32   # 29 harf + boşluk + <BLANK>
    n_mels = 80
    batch_size = 4
    time_frames = 200  # ~2 saniyelik sesin Log-Mel kare sayısı

    model = SpeechRecognitionModel(num_classes=num_classes, n_mels=n_mels)
    
    # Sahte bir Log-Mel tensörü oluşturalım: [Batch, 1, Mel, Zaman]
    dummy_input = torch.randn(batch_size, 1, n_mels, time_frames)
    
    # İleri besleme testi
    output = model(dummy_input)
    
    # Giriş zaman uzunlukları ve CTC için hesaplanan çıkış uzunlukları
    dummy_input_lengths = torch.tensor([200, 180, 150, 100])
    output_lengths = model.get_output_lengths(dummy_input_lengths)

    print("=" * 60)
    print("MODEL DOĞRULAMA TESTİ BAŞARILI!")
    print("=" * 60)
    print(f"Girdi Boyutu:           {list(dummy_input.shape)} -> [Batch, Kanal, Mel, Zaman]")
    print(f"Model Çıktı Boyutu:     {list(output.shape)} -> [Batch, Çıkış_Zamanı, Sınıf_Sayısı]")
    print(f"Girdi Zaman Boyutları:  {dummy_input_lengths.tolist()}")
    print(f"CTC Çıkış Boyutları:    {output_lengths.tolist()}")
    print(f"Toplam Parametre:       {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("=" * 60)
