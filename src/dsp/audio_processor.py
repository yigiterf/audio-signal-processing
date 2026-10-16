# src/dsp/audio_processor.py
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np

class AudioProcessor:
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,        # ~32 ms pencere boyutu
        win_length: int = 400,   # 25 ms pencere boyutu
        hop_length: int = 160,   # 10 ms atlama adımı
        n_mels: int = 80,        # 80 filtre bankası
        preemphasis_coeff: float = 0.97
    ):
        self.sample_rate = sample_rate
        self.preemphasis_coeff = preemphasis_coeff

        # Torchaudio MelSpectrogram Dönüştürücüsü
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0  # Güç spektrogramı (|STFT|^2)
        )

    def apply_preemphasis(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Pre-emphasis Filtresi: Yüksek frekansları güçlendirir.
        y[n] = x[n] - alpha * x[n-1]
        """
        if self.preemphasis_coeff <= 0:
            return waveform
        # 1. dereceden FIR filtre
        filtered = torch.cat([
            waveform[:, :1],
            waveform[:, 1:] - self.preemphasis_coeff * waveform[:, :-1]
        ], dim=-1)
        return filtered

    def load_audio(self, file_path: str) -> torch.Tensor:
        """Ses dosyasını yükler, mono yapar ve 16kHz'e dönüştürür."""
        import soundfile as sf
        data, sr = sf.read(file_path)
        
        # NumPy dizisini Torch tensörüne çevir
        waveform = torch.tensor(data, dtype=torch.float32)
        
        # Boyut düzenleme: Eğer tek boyutlu ise [1, N] yap
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2:
            # [N, Kanallar] -> [Kanallar, N]
            waveform = waveform.t()
            # Çok kanallı ise mono'ya düşür
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Örnekleme frekansı farklıysa yeniden örnekle (Resample)
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        return waveform

    def compute_log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Dalga formundan Log-Mel spektrogramı çıkarır.
        Çıktı Boyutu: [Kanal (1), n_mels (80), Zaman Çerçeveleri (T)]
        """
        # 1. Pre-emphasis
        waveform = self.apply_preemphasis(waveform)
        
        # 2. Mel Spektrogram
        mel_spec = self.mel_transform(waveform)
        
        # 3. Log Dönüşümü (Sıfıra bölünmeyi önlemek için küçük epsilon eklenir)
        log_mel = torch.log(mel_spec + 1e-6)
        
        # 4. Spektral Normalizasyon (Ortalama 0, Standart Sapma 1)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        
        return log_mel


if __name__ == "__main__":
    processor = AudioProcessor()
    
    # 2 saniyelik sentetik bir ses dalgasıyla test edelim
    dummy_audio = torch.randn(1, 16000 * 2)  # 2 saniyelik beyaz gürültü
    log_mel = processor.compute_log_mel(dummy_audio)
    
    print(f"Girdi Sinyal Boyutu:     {dummy_audio.shape}")
    print(f"Log-Mel Spektrogram:     {log_mel.shape} -> [1 kanal, {log_mel.shape[1]} frekans, {log_mel.shape[2]} zaman]")