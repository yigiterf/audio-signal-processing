# src/dataset/speech_dataset.py
"""
PyTorch Konuşma Veri Seti ve Dinamik Dolgulama (Collate) Modülü
Bu modül; farklı uzunluktaki ses dosyalarını ve metin etiketlerini yükler,
Log-Mel spektrogramına dönüştürür ve batch (küme) halinde modele verilebilmesi için
dinamik dolgulama (padding) uygular.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any, Optional

from speech_project.src.dsp.audio_processor import AudioProcessor
from speech_project.src.dataset.tokenizer import TurkishTokenizer


class SpeechDataset(Dataset):
    """
    Ses dosyası yolları ve metin transkriptlerinden oluşan PyTorch Dataset sınıfı.
    """
    def __init__(
        self,
        audio_paths: List[str],
        transcripts: List[str],
        audio_processor: Optional[AudioProcessor] = None,
        tokenizer: Optional[TurkishTokenizer] = None
    ):
        assert len(audio_paths) == len(transcripts), "Ses dosyası ve metin sayıları eşit olmalıdır!"
        self.audio_paths = audio_paths
        self.transcripts = transcripts
        self.processor = audio_processor if audio_processor is not None else AudioProcessor()
        self.tokenizer = tokenizer if tokenizer is not None else TurkishTokenizer()

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        audio_path = self.audio_paths[idx]
        text = self.transcripts[idx]

        # 1. Sesi yükle ve Log-Mel çıkar: [1, 80, Zaman]
        if os.path.exists(audio_path):
            waveform = self.processor.load_audio(audio_path)
        else:
            # Test veya dosya yoksa sentetik rastgele ses üret
            waveform = torch.randn(1, 16000 * 2)

        mel_spec = self.processor.compute_log_mel(waveform)  # [1, 80, T]

        # 2. Metni indeks tensörüne çevir
        indices = self.tokenizer.text_to_indices(text)
        target_tensor = torch.tensor(indices, dtype=torch.long)

        return mel_spec, target_tensor, text


def speech_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, str]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Dinamik Dolgulama (Dynamic Padding):
    Farklı uzunluktaki spektrogramları ve hedef metin dizilerini
    küme içindeki en uzun elemanın boyutuna kadar sıfırlarla doldurur (pad eder).

    Dönüş:
        padded_mels:     [Batch, 1, 80, T_max]
        padded_targets:  [Batch, L_max]
        input_lengths:   [Batch]  (Her spektrogramın gerçek zaman boyutu)
        target_lengths:  [Batch]  (Her metnin gerçek karakter sayısı)
        raw_transcripts: Ham metin listesi
    """
    mels, targets, raw_transcripts = zip(*batch)

    batch_size = len(mels)
    n_mels = mels[0].shape[1]

    # En uzun zaman adımını (T_max) ve en uzun hedef metin boyutunu (L_max) bul
    time_lengths = [m.shape[2] for m in mels]
    target_lengths = [len(t) for t in targets]

    max_time = max(time_lengths)
    max_target_len = max(target_lengths) if max(target_lengths) > 0 else 1

    # Doldurulmuş (padded) tensörleri oluştur
    padded_mels = torch.zeros(batch_size, 1, n_mels, max_time, dtype=torch.float32)
    padded_targets = torch.zeros(batch_size, max_target_len, dtype=torch.long)

    for i in range(batch_size):
        t_len = time_lengths[i]
        padded_mels[i, :, :, :t_len] = mels[i]

        l_len = target_lengths[i]
        if l_len > 0:
            padded_targets[i, :l_len] = targets[i]

    input_lengths_tensor = torch.tensor(time_lengths, dtype=torch.long)
    target_lengths_tensor = torch.tensor(target_lengths, dtype=torch.long)

    return (
        padded_mels,
        padded_targets,
        input_lengths_tensor,
        target_lengths_tensor,
        list(raw_transcripts)
    )


if __name__ == "__main__":
    # Test: Farklı uzunlukta 3 örnek veri simüle edelim
    sample_texts = [
        "merhaba dunya",
        "sinyaller ve sistemler",
        "ktu bilgisayar muhendisligi donem projesi"
    ]
    # Gerçek dosya olmadan sentetik test için sanal yollar veriyoruz
    sample_paths = ["dummy1.wav", "dummy2.wav", "dummy3.wav"]

    dataset = SpeechDataset(sample_paths, sample_texts)
    dataloader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        collate_fn=speech_collate_fn
    )

    for batch in dataloader:
        mels, targets, in_lens, tgt_lens, transcripts = batch
        print("=" * 60)
        print("SPEECH DATASET & COLLATE DOĞRULAMA TESTİ")
        print("=" * 60)
        print(f"Batch Spektrogram Boyutu: {list(mels.shape)} -> [Batch, Kanal, Mel, Zaman_Max]")
        print(f"Batch Hedef Boyutu:      {list(targets.shape)} -> [Batch, Hedef_Max]")
        print(f"Gerçek Ses Kareleri:      {in_lens.tolist()}")
        print(f"Gerçek Karakter Boyları:  {tgt_lens.tolist()}")
        print(f"Transkriptler:            {transcripts}")
        print("=" * 60)
        break
