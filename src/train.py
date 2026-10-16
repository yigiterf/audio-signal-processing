# src/train.py
"""
Konuşma Tanıma Modeli Eğitim Döngüsü (KTÜ Dönem Projesi)
Bu modül; PyTorch ile CNN-BiLSTM modelini CTC Loss fonksiyonu kullanarak eğitir,
GPU (CUDA) hızlandırmasını kullanır, WER/CER metriklerini takip eder ve
eğitim kayıp (Loss) grafiğini kaydeder.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

from speech_project.src.models.speech_model import SpeechRecognitionModel
from speech_project.src.dataset.tokenizer import TurkishTokenizer
from speech_project.src.dataset.speech_dataset import SpeechDataset, speech_collate_fn
from speech_project.src.decoder.ctc_decoder import CTCGreedyDecoder


def train_speech_model(
    manifest_csv: str = "speech_project/data/sample_data/train_manifest.csv",
    epochs: int = 15,
    batch_size: int = 4,
    learning_rate: float = 5e-4,
    save_model_path: str = "speech_project/models/best_model.pt",
    loss_plot_path: str = "speech_project/reports/figures/training_loss.png"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"EĞİTİM BAŞLIYOR | Cihaz: {device.type.upper()} | Toplam Epoch: {epochs}")
    print("=" * 65)

    # 1. Veriyi Oku ve DataLoader Oluştur
    df = pd.read_csv(manifest_csv)
    audio_paths = df["audio_path"].tolist()
    transcripts = df["transcript"].tolist()

    tokenizer = TurkishTokenizer()
    dataset = SpeechDataset(audio_paths, transcripts, tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=speech_collate_fn
    )

    # 2. Modeli Başlat
    model = SpeechRecognitionModel(
        num_classes=tokenizer.vocab_size,
        n_mels=80,
        projection_dim=256,
        rnn_hidden=256,
        rnn_layers=2,
        dropout=0.2
    ).to(device)

    # 3. CTC Loss ve Optimizasyon
    # blank=0 -> CTC Blank token indeksi
    # zero_infinity=True -> Sayısal taşma durumunda loss'u sıfırlayarak eğitimi çökmeden korur
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    decoder = CTCGreedyDecoder(tokenizer)

    epoch_losses: List[float] = []
    epoch_wers: List[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        all_refs: List[str] = []
        all_hyps: List[str] = []

        for batch in dataloader:
            mels, targets, input_lengths, target_lengths, batch_refs = batch
            mels = mels.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # İleri besleme: [Batch, Zaman_Çıkış, Sınıf_Sayısı]
            log_probs = model(mels)

            # CNN katmanları zaman boyutunu yarıya indirdiği için gerçek çıkış uzunlukları:
            output_lengths = model.get_output_lengths(input_lengths)

            # PyTorch CTC Loss tensör sıralaması: [Zaman, Batch, Sınıflar]
            loss = criterion(
                log_probs.permute(1, 0, 2),
                targets,
                output_lengths,
                target_lengths
            )

            # Geri yayılım (Backpropagation) ve gradyan kırpma (clipping)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()

            # Greedy Decode ile tahminleri al
            decoded = decoder.decode(log_probs.detach(), output_lengths)
            all_refs.extend(batch_refs)
            all_hyps.extend(decoded)

        avg_loss = running_loss / len(dataloader)
        metrics = decoder.compute_metrics(all_refs, all_hyps)
        avg_wer = metrics["WER"]

        epoch_losses.append(avg_loss)
        epoch_wers.append(avg_wer)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - CTC Loss: {avg_loss:.4f} | WER: {avg_wer * 100:.2f}% | Örnek Tahmin: '{all_hyps[0]}'")

    # 4. Modeli Kaydet
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": tokenizer.vocab,
        "n_mels": 80
    }, save_model_path)
    print(f"\n[BAŞARILI] En iyi model kaydedildi: {save_model_path}")

    # 5. Eğitim Kayıp Grafiğini Çiz ve Kaydet
    os.makedirs(os.path.dirname(loss_plot_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    fig.suptitle("CNN-BiLSTM-CTC Eğitim Süreci Analizi", fontsize=14, fontweight="bold")

    ax1.plot(range(1, epochs + 1), epoch_losses, marker="o", color="#d62728", linewidth=1.8)
    ax1.set_title("CTC Kayıp Değeri (Loss)", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CTC Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(range(1, epochs + 1), [w * 100 for w in epoch_wers], marker="s", color="#1f77b4", linewidth=1.8)
    ax2.set_title("Kelime Hata Oranı (WER %)", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Hata Oranı (%)")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[BAŞARILI] Eğitim kayıp grafiği kaydedildi: {loss_plot_path}")
    print("=" * 65)


if __name__ == "__main__":
    train_speech_model(epochs=10, batch_size=4)
