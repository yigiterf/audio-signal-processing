# src/experiments/param_sensitivity.py
"""
Log-Mel Parametre Duyarlılık Analizi (KTÜ Sinyaller & Sistemler Özgünlük Kriteri c)
Bu modül; Mel filtre sayısı (40 vs 80 vs 128) ve Pre-emphasis katsayısının (0.0 vs 0.97)
spektrogram üzerindeki etkisini niceliksel ve görsel olarak inceler.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from speech_project.src.dsp.audio_processor import AudioProcessor
from speech_project.src.dsp.visualizer import DSPVisualizer


def run_parameter_sensitivity_study(save_path: Optional[str] = None):
    visualizer = DSPVisualizer()
    waveform = visualizer.generate_synthetic_speech_signal(duration=1.0, sr=16000)

    # 1. Farklı Mel filtre sayılarını test et (40, 80, 128)
    mel_configs = [40, 80, 128]
    processors = [AudioProcessor(n_mels=n) for n in mel_configs]
    specs = [p.compute_log_mel(waveform).squeeze().cpu().numpy() for p in processors]

    # 2. Pre-emphasis açık (0.97) vs kapalı (0.0) karşılaştırması
    proc_no_pre = AudioProcessor(n_mels=80, preemphasis_coeff=0.0)
    proc_with_pre = AudioProcessor(n_mels=80, preemphasis_coeff=0.97)

    spec_no_pre = proc_no_pre.compute_log_mel(waveform).squeeze().cpu().numpy()
    spec_with_pre = proc_with_pre.compute_log_mel(waveform).squeeze().cpu().numpy()

    # Grafik oluştur
    fig, axs = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    fig.suptitle("Log-Mel Spektrogram Parametre Duyarlılık Analizi", fontsize=15, fontweight="bold")

    # Satır 1: Mel = 40 ve Mel = 80
    im0 = axs[0, 0].imshow(specs[0], origin="lower", aspect="auto", cmap="viridis")
    axs[0, 0].set_title("a) 40 Mel Filtresi (Düşük Frekans Çözünürlüğü - Hızlı)", fontweight="bold")
    axs[0, 0].set_ylabel("Mel Kanalı")
    fig.colorbar(im0, ax=axs[0, 0], pad=0.01)

    im1 = axs[0, 1].imshow(specs[1], origin="lower", aspect="auto", cmap="viridis")
    axs[0, 1].set_title("b) 80 Mel Filtresi (Standart & Önerilen Model Konfigürasyonu)", fontweight="bold")
    axs[0, 1].set_ylabel("Mel Kanalı")
    fig.colorbar(im1, ax=axs[0, 1], pad=0.01)

    # Satır 2: Mel = 128 ve Pre-emphasis Kapalı (alpha=0.0)
    im2 = axs[1, 0].imshow(specs[2], origin="lower", aspect="auto", cmap="viridis")
    axs[1, 0].set_title("c) 128 Mel Filtresi (Yüksek Boyut - Aşırı Öğrenme Riski)", fontweight="bold")
    axs[1, 0].set_ylabel("Mel Kanalı")
    fig.colorbar(im2, ax=axs[1, 0], pad=0.01)

    im3 = axs[1, 1].imshow(spec_no_pre, origin="lower", aspect="auto", cmap="magma")
    axs[1, 1].set_title("d) Pre-emphasis Kapalı (alpha = 0.0 - Yüksek Frekanslar Zayıf)", fontweight="bold")
    axs[1, 1].set_ylabel("Mel Kanalı")
    fig.colorbar(im3, ax=axs[1, 1], pad=0.01)

    # Satır 3: Pre-emphasis Açık (alpha=0.97) ve Fark Spektrogramı
    im4 = axs[2, 0].imshow(spec_with_pre, origin="lower", aspect="auto", cmap="magma")
    axs[2, 0].set_title("e) Pre-emphasis Açık (alpha = 0.97 - Dengelenmiş Spektrum)", fontweight="bold")
    axs[2, 0].set_ylabel("Mel Kanalı")
    axs[2, 0].set_xlabel("Zaman Çerçeveleri")
    fig.colorbar(im4, ax=axs[2, 0], pad=0.01)

    diff_spec = spec_with_pre - spec_no_pre
    im5 = axs[2, 1].imshow(diff_spec, origin="lower", aspect="auto", cmap="coolwarm")
    axs[2, 1].set_title("f) Pre-emphasis Fark Spektrogramı (Üst Bantlardaki Kazanç)", fontweight="bold")
    axs[2, 1].set_ylabel("Mel Kanalı")
    axs[2, 1].set_xlabel("Zaman Çerçeveleri")
    fig.colorbar(im5, ax=axs[2, 1], pad=0.01)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[BAŞARILI] Log-Mel parametre duyarlılık grafiği kaydedildi: {save_path}")

    plt.close(fig)


if __name__ == "__main__":
    out_file = "speech_project/reports/figures/mel_param_analysis.png"
    run_parameter_sensitivity_study(save_path=out_file)
