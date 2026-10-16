# src/utils/noise_simulator.py
"""
Gürültü ve Ortam Simülatörü (KTÜ Sinyaller ve Sistemler Özgünlük Analizi)
Bu modül; temiz ses sinyallerine beyaz gürültü (Gaussian), renkli arka plan gürültüsü
ve oda yankısı (Reverberation - LTI Evrişim) ekleyerek modelin farklı akustik
ortamlardaki dayanıklılığını (robustness) test eder.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class NoiseSimulator:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def add_white_noise(self, waveform: torch.Tensor, snr_db: float = 10.0) -> torch.Tensor:
        """
        İstenen SNR (Sinyal-Gürültü Oranı) seviyesinde Beyaz Gauss Gürültüsü ekler.
        SNR_dB = 10 * log10(P_signal / P_noise)
        """
        # Sinyal gücü: P_signal = E[x^2]
        signal_power = torch.mean(waveform ** 2)
        if signal_power == 0:
            return waveform

        # Hedef gürültü gücü
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise_std = torch.sqrt(noise_power)

        # Gauss gürültüsü üret ve ekle
        noise = torch.randn_like(waveform) * noise_std
        noisy_waveform = waveform + noise

        # Kırpma (Clipping) önleme
        max_val = torch.max(torch.abs(noisy_waveform))
        if max_val > 1.0:
            noisy_waveform = noisy_waveform / max_val

        return noisy_waveform

    def add_reverberation(
        self,
        waveform: torch.Tensor,
        delays_ms: Tuple[int, ...] = (30, 60, 90),
        decay_factors: Tuple[float, ...] = (0.4, 0.25, 0.15)
    ) -> torch.Tensor:
        """
        Oda akustiği / Yankı simülasyonu:
        Yankı, LTI (Doğrusal Zamanla Değişmeyen) bir sistemin dürtü yanıtı (Impulse Response)
        ile ses sinyalinin evrişimidir (Convolution).
        y[n] = x[n] + a1*x[n - d1] + a2*x[n - d2] + ...
        """
        reverb = waveform.clone()
        num_samples = waveform.shape[-1]

        for delay_ms, decay in zip(delays_ms, decay_factors):
            delay_samples = int((delay_ms / 1000.0) * self.sample_rate)
            if delay_samples < num_samples:
                # Gecikmeli sinyali ekle
                reverb[:, delay_samples:] += decay * waveform[:, :num_samples - delay_samples]

        # Normalizasyon
        max_val = torch.max(torch.abs(reverb))
        if max_val > 1.0:
            reverb = reverb / max_val

        return reverb

    def plot_comparison(
        self,
        clean_waveform: torch.Tensor,
        noisy_waveform: torch.Tensor,
        reverb_waveform: torch.Tensor,
        save_path: Optional[str] = None
    ):
        """
        Temiz, gürültülü ve yankılı ses sinyallerini karşılaştıran rapor grafiği oluşturur.
        """
        sr = self.sample_rate
        clean_np = clean_waveform.squeeze().cpu().numpy()
        noisy_np = noisy_waveform.squeeze().cpu().numpy()
        reverb_np = reverb_waveform.squeeze().cpu().numpy()

        t = np.linspace(0, len(clean_np) / sr, len(clean_np))

        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
        fig.suptitle("Farklı Ortam ve Akustik Koşulların Sinyal Analizi", fontsize=14, fontweight="bold")

        # 1. Temiz Ses
        axs[0].plot(t, clean_np, color="#1f77b4", linewidth=0.8)
        axs[0].set_title("1. Temiz Ses Sinyali (Sessiz Ortam)", fontweight="bold")
        axs[0].set_ylabel("Genlik")
        axs[0].grid(True, linestyle="--", alpha=0.5)

        # 2. Gürültülü Ses (10 dB SNR)
        axs[1].plot(t, noisy_np, color="#d62728", linewidth=0.8)
        axs[1].set_title("2. Beyaz Gürültülü Ses (SNR = 10 dB - Kafe/Sokak Koşulu)", fontweight="bold")
        axs[1].set_ylabel("Genlik")
        axs[1].grid(True, linestyle="--", alpha=0.5)

        # 3. Yankılı Ses (Reverberation)
        axs[2].plot(t, reverb_np, color="#2ca02c", linewidth=0.8)
        axs[2].set_title("3. Yankılı Ses (Kapalı Salon / Oda Akustiği)", fontweight="bold")
        axs[2].set_xlabel("Zaman (Saniye)")
        axs[2].set_ylabel("Genlik")
        axs[2].grid(True, linestyle="--", alpha=0.5)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[BAŞARILI] Akustik ortam karşılaştırma grafiği kaydedildi: {save_path}")

        plt.close(fig)


if __name__ == "__main__":
    from speech_project.src.dsp.visualizer import DSPVisualizer

    visualizer = DSPVisualizer()
    clean_audio = visualizer.generate_synthetic_speech_signal(duration=1.0, sr=16000)

    sim = NoiseSimulator(sample_rate=16000)
    noisy_audio = sim.add_white_noise(clean_audio, snr_db=10.0)
    reverb_audio = sim.add_reverberation(clean_audio)

    save_file = "speech_project/reports/figures/noise_comparison.png"
    sim.plot_comparison(clean_audio, noisy_audio, reverb_audio, save_path=save_file)
