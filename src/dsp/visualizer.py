# src/dsp/visualizer.py
"""
Ses Sinyali ve DSP Analiz Görselleştirici (KTÜ Sinyaller & Mobil Programlama Projesi)
Bu modül; ham ses sinyalini, pre-emphasis filtresinin frekans yanıtını,
FFT büyüklük spektrumunu ve 80-kanal Log-Mel spektrogramını akademik rapor
standartlarında grafikleştirir.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from speech_project.src.dsp.audio_processor import AudioProcessor


class DSPVisualizer:
    def __init__(self, processor: Optional[AudioProcessor] = None):
        self.processor = processor if processor is not None else AudioProcessor()
        # Matplotlib akademik stil ayarları
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.titlesize": 14
        })

    @staticmethod
    def generate_synthetic_speech_signal(duration: float = 1.0, sr: int = 16000) -> torch.Tensor:
        """
        Görselleştirme ve test için insan sesine benzer sentetik bir sinyal üretir:
        İlk yarı: /a/ ünlüsü (F0=150Hz temel frekans + 700Hz ve 1220Hz formantlar)
        İkinci yarı: /s/ sürtünmeli ünsüzü (4500-7000 Hz arası yüksek frekanslı gürültü)
        """
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        half_n = len(t) // 2

        # 1. Parça: Ünlü ses (/a/ benzeri harmonikler)
        t_vowel = t[:half_n]
        f0 = 150.0   # Temel ses telleri frekansı
        f1 = 700.0   # 1. Formant frekansı
        f2 = 1220.0  # 2. Formant frekansı
        vowel = (
            0.6 * np.sin(2 * np.pi * f0 * t_vowel) +
            0.4 * np.sin(2 * np.pi * 2 * f0 * t_vowel) +
            0.5 * np.sin(2 * np.pi * f1 * t_vowel) +
            0.3 * np.sin(2 * np.pi * f2 * t_vowel)
        )
        # Doğal sönümlenme penceresi (zarf)
        envelope_vowel = np.hanning(len(t_vowel))
        vowel = vowel * envelope_vowel

        # 2. Parça: Ünsüz ses (/s/ sürtünmeli yüksek frekans)
        t_cons = t[half_n:]
        noise = np.random.randn(len(t_cons))
        # Yüksek frekans modülasyonu (5500 Hz civarı)
        carrier = np.sin(2 * np.pi * 5500 * t_cons)
        consonant = 0.25 * noise * carrier * np.hanning(len(t_cons))

        signal = np.concatenate([vowel, consonant])
        # Tensör formatına çevir: [1, N]
        return torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

    def plot_full_analysis(
        self,
        waveform: torch.Tensor,
        title: str = "Ses Sinyali ve Log-Mel DSP Analizi",
        save_path: Optional[str] = None
    ):
        """
        Rapor için 4 panelli kapsamlı DSP analiz grafiği üretir:
        1. Ham Dalga Formu (Zaman Domeni)
        2. Pre-emphasis Filtresi Etkisi Karşılaştırması
        3. Frekans Domeni Büyüklük Spektrumu (FFT / dB)
        4. 80-Kanal Log-Mel Spektrogramı (Frekans & Zaman Isı Haritası)
        """
        sr = self.processor.sample_rate
        raw_wave_np = waveform.squeeze().cpu().numpy()
        time_axis = np.linspace(0, len(raw_wave_np) / sr, len(raw_wave_np))

        # Pre-emphasis uygula
        filtered_wave = self.processor.apply_preemphasis(waveform)
        filtered_wave_np = filtered_wave.squeeze().cpu().numpy()

        # Log-Mel Spektrogramı hesapla
        log_mel = self.processor.compute_log_mel(waveform).squeeze().cpu().numpy()

        fig, axs = plt.subplots(4, 1, figsize=(11, 12), constrained_layout=True)
        fig.suptitle(title, fontsize=15, fontweight="bold")

        # -------------------------------------------------------------
        # Panel 1: Ham Zaman Sinyali
        # -------------------------------------------------------------
        axs[0].plot(time_axis, raw_wave_np, color="#1f77b4", linewidth=0.9, label="Ham Sinyal x[n]")
        axs[0].set_title("1. Ham Ses Sinyali (Zaman Domeni)", fontweight="bold")
        axs[0].set_xlabel("Zaman (Saniye)")
        axs[0].set_ylabel("Genlik (Amplitüd)")
        axs[0].set_xlim([0, time_axis[-1]])
        axs[0].legend(loc="upper right")

        # -------------------------------------------------------------
        # Panel 2: Pre-emphasis Karşılaştırması (Zaman Domeni)
        # -------------------------------------------------------------
        axs[1].plot(time_axis, raw_wave_np, color="#aec7e8", alpha=0.7, linewidth=0.8, label="Ham x[n]")
        axs[1].plot(time_axis, filtered_wave_np, color="#d62728", linewidth=0.9,
                    label=rf"Pre-emphasis: $y[n] = x[n] - {self.processor.preemphasis_coeff} \cdot x[n-1]$")
        axs[1].set_title("2. Pre-emphasis Filtresi (Yüksek Frekans Dengeleme)", fontweight="bold")
        axs[1].set_xlabel("Zaman (Saniye)")
        axs[1].set_ylabel("Genlik")
        axs[1].set_xlim([0, time_axis[-1]])
        axs[1].legend(loc="upper right")

        # -------------------------------------------------------------
        # Panel 3: FFT Büyüklük Spektrumu (Frekans Domeni)
        # -------------------------------------------------------------
        n_samples = len(raw_wave_np)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sr)
        fft_raw = np.abs(np.fft.rfft(raw_wave_np))
        fft_filt = np.abs(np.fft.rfft(filtered_wave_np))

        # dB cinsine çevir: 20 * log10(genlik + eps)
        raw_db = 20 * np.log10(fft_raw + 1e-6)
        filt_db = 20 * np.log10(fft_filt + 1e-6)

        axs[2].plot(freqs, raw_db, color="#1f77b4", alpha=0.6, linewidth=0.9, label="Ham Spektrum (dB)")
        axs[2].plot(freqs, filt_db, color="#ff7f0e", linewidth=1.1, label="Pre-emphasis Sonrası Spektrum (dB)")
        axs[2].set_title("3. Frekans Domeni Spektrumu (FFT Büyüklüğü)", fontweight="bold")
        axs[2].set_xlabel("Frekans (Hz)")
        axs[2].set_ylabel("Güç (dB)")
        axs[2].set_xlim([0, sr / 2])
        axs[2].axvline(x=4000, color="gray", linestyle="--", alpha=0.5, label="Yüksek Frekans Bölgesi (>4kHz)")
        axs[2].legend(loc="upper right")

        # -------------------------------------------------------------
        # Panel 4: 80-Kanal Log-Mel Spektrogramı
        # -------------------------------------------------------------
        # log_mel boyutu: [80, Zaman_Kareleri]
        im = axs[3].imshow(
            log_mel,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            extent=[0, time_axis[-1], 0, self.processor.mel_transform.n_mels]
        )
        axs[3].set_title(f"4. Log-Mel Spektrogramı (80 Mel Filtre Bankası - Model Girdisi)", fontweight="bold")
        axs[3].set_xlabel("Zaman (Saniye)")
        axs[3].set_ylabel("Mel Filtre Kanalı (0 - 80)")
        cbar = fig.colorbar(im, ax=axs[3], orientation="vertical", pad=0.01)
        cbar.set_label("Normalize Enerji (dB)", fontsize=9)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[BAŞARILI] DSP analiz grafiği kaydedildi: {save_path}")

        plt.close(fig)
        return save_path


if __name__ == "__main__":
    visualizer = DSPVisualizer()
    print("[BİLGİ] Örnek sentetik insan sesi sinyali üretiliyor...")
    synth_wave = visualizer.generate_synthetic_speech_signal(duration=1.2, sr=16000)

    output_fig_path = "speech_project/reports/figures/dsp_analysis.png"
    visualizer.plot_full_analysis(
        waveform=synth_wave,
        title="KTÜ Ses Analizi ve DSP Boru Hattı Doğrulaması",
        save_path=output_fig_path
    )
