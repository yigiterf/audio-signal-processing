# src/utils/generate_mock_dataset.py
"""
Hızlı Test ve Başlangıç Veri Seti Üreticisi
Bu modül; modeli ve eğitim boru hattını doğrulamak için 20 adet örnek Türkçe ses
ve transkript üretir, CSV manifest dosyasını hazırlar.
"""

import os
import torch
import torchaudio
import pandas as pd
from speech_project.src.dsp.visualizer import DSPVisualizer


SAMPLE_SENTENCES = [
    "merhaba dunya",
    "sinyaller ve sistemler",
    "ktu bilgisayar muhendisligi",
    "ses analizi ve derin ogrenme",
    "konusma tanima sistemi",
    "yapay zeka modeli",
    "sayisal sinyal isleme",
    "fourier donusumu ve spektrogram",
    "mobil programlama dersi",
    "donem projesi calismasi",
    "turkce konusma verisi",
    "akustik ozellik cikarimi",
    "derin ogrenme egitimi",
    "mikrofon ses kaydi",
    "gurultulu ortam testi",
    "sessiz oda kayitlari",
    "frekans bandi analizi",
    "karakter tabanli model",
    "baglamsal dil ogrenimi",
    "test ve gecerleme asamalari"
]


def generate_sample_dataset(output_dir: str = "speech_project/data/sample_data"):
    wav_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    visualizer = DSPVisualizer()

    records = []
    print(f"[BAŞLADI] {len(SAMPLE_SENTENCES)} adet örnek ses üretiliyor...")

    for i, text in enumerate(SAMPLE_SENTENCES):
        wav_filename = f"sample_{i+1:03d}.wav"
        wav_path = os.path.join(wav_dir, wav_filename)

        # 1.5 - 2.5 saniyelik sentetik ses üret
        duration = 1.2 + (len(text) * 0.05)
        waveform = visualizer.generate_synthetic_speech_signal(duration=duration, sr=16000)

        # WAV olarak kaydet (soundfile kullanarak)
        import soundfile as sf
        sf.write(wav_path, waveform.squeeze().cpu().numpy(), 16000)

        records.append({
            "audio_path": wav_path,
            "transcript": text,
            "duration": round(duration, 2),
            "speaker": "sentetik_ornek"
        })

    manifest_csv = os.path.join(output_dir, "train_manifest.csv")
    df = pd.DataFrame(records)
    df.to_csv(manifest_csv, index=False)

    print(f"[TAMAMLANDI] Örnek veri seti hazırlandı: {manifest_csv}")
    print(f"Toplam dosya sayısı: {len(df)}")
    return manifest_csv


if __name__ == "__main__":
    generate_sample_dataset()
