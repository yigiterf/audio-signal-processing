# src/utils/record_audio.py
"""
Öğrenci Kendi Ses Kayıt Aracı (KTÜ Ses Analizi Özgünlük Kriteri a)
Bu script; mikrofon üzerinden kullanıcının kendi sesini 16 kHz mono WAV formatında
kaydeder ve otomatik olarak manifest.csv dosyasına etiketler.
Farklı ortamlar (sessiz, kafe gürültüsü, yankılı) için etiketleme desteği sunar.
"""

import os
import sys
import time
import subprocess
import pandas as pd
import argparse


def record_voice(
    output_wav_path: str,
    duration_sec: float = 3.0,
    sample_rate: int = 16000
) -> bool:
    """Linux ALSA 'arecord' kullanarak mikrofondan ses kaydeder."""
    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
    print(f"\n[DİKKAT] Kayıt {duration_sec} saniye sürecek. Konuşmaya başlayabilirsiniz...")
    print("3... 2... 1... KAYIT BAŞLADI!")

    cmd = [
        "arecord",
        "-d", str(int(duration_sec)),
        "-f", "S16_LE",
        "-r", str(sample_rate),
        "-c", "1",
        output_wav_path
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"[TAMAMLANDI] Ses başarıyla kaydedildi: {output_wav_path}\n")
        return True
    except FileNotFoundError:
        print("[HATA] 'arecord' komutu bulunamadı. Lütfen ALSA paketini kontrol edin.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[HATA] Kayıt sırasında hata oluştu: {e}")
        return False


def add_to_manifest(
    manifest_csv: str,
    audio_path: str,
    transcript: str,
    condition: str = "clean",
    speaker: str = "ogrenci"
):
    """Kayıt bilgilerini manifest.csv dosyasına ekler."""
    record = {
        "audio_path": audio_path,
        "transcript": transcript,
        "condition": condition,
        "speaker": speaker
    }
    df_new = pd.DataFrame([record])

    if os.path.exists(manifest_csv):
        df_old = pd.read_csv(manifest_csv)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    os.makedirs(os.path.dirname(manifest_csv), exist_ok=True)
    df_all.to_csv(manifest_csv, index=False)
    print(f"[MANİFEST GÜNCELLENDİ] Toplam kayıt sayısı: {len(df_all)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kendi Sesini Kaydetme ve Etiketleme Aracı")
    parser.add_argument("--text", type=str, default="merhaba sinyaller ve sistemler dersi", help="Okunacak metin")
    parser.add_argument("--duration", type=float, default=3.5, help="Kayıt süresi (saniye)")
    parser.add_argument("--condition", type=str, default="clean", choices=["clean", "noisy", "reverb"], help="Kayıt ortamı")
    parser.add_argument("--id", type=int, default=1, help="Kayıt ID numarası")

    args = parser.parse_args()

    recordings_dir = "speech_project/data/own_recordings"
    filename = f"{args.condition}_{args.id:03d}.wav"
    out_path = os.path.join(recordings_dir, filename)
    manifest_path = os.path.join(recordings_dir, "manifest.csv")

    print("=" * 60)
    print("KTÜ SES KAYIT MODÜLÜ")
    print(f"Okunacak Cümle: '{args.text}'")
    print(f"Ortam Koşulu:   {args.condition}")
    print(f"Hedef Dosya:    {out_path}")
    print("=" * 60)

    # CLI üzerinden çalıştırıldığında kaydı al
    success = record_voice(out_path, duration_sec=args.duration)
    if success:
        add_to_manifest(manifest_path, out_path, args.text, condition=args.condition)
