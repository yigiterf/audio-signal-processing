# src/evaluate.py
"""
Ses Tanıma Modeli Değerlendirme ve Çıkarım (Inference) Modülü
Bu modül; eğitilen en iyi modeli yükler, verilen bir ses dosyasını Log-Mel dönüşümünden
geçirip metne dönüştürür (Speech-to-Text). Referans metin verilirse WER ve CER hesaplar.
"""

import os
import torch
import argparse
from typing import Optional

from speech_project.src.models.speech_model import SpeechRecognitionModel
from speech_project.src.dataset.tokenizer import TurkishTokenizer
from speech_project.src.dsp.audio_processor import AudioProcessor
from speech_project.src.decoder.ctc_decoder import CTCGreedyDecoder


class SpeechPredictor:
    def __init__(self, checkpoint_path: str = "speech_project/models/best_model.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = TurkishTokenizer()
        self.processor = AudioProcessor(n_mels=80)
        self.decoder = CTCGreedyDecoder(self.tokenizer)

        # Modeli oluştur ve ağırlıkları yükle
        self.model = SpeechRecognitionModel(
            num_classes=self.tokenizer.vocab_size,
            n_mels=80,
            projection_dim=256,
            rnn_hidden=256,
            rnn_layers=2
        ).to(self.device)

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            print(f"[MODEL YÜKLENDİ] {checkpoint_path} başarıyla yüklendi ({self.device.type.upper()}).")
        else:
            print(f"[UYARI] {checkpoint_path} bulunamadı! Model rastgele ağırlıklarla çalışıyor.")

    def predict(self, audio_path: str, reference_text: Optional[str] = None):
        """Ses dosyasını dinleyip metne döker."""
        waveform = self.processor.load_audio(audio_path)
        mel_spec = self.processor.compute_log_mel(waveform)  # [1, 80, Zaman]
        mel_input = mel_spec.unsqueeze(0).to(self.device)    # [Batch=1, 1, 80, Zaman]

        with torch.no_grad():
            log_probs = self.model(mel_input)                # [1, T_out, 31]
            time_frames = torch.tensor([mel_spec.shape[-1]], dtype=torch.long)
            out_lens = self.model.get_output_lengths(time_frames)
            decoded_texts = self.decoder.decode(log_probs, out_lens)

        pred_text = decoded_texts[0]

        print("=" * 60)
        print(f"SES DOSYASI: {audio_path}")
        print(f"TAHMİN:      '{pred_text}'")

        if reference_text:
            metrics = self.decoder.compute_metrics([reference_text], [pred_text])
            print(f"REFERANS:    '{reference_text}'")
            print(f"WER (Kelime Hatası):   {metrics['WER'] * 100:.2f}%")
            print(f"CER (Karakter Hatası): {metrics['CER'] * 100:.2f}%")
        print("=" * 60)

        return pred_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ses Tanıma Çıkarım (Inference) Scripti")
    parser.add_argument("--audio", type=str, default="speech_project/data/sample_data/wavs/sample_001.wav", help="Ses dosyası yolu")
    parser.add_argument("--ref", type=str, default="merhaba dunya", help="Beklenen doğru metin (opsiyonel)")
    args = parser.parse_args()

    predictor = SpeechPredictor()
    predictor.predict(args.audio, args.ref)
