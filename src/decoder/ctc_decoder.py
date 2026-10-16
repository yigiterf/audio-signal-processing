# src/decoder/ctc_decoder.py
"""
CTC Greedy Decoder ve Metrik Hesaplayıcı (KTÜ Ses Analizi Projesi)
Bu modül; modelin çıkardığı Log-Softmax olasılık tensörünü anlamlı Türkçe metne
dönüştürür (Greedy CTC Decoding) ve WER (Kelime Hata Oranı) / CER (Karakter Hata Oranı)
metriklerini hesaplar.
"""

import torch
import jiwer
from typing import List, Dict, Optional
from speech_project.src.dataset.tokenizer import TurkishTokenizer


class CTCGreedyDecoder:
    def __init__(self, tokenizer: Optional[TurkishTokenizer] = None):
        self.tokenizer = tokenizer if tokenizer is not None else TurkishTokenizer()
        self.blank_idx = 0  # <BLANK> token indeksi daima 0'dır

    def decode(
        self,
        log_probs: torch.Tensor,
        output_lengths: Optional[torch.Tensor] = None
    ) -> List[str]:
        """
        Greedy (Açgözlü) CTC Kod Çözme:
        1. Her zaman adımında en yüksek olasılığa sahip indeksi seç (Argmax).
        2. Art arda gelen aynı indeksleri tekilleştir ([m, m, m] -> [m]).
        3. <BLANK> (0) token'larını sil.
        4. Kalan indeksleri harfe dönüştür.

        Parametreler:
            log_probs: [Batch, Zaman_Adımları, Sınıf_Sayısı] tensörü
            output_lengths: Her örneğin geçerli zaman adımı uzunlukları [Batch]
        Dönüş:
            Tahmin edilen metinler listesi (Örn: ["merhaba dunya", "gunaydin"])
        """
        # Argmax: En yüksek log-olasılığa sahip sınıfları bul -> [Batch, Zaman]
        predictions = torch.argmax(log_probs, dim=-1).cpu().numpy()
        batch_size = predictions.shape[0]

        decoded_transcripts = []

        for b in range(batch_size):
            valid_len = (
                int(output_lengths[b].item())
                if output_lengths is not None
                else predictions.shape[1]
            )
            seq = predictions[b, :valid_len]

            # 1 & 2. Ardışık yinelenenleri sıkıştır
            collapsed = []
            prev_token = None
            for token in seq:
                if token != prev_token:
                    collapsed.append(token)
                    prev_token = token

            # 3. Blank (0) token'ları filtrele
            final_indices = [idx for idx in collapsed if idx != self.blank_idx]

            # 4. İndeksleri metne çevir
            text = self.tokenizer.indices_to_text(final_indices)
            decoded_transcripts.append(text)

        return decoded_transcripts

    def compute_metrics(
        self,
        references: List[str],
        hypotheses: List[str]
    ) -> Dict[str, float]:
        """
        Akademik başarım metrikleri:
        WER (Word Error Rate - Kelime Hata Oranı)
        CER (Character Error Rate - Karakter Hata Oranı)
        """
        cleaned_refs = [self.tokenizer.clean_text(r) for r in references]
        cleaned_hyps = [self.tokenizer.clean_text(h) for h in hypotheses]

        # Boş tahmin veya referans kontrolü
        valid_refs = []
        valid_hyps = []
        for r, h in zip(cleaned_refs, cleaned_hyps):
            if len(r) == 0:
                r = " "
            if len(h) == 0:
                h = " "
            valid_refs.append(r)
            valid_hyps.append(h)

        wer_score = jiwer.wer(valid_refs, valid_hyps)
        cer_score = jiwer.cer(valid_refs, valid_hyps)

        return {
            "WER": float(wer_score),
            "CER": float(cer_score)
        }


if __name__ == "__main__":
    tokenizer = TurkishTokenizer()
    decoder = CTCGreedyDecoder(tokenizer)

    # Örnek senaryo:
    # "merhaba" kelimesi için modelin zamansal çıktısı:
    # [BLANK, m, m, BLANK, e, e, r, BLANK, h, a, a, b, a, a, BLANK]
    # Beklenen çıktı: "merhaba"
    char_seq = ["<BLANK>", "m", "m", "<BLANK>", "e", "e", "r", "<BLANK>", "h", "a", "a", "b", "a", "a", "<BLANK>"]
    mock_indices = [tokenizer.char2idx.get(c, 0) for c in char_seq]

    # Sahte one-hot log_probs tensörü üretelim: [Batch=1, Zaman=len, Sınıflar=31]
    time_len = len(mock_indices)
    vocab_size = tokenizer.vocab_size
    mock_probs = torch.zeros(1, time_len, vocab_size)
    for t, idx in enumerate(mock_indices):
        mock_probs[0, t, idx] = 10.0  # Seçilen sınıf yüksek logit alsın

    decoded = decoder.decode(mock_probs)
    print("=" * 60)
    print("CTC GREEDY DECODER TESTİ")
    print("=" * 60)
    print(f"Ham Model Zaman Dizisi: {char_seq}")
    print(f"Kod Çözülmüş Metin:     '{decoded[0]}'")

    # Metrik testi:
    ref = ["merhaba dunya"]
    hyp = ["merhaba donya"]  # 'u' yerine 'o' yazılmış (1 karakter hatası)
    metrics = decoder.compute_metrics(ref, hyp)
    print(f"Referans:               '{ref[0]}'")
    print(f"Hipotez (Tahmin):       '{hyp[0]}'")
    print(f"WER (Kelime Hatası):    {metrics['WER'] * 100:.2f}%")
    print(f"CER (Karakter Hatası):  {metrics['CER'] * 100:.2f}%")
    print("=" * 60)
