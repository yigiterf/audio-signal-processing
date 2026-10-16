# src/dataset/tokenizer.py
import re
from typing import List

class TurkishTokenizer:
    """
    Türkçe konuşma tanıma için karakter tabanlı CTC Tokenizer.
    İndeks 0 daima CTC Boşluk (Blank) token'ına ayrılmıştır.
    """
    def __init__(self):
        # 0 indeksi CTC Blank token'ıdır.
        self.blank_token = "<BLANK>"
        
        # Türkçe küçük harfler + boşluk karakteri
        self.vocab = [
            self.blank_token,
            " ", "a", "b", "c", "ç", "d", "e", "f", "g", "ğ",
            "h", "ı", "i", "j", "k", "l", "m", "n", "o", "ö",
            "p", "r", "s", "ş", "t", "u", "ü", "v", "y", "z"
        ]
        
        self.char2idx = {ch: idx for idx, ch in enumerate(self.vocab)}
        self.idx2char = {idx: ch for idx, ch in enumerate(self.vocab)}
    
    def clean_text(self, text: str) -> str:
        """
        Türkçe harf duyarlılığını koruyarak metni temizler.
        Büyük harfleri I/İ kuralına uygun küçültür ve noktalama işaretlerini eler.
        """
        # Türkçe özel büyük/küçük harf dönüşümü
        text = text.replace("İ", "i").replace("I", "ı")
        text = text.lower()
        
        # Sadece alfabemizdeki harfleri ve boşlukları koru
        valid_chars = set(self.vocab[1:])  # Blank hariç tüm geçerli karakterler
        cleaned = "".join([ch if ch in valid_chars else " " for ch in text])
        
        # Birden fazla boşluğu tek boşluğa indir ve kenarları kırp
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def text_to_indices(self, text: str) -> List[int]:
        """Temizlenmiş metni sayı dizisine çevirir."""
        cleaned = self.clean_text(text)
        return [self.char2idx[ch] for ch in cleaned]

    def indices_to_text(self, indices: List[int]) -> str:
        """Sayı dizisini tekrar metne dönüştürür (Blank token'ları atlar)."""
        chars = [self.idx2char[idx] for idx in indices if idx != 0 and idx in self.idx2char]
        return "".join(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


if __name__ == "__main__":
    # Test edelim
    tokenizer = TurkishTokenizer()
    sample_text = "Merhaba Dünya! Sinyaller ve Sistemler Dersi: 2026."
    cleaned = tokenizer.clean_text(sample_text)
    indices = tokenizer.text_to_indices(sample_text)
    reconstructed = tokenizer.indices_to_text(indices)
    
    print(f"Vocab Boyutu: {tokenizer.vocab_size}")
    print(f"Orijinal:     {sample_text}")
    print(f"Temizlenmiş:  {cleaned}")
    print(f"İndeksler:    {indices}")
    print(f"Geri Dönen:   {reconstructed}")