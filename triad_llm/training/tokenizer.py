from __future__ import annotations

from typing import List


class TiktokenWrapper:
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        import tiktoken

        self.encoding_name = encoding_name
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str) -> List[int]:
        return list(self.enc.encode(text))

    def decode(self, tokens: List[int]) -> str:
        return self.enc.decode(tokens)

    def encode_file(self, path: str) -> List[int]:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.encode(text)
