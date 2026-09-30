"""Character-level tokenizer with special tokens (MASK, PAD)."""


class CharTokenizer:
    def __init__(self, chars, special_tokens=None):
        self.special_tokens = special_tokens or {}
        self.chars = list(chars)
        self.token_list = list(self.chars)
        self.special_ids = {}
        for name, tok in self.special_tokens.items():
            self.special_ids[name] = len(self.token_list)
            self.token_list.append(tok)
        self.stoi = {ch: i for i, ch in enumerate(self.token_list)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.token_list)

    def encode(self, s):
        return [self.stoi[ch] for ch in s]

    def decode(self, ids):
        return ''.join(self.itos.get(i, '?') for i in
                       (ids if isinstance(ids, list) else ids.tolist()))

    def __len__(self):
        return self.vocab_size
