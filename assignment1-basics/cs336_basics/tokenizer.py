from collections.abc import Iterable, Iterator
import pickle
import os
import regex as re

class Tokenizer:
    vocab = dict() # dict[int, bytes]
    merges = [] # list[tuple[bytes, bytes]]
    special_tokens = [] #list[str]
    num_of_process = 0
    target_chunk_bytes = 0
    pretokens = [] # list[typle(bytes, ...)]
    bytes_to_id = dict()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.num_of_process = min(4, os.cpu_count() or 1)
        self.target_chunk_bytes = 50 * 1024 * 1024 # 50MB
        self.bytes_to_id = {b: i for i, b in self.vocab.items()} 

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str]=None):
        """ Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
            (in the same format that your BPE training code output) and (optionally) a list of special
            tokens. This method should accept the following additional parameters"""
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        special_tokens = special_tokens
        return cls(vocab, merges, special_tokens)


    def pretokenize(
        self,
        text: str,
    ) -> list[tuple[bytes, ...]]:
        tokens = []
        if self.special_tokens:
            special_pattern = "(" + "|".join(
                re.escape(token)
                for token in sorted(self.special_tokens, key=len, reverse=True)
            ) + ")"
            parts = re.split(special_pattern, text)
        else:
            parts = [text]

        for part in parts:
            if part in self.special_tokens:
                tokens.append(self.bytes_to_id[part.encode("utf-8")])
                continue
            for match in re.finditer(self.PAT, part):
                pretoken_bytes = match.group().encode("utf-8")
                tokens.append(tuple(bytes([byte]) for byte in pretoken_bytes))
        return tokens

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs """  
        tokens = self.pretokenize(text)
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
            required for memory-eﬀicient tokenization of large files that we cannot directly load into
            memory."""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """ Decode a sequence of token IDs into text."""
