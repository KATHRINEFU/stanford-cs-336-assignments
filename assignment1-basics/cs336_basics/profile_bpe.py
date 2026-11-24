from cs336_basics.train_bpe import train_bpe
if __name__ == "__main__":
    train_bpe("tests/fixtures/corpus.en", 500, ["<|endoftext|>"])