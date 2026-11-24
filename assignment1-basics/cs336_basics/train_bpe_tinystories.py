import json
import pickle
from pathlib import Path

from cs336_basics.train_bpe import train_bpe

INPUT_PATH = "data/TinyStoriesV2-GPT4-valid.txt"
OUTPUT_DIR = Path("data/tinystories_bpe")


def save_tokenizer(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) For programs: pickle stores bytes exactly as they are, so loading it back gives
    #    the same vocab/merges. Not human-readable.
    with open(output_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(output_dir / "merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    # 2) For humans: JSON / text. Some tokens are not valid UTF-8 (e.g. half of a Chinese
    #    character), so errors="backslashreplace" shows those bytes as \xNN instead of crashing.
    def readable(token: bytes) -> str:
        return token.decode("utf-8", errors="backslashreplace")

    with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump({token_id: readable(token) for token_id, token in vocab.items()}, f, ensure_ascii=False, indent=2)
    with open(output_dir / "merges.txt", "w", encoding="utf-8") as f:
        for first, second in merges:
            # repr() keeps spaces/newlines visible, e.g. ' t' 'he'
            f.write(f"{readable(first)!r} {readable(second)!r}\n")


def load_tokenizer(output_dir: Path) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(output_dir / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open(output_dir / "merges.pkl", "rb") as f:
        merges = pickle.load(f)
    return vocab, merges


if __name__ == "__main__":
    vocab, merges = train_bpe(INPUT_PATH, 10000, ["<|endoftext|>"])

    save_tokenizer(vocab, merges, OUTPUT_DIR)
    print(f"Saved vocab ({len(vocab)}) and merges ({len(merges)}) to {OUTPUT_DIR}/")

    # Sanity check: what we load back should equal what we saved.
    loaded_vocab, loaded_merges = load_tokenizer(OUTPUT_DIR)
    assert loaded_vocab == vocab and loaded_merges == merges

    # Longest token: max over the bytes values, comparing by length.
    longest_token = max(vocab.values(), key=len)
    print(f"Longest token: {longest_token!r} ({len(longest_token)} bytes)")
    print(f"Decoded: {longest_token.decode('utf-8', errors='backslashreplace')!r}")
