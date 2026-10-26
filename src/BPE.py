from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from minbpe import BasicTokenizer


def main() -> None:
    with open(str(Path(__file__).resolve().parent.parent / "output" / "combined_text.txt"), "r", encoding="utf-8") as f:
        text_sequence = f.read()

    t = BasicTokenizer()
    t.train(text_sequence, vocab_size = 2048, verbose=True)
    vocab = t.vocab

    #held_out = text_sequence[-50000:]
    #print(len(held_out)/len(t.encode(held_out)))

    next_id = max(t.vocab) + 1
    t.register_special_tokens({
        "<|start_turn|>": next_id,
        "<|separator|>": next_id + 1,
        "<|end_turn|>": next_id + 2,
        "<|padding|>": next_id + 3,
    })

    t.save(file_prefix=str(Path(__file__).resolve().parent.parent / "output" / "tokenizer" / "my_tokenizer"))
    #print(len(t.encode(text_sequence)))


if __name__ == "__main__":
    main()
