from pathlib import Path
import sys
import json
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from minbpe import BasicTokenizer

def load_training_text(path: Path) -> str:
    pts = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            ex = json.loads(line)

            for turn in ex["context"]:
                pts.extend([turn["sender"], turn["text"]])
            pts.extend(["Me", ex["reply"]])
    if not pts:
        raise ValueError("No Training text found")
    return "\n".join(pts)

def main() -> None:
    training_path = (PROJECT_ROOT / "output" / "conversations" / "train.jsonl")
    txt_seq = load_training_text(training_path)

    t = BasicTokenizer()
    t.train(txt_seq, vocab_size=2048, verbose=False)
    next_id = max(t.vocab) + 1
    t.register_special_tokens({
        "<|start_turn|>": next_id,
        "<|separator|>": next_id + 1,
        "<|end_turn|>": next_id + 2,
        "<|padding|>": next_id + 3,
    })

    tokenizer_dir = PROJECT_ROOT / "output" / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    t.save(str(tokenizer_dir / "conversation_tokenizer"))

    print("Yaya convo tokenizer saved")

if __name__ == "__main__":
    main()
