from pathlib import Path
from datetime import datetime
import sys
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import hashlib
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from minbpe import BasicTokenizer
from src.TransformerModel import GPT
from src.ConversationDataset import get_conversation_loaders


def select_device() -> torch.device:
    """Prefer my desktop's CUDA GPU, then the Mac's MPS GPU, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_vocab_size(tokenizer: BasicTokenizer) -> int:
    return max(tokenizer.vocab) + 1


class TextDataset(Dataset):
    def __init__(self,data: torch.Tensor, block_size: int) -> None:
        if len(data) <= block_size:
            raise ValueError(f"The length of the data({len(data)}) needs to be greater than the block size ({block_size}).")
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + self.block_size + 1]
        return x, y

def get_data_loaders(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = TextDataset(train_data.cpu(), block_size)
    val_dataset = TextDataset(val_data.cpu(), block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    eval_iters: int
) -> Dict[str,float]:
    if eval_iters <= 0:
        raise ValueError("eval_iters needs to be a pos int.")
    device = next(model.parameters()).device
    was_training = model.training

    model.eval()
    out = {}
    try:
        for split, loader in [("train", train_loader), ('val', val_loader)]:
            total_loss = 0.0
            total_tokens = 0
            for i, (x,y) in enumerate(loader):
                if i >= eval_iters:
                    break
                x = x.to(device)
                y = y.to(device)

                _, loss = model(x,y)
                #cross-entropy avg over unmasked targets
                token_count = (y != -100).sum().item()
                total_loss += loss.item() * token_count
                total_tokens += token_count
            if total_tokens == 0:
                raise ValueError("No tokens, cannot compute average loss.")
            out[split] = total_loss / total_tokens
    finally:
        model.train(was_training)
    return out

def save_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    file_path: str | Path = "checkpoint.pth"
) -> None:
    checkpoint={
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }
    checkpoint_path = Path(file_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)

#Cache existing token IDs
def load_or_encode(
    txt: str,
    tokenizer: BasicTokenizer,
    tokenizer_path: Path,
    cache_dir: Path,
) -> torch.Tensor:
    # Identify exact txt and saved tokenizer
    txt_hash = hashlib.sha256(txt.encode('utf-8')).hexdigest()
    tokenizer_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"basic-v1-{txt_hash}-{tokenizer_hash}.pt"

    #Load tensor to RAM
    if cache_path.is_file():
        print("Loading cached token ids", flush = True)
        return torch.load(cache_path,map_location="cpu",weights_only=True)

    print("No matching cache. Cont tokenization", flush=True)
    data = torch.tensor(tokenizer.encode(txt), dtype=torch.long, device="cpu")

    temp_path = cache_path.with_suffix(".tmp")
    torch.save(data, temp_path)
    temp_path.replace(cache_path)
    print("Tokenization done, cache saved", flush=True)
    return data

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objective",
        choices=["flat-text", "reply"],
        default="flat-text"
    )
    args = parser.parse_args()
    torch.manual_seed(42)

    tokenizer = BasicTokenizer()
    tokenizer_filename = (
        "conversation_tokenizer.model"
        if args.objective == "reply"
        else "my_tokenizer.model"
    )
    tokenizer_path = PROJECT_ROOT / "output" / "tokenizer" / tokenizer_filename
    tokenizer.load(model_file=str(tokenizer_path))


    block_size = 256
    n_embd = 512
    n_head = 8
    n_layer = 8
    dropout = 0.2
    batch_size = 4
    vocab_size = get_vocab_size(tokenizer)
    device = select_device()
    print(f"Training device: {device}")

    model = GPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
    ).to(device)

    # model = torch.compile(model)

    if args.objective == "reply":
        required_markers = {
            "<|start_turn|>",
            "<|separator|>",
            "<|end_turn|>",
            "<|padding|>"
        }
        if not required_markers.issubset(tokenizer.special_tokens):
            raise ValueError("Saved tokenizer is missing conversation markers")

        convo_dir = PROJECT_ROOT / "output" / "conversations"
        train_loader, val_loader = get_conversation_loaders(
            train_path=convo_dir / "train.jsonl",
            val_path=convo_dir / "validate.jsonl",
            tokenizer=tokenizer,
            block_size=block_size,
            batch_size=batch_size,
        )
    else:
        corpus_path = PROJECT_ROOT / "output" / "combined_text.txt"
        txt_seq = corpus_path.read_text(encoding="utf-8")

        data = load_or_encode(
            txt=txt_seq,
            tokenizer=tokenizer,
            tokenizer_path=tokenizer_path,
            cache_dir=PROJECT_ROOT / "output" / "token_cache"
        )
        split_idx = int(0.9 * len(data))

        train_loader, val_loader = get_data_loaders(
            train_data=data[:split_idx],
            val_data=data[split_idx:],
            block_size=block_size,
            batch_size=batch_size,
        )

    max_iters = 1
    eval_interval = 100
    eval_iters = 200
    learning_rate = 3e-4

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    run_name = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    checkpoint_dir = (
        PROJECT_ROOT / "output" / "runs" / args.objective / run_name
    )

    for i in range(max_iters):
        for batch_idx, (x,y) in enumerate(train_loader):
            if batch_idx % eval_interval == 0 or batch_idx == len(train_loader) -1:
                losses = estimate_loss(
                    model = model,
                    train_loader = train_loader,
                    val_loader = val_loader,
                    eval_iters = eval_iters
                )
                train_losses.append(losses['train'])
                val_losses.append(losses['val'])
                print(f"iteration {i} / step {batch_idx}: " f"train loss {losses['train']:.4f}, " f"val loss {losses['val']:.4f}")

            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x,y)
            loss.backward()
            optimizer.step()
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=i,
            loss=loss.item(),
            file_path=checkpoint_dir / f"checkpoint_{i}.pth"
        )

    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.xlabel("eval step")
    plt.ylabel("loss")
    plt.title("Training & Validation Loss Over Time")
    plt.legend()
    plt.grid()
    plt.show()

    input_tokens = tokenizer.encode("The quick brown fox jumps over the lazy dog.")
    input_tokens = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out = model.generate(input_tokens = input_tokens, max_new_tokens=50, temperature=0.7, top_k=50)

    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
