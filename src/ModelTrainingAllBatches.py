from pathlib import Path
import sys
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from minbpe import BasicTokenizer
from src.TransformerModel import GPT

def get_vocab_size(tokenizer: BasicTokenizer) -> int:
    vocab = tokenizer.vocab
    special_tokens = tokenizer.special_tokens
    return len(vocab) + len(special_tokens)


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
    device: torch.device
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = TextDataset(train_data.to(device), block_size)
    val_dataset = TextDataset(val_data.to(device), block_size)

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
    model.eval()
    out = {}

    for split, loader in [("train", train_loader), ('val', val_loader)]:
        total_loss = torch.zeros(eval_iters)
        for i, (x,y) in enumerate(loader):
            if i >= eval_iters:
                break
            with torch.no_grad():
                _, loss = model(x,y)
                total_loss[i] = loss.item()
            out[split] = total_loss.mean().item()
    model.train()
    return out

def save_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    file_path: str = "checkpoint.pth"
) -> None:
    checkpoint={
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }
    torch.save(checkpoint, file_path)


def main() -> None:
    torch.manual_seed(42)

    tokenizer = BasicTokenizer()
    tokenizer_path = PROJECT_ROOT / "output" / "tokenizer" / "my_tokenizer.model"
    tokenizer.load(model_file=str(tokenizer_path))

    with open(PROJECT_ROOT / "output" / "combined_text.txt", "r", encoding="utf-8") as f:
        text_seq = f.read()
    encoded_text_seq = tokenizer.encode(text_seq)
    data = torch.tensor(encoded_text_seq, dtype=torch.long)
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    block_size = 256
    n_embd = 512
    n_head = 8
    n_layer = 8
    dropout = 0.2
    batch_size = 64
    vocab_size = get_vocab_size(tokenizer)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
    ).to(device)

    # model = torch.compile(model)

    train_loader, val_loader = get_data_loaders(
        train_data=train_data,
        val_data=val_data,
        block_size=block_size,
        batch_size=batch_size,
        device=device
    )
    x,y = next(iter(train_loader))
    x.shape, y.shape

    max_iters = 1
    eval_interval = 100
    eval_iters = 200
    learning_rate = 3e-4

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_loader, val_loader = get_data_loaders(
        train_data = train_data,
        val_data = val_data,
        block_size = block_size,
        batch_size = batch_size,
        device = device
    )

    train_losses = []
    val_losses = []

    for i in range(max_iters):
        for batch_idx, (x,y) in enumerate(train_loader):
            if batch_idx % eval_interval == 0 or batch_idx == len(train_loader) -1:
                losses = estimate_loss(
                    model = model,
                    train_loader = train_loader,
                    val_loader = val_loader,
                    eval_iters = min(eval_iters, len(val_loader))
                )
                train_losses.append(losses['train'])
                val_losses.append(losses['val'])
                print(f"iteration {i} / step {batch_idx}: " f"train loss {losses['train']:.4f}, " f"val loss {losses['val']:.4f}")

            logits, loss = model(x,y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=i,
            loss=loss.item(),
            file_path=f"../output/pre_training/run_4/checkpoint_{i}.pth"
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
