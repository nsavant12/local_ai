import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from minbpe import BasicTokenizer
tokenizer = BasicTokenizer()
tokenizer.load(mode_file="../output/tokenizer/my_tokenizer.model")
import torch
torch.manual_seed(42)
from transformer.model import GPT
from typing import Tuple
from toch.utils.data import DataLoader, Dataset
with open("../output/tokenizer/my_tokenizer.model", "r", encoding="utf-8") as f:
    text_seq = f.read()
encoded_text_seq = tokenizer.encode(text_seq)
data = torch.tensor(encoded_text_seq, dtype=torch.long)
split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]


def get_vocab_size(tokenizer: BasicTokenizer) -> int:
    vocab = tokenizer.vocab
    special_tokens = tokenizer.special_tokens
    return len(vocab) + len(special_tokens)

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
    dropout=dropout
    device = device
).to(device)

model = torch.compile(model)

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

train_loader, val_loader = get_data_loaders(
    train_data=train_data,
    val_data=val_data,
    block_size=block_size,
    batch_size=batch_size,
    device=device
)
x,y = next(iter(train_loader))
x.shape, y.shape