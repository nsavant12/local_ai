import json 
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

from minbpe import BasicTokenizer
from src.ConversationEncoding import encode_ex, ExampleTooLongError

class ConversationDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer: BasicTokenizer,
        block_size: int,
        self_sender: str = "Me"
    ) -> None:
        if block_size <= 0:
            raise ValueError("block size must be > 0")

        self.samples = []
        self.skipped_too_long = 0

        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                
                ex = json.loads(line)
                try:
                    sample = encode_ex(ex, tokenizer, block_size, self_sender)
                except ExampleTooLongError:
                    self.skipped_too_long += 1
                    continue
                self.samples.append(sample)
        if not self.samples:
            raise ValueError("there is no usable conversation examples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]

def get_conversation_loaders(
    train_path: Path,
    val_path: Path,
    tokenizer: BasicTokenizer,
    block_size: int,
    batch_size: int,
    self_sender: str = "Me",
) -> tuple[DataLoader, DataLoader]:

    train_dataset = ConversationDataset(train_path, tokenizer, block_size, self_sender)
    val_dataset = ConversationDataset(val_path, tokenizer, block_size, self_sender)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader