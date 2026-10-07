import sys
import torch
sys.path.append(str(Path(__file__).resolve().parent.parent))
from minbpe import BasicTokenizer
t = BasicTokenizer()
t.load(file_prefix=str(Path(__file__).resolve().parent.parent / "output" / "tokenizer" / "my_tokenizer.model"))
from typing import Optional, Tuple
import torch.nn as nn
from torch.nn import functional as F

def get_vocab_size(t : BasicTokenizer) -> int:
    vocab = t.vocab
    special_tokens = t.special_tokens
    return len(vocab) + len(special_tokens)

torch.manual_seed(3647)
block_size = 256
n_embd = 384
n_head = 6
n_layer = 6
dropout = .2
vocab_size = get_vocab_size(t)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Step 1: Head, represents a single attention head in an attention mechanism
class Head(nn.Module):
    def __init__(self, head_size:int) -> None:
        super().__init__()
        #
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:






