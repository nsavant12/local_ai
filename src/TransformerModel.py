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
#hyperparameters for the transformer model
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
        # Compute the attention scores
        _,T, _ = x.shape
        keys = self.key(x)
        queries = self.query(x)
        weights = queries @ keys.transpose(-2,-1) * keys.shape[-1]**-0.5
        #prevent looking at future tokens
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        #turn the attention weights into a probability distribution
        weights = F.softmax(weights, dim=-1)
        #apply dropout to prevent overfitting
        weights = self.dropout(weights)
        values = self.value(x)
        return weights @ values

class MultiHeadSelfAttention(nn.Module):
    # mutliple heads of self attention in parallel

    def __init__(self, n_head: int, head_size: int) -> None:
        super().__init__()
        self.heads = = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * n_head. n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads]m dim=1)
        out = self.dropout(self.projection(out))
        return out

class FeedForward(nn.Module):
    # Linear Layer followed by a non-linear layer

    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd)
            nn.Dropout(dropout),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class Block(nn.Module):
    # Transformer block: multi-head self-attention + feed-forward network
    # n_embd: is our embedding dimension and n_head: is the number of heads that we want to use in the multi-head self-attention mechanism.
    def __init__(self, n_embd:int, n_head: int) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention = MultiHeadSelfAttention(n_head, head_size)
        self.feed_forward = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x






