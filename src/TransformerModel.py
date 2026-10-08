import sys
import torch
sys.path.append(str(Path(__file__).resolve().parent.parent))
from minbpe import BasicTokenizer
t = BasicTokenizer()
t.load(file_prefix=str(Path(__file__).resolve().parent.parent / "output" / "tokenizer" / "my_tokenizer.model"))
from typing import Optional, Tuple
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd

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
        #create multiple attention heads, feeding into the module list so that we can run the heads in parallel
        self.heads = = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * n_head. n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #loop over the heads and feed the input to each head independently and concatenate the outputs
        out = torch.cat([head(x) for head in self.heads]m dim=1)
        #apply dropout and project the concatenated output back to the original embedding dimension
        out = self.dropout(self.projection(out))
        return out

class FeedForward(nn.Module):
    # Linear Layer followed by a non-linear layer

    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            #first linear layer to expand the embedding dimension
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

class GPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embd_table = nn.Embedding(vocab_size, n_embd)
        self.postion_embd_table = nn.Embedding(block_size,n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.final_ln = nn.LayerNorm(n_embd)
        self.final_ll = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0,std=0.02)
    def forward(self, input_tokens: torch.Tensorm targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of GPT model.
        Args:
            input_tokens: Tensor of token indices of shape (batch_size, sequence_len)
            targets: Optional Tensor of target token indices of shape as input_tokens
        Returns:
            Tuple of (logits, loss) where logits has shape (batch_size, seq_len, vocab_size) 
            and loss is optional cross-entropy loss if targets are provided.
        """
        B, T = input_tokens.shape
        token_embd = self.token_embd_table(input_tokens)
        positional_embd = self.position_emdb_table(torch.arange(T, device = device))
        x = token_embd + positional_embd
        x = self.blocks(x)
        x = self.final_ln(x)
        logits = self.final_ll(x)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    # Generate new tokens based on the current input sequence.
    def generate(self, input_tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            cropped_input = input_tokens[:, -block_size:]
            logits, _ = self.forward(cropped_input)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_tokens = torch.cat([input_tokens, next_token], dim=1)
        return input_tokens

model = GPT()
model = model.to(device)
# num params: print(sum(p.numel() for p in model.parameters())/1e6, "M parameters")
batch_size = 1
seq_len = 6
x = torch.radint(0, vocab_size, (batch_size, seq_len))
x = x.to(device)
logits, loss = model(x)
print(logits.shape, loss)

def print_model_summary(model: torch.nn.Module, indent: str = '') -> None:
    for name, child in model.named_children():
        params = sum(p.numel() for p in child.parameters())
        print(f"{indent}| {name}: {child.__class__.__name__} ({params:,} parameters)")
        print_model_summary(child, indent + '|  ')

#print_model_summary(model)

def get_model_stats(model: torch.nn.Module) -> pd.DataFrame:
    stats = []
    for name, child in model.named_modules():
        if len(list(module.children())) == 0:
            params = sum(p.numel() for p in child.parameters())
            stats.append({
                'Layer name': name,
                'Type': module.__class__.__name__,
                'Parameters': params,
                'Trainable': sum(p.numel() for p in module.parameters() if p.requires_grad)
            })
    return pd.DataFrame(stats)

stats_df = get_model_stats(model)


