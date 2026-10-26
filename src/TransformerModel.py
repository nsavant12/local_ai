from typing import Optional, Tuple
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F


#Step 1: Head, represents a single attention head in an attention mechanism
class Head(nn.Module):
    
    def __init__(
        self, 
        n_embd: int, 
        head_size:int,
        block_size: int,
        dropout: float,
    ) -> None:
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

    def __init__(
        self,
        n_embd: int, 
        n_head: int, 
        head_size: int,
        block_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        #create multiple attention heads, feeding into the module list so that we can run the heads in parallel
        self.heads = nn.ModuleList([Head(n_embd=n_embd,head_size=head_size,block_size=block_size,dropout=dropout) for _ in range(n_head)])
        self.projection = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #loop over the heads and feed the input to each head independently and concatenate the outputs
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        #apply dropout and project the concatenated output back to the original embedding dimension
        out = self.dropout(self.projection(out))
        return out

class FeedForward(nn.Module):
    # Linear Layer followed by a non-linear layer

    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            #first linear layer to expand the embedding dimension
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class Block(nn.Module):
    # Transformer block: multi-head self-attention + feed-forward network
    # n_embd: is our embedding dimension and n_head: is the number of heads that we want to use in the multi-head self-attention mechanism.
    def __init__(
        self, 
        n_embd:int,
        n_head: int, 
        block_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if n_head <= 0 or n_embd % n_head != 0:
            raise ValueError(f"n_head must be pos. and divide n_embd evenly")
        head_size = n_embd // n_head
        self.self_attention = MultiHeadSelfAttention(
            n_embd=n_embd,
            n_head=n_head, 
            head_size=head_size, 
            block_size=block_size, 
            dropout=dropout,
        )
        self.feed_forward = FeedForward(n_embd=n_embd, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embd_table = nn.Embedding(vocab_size, n_embd)
        self.position_embd_table = nn.Embedding(block_size,n_embd)
        self.blocks = nn.Sequential(*[
            Block(
                n_embd=n_embd,
                n_head=n_head,
                block_size=block_size,
                dropout=dropout,
            )
            for _ in range(n_layer)
        ])
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
    def forward(self, input_tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        positional_embd = self.position_embd_table(torch.arange(T, device = input_tokens.device))
        x = token_embd + positional_embd
        x = self.blocks(x)
        x = self.final_ln(x)
        logits = self.final_ll(x)

        loss = None
        if targets is not None:
            #Catch bad batches where all targets are ignored
            if not (targets != -100).any().item():
                raise ValueError("This batch doesn't have any targets to learn from")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    # Generate new tokens based on the current input sequence.
    @torch.no_grad()
    def generate(self, input_tokens: torch.Tensor, max_new_tokens: int, temp: float=1.0, top_k: Optional[int] = None, stop_token_id: Optional[int] = None) -> torch.Tensor:

        if input_tokens.ndim != 2 or input_tokens.numel() == 0:
            raise ValueError("Expected nonempty batch, seq tensor")
        if max_new_tokens < 0:
            raise ValueError("max new tokens must be non negative")
        if temp <= 0:
            raise ValueError("temp must be pos")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be pos")
        if stop_token_id is not None and input_tokens.shape[0] != 1:
            raise ValueError("Stopping currently supports one conversation")

        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                cropped_input = input_tokens[:, -self.block_size:]
                logits, _ = self(cropped_input)
                logits = logits[:, -1, :] / temp

                if top_k is not None:
                    k = min(top_k, logits.shape[-1])
                    vals, _ = torch.topk(logits, k, dim=-1)
                    logits = logits.masked_fill(
                        logits < vals[:, -1:],
                        float("-inf"),
                    )
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_tokens = torch.cat([input_tokens, next_token], dim=1)
                if (stop_token_id is not None and next_token.item() == stop_token_id):
                    break
        finally:
            self.train(was_training)
        return input_tokens

def print_model_summary(model: torch.nn.Module, indent: str = '') -> None:
    for name, child in model.named_children():
        params = sum(p.numel() for p in child.parameters())
        print(f"{indent}| {name}: {child.__class__.__name__} ({params:,} parameters)")
        print_model_summary(child, indent + '|  ')



def get_model_stats(model: torch.nn.Module) -> pd.DataFrame:
    stats = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            params = sum(p.numel() for p in module.parameters())
            stats.append({
                'Layer name': name,
                'Type': module.__class__.__name__,
                'Parameters': params,
                'Trainable': sum(p.numel() for p in module.parameters() if p.requires_grad)
            })
    return pd.DataFrame(stats)


if __name__ == "__main__":
    torch.manual_seed(42)

    model = GPT(
        vocab_size=32,
        block_size=8,
        n_embd=16,
        n_head=2,
        n_layer=1,
        dropout=0.0,
    )
    model.eval()
    #Context tokens: 10, 11, 12. Reply tokens: 20, 21, 22
    tokens = torch.tensor([[10,11,12,20,21,22]])
    is_reply = torch.tensor([[False, False, False, True, True, True]])
    x= tokens[:, :-1]
    y = tokens[:, 1:].clone()
    y[~is_reply[:, 1:]] = -100

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.eval()
    with torch.no_grad():
        _, initial_loss = model(x, targets=y)
    print(f"Initial loss: {initial_loss.item():.4f}")
    model.train()
    for step in range(50):
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(x, targets=y)
        loss.backward()
        optimizer.step()
        if (step+1) % 10 == 0:
            print(f"Step {step+1}: loss = {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        logits, final_loss = model(x, targets=y)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Final loss: {final_loss.item():.4f}")

    #print_model_summary(model), stats_df = get_model_stats(model)


