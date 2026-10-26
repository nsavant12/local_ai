from minbpe import BasicTokenizer
import torch


class ExampleTooLongError(ValueError):
    """ The reply and newest context exceed the window"""

# puts structural ids around an encoded speaker names
def encode_header(
    sender: str,
    tokenizer: BasicTokenizer,
) -> list[int]:
    special = tokenizer.special_tokens
    return (
        [special["<|start_turn|>"]]
        + tokenizer.encode(sender)
        + [special["<|separator|>"]]
    )
# adds the message body and shared turn-ending id
def encode_turn(
    sender: str,
    text: str,
    tokenizer: BasicTokenizer,
) -> list[int]:
    return (encode_header(sender, tokenizer) + tokenizer.encode(text) + [tokenizer.special_tokens["<|end_turn|>"]])

#joins complete context turns and opens reply
def encode_prompt(
    ctx: list[dict],
    tokenizer: BasicTokenizer,
    self_sender: str = "Me",
) -> list[int]:
    tokens = []
    for turn in ctx:
        tokens.extend(
            encode_turn(turn["sender"], turn["text"], tokenizer)
        )
    tokens.extend(encode_header(self_sender, tokenizer))
    return tokens

def encode_ex(
    ex: dict,
    tokenizer: BasicTokenizer,
    block_size: int,
    self_sender: str = "Me",
) -> tuple[torch.Tensor, torch.Tensor]:
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    ctx = list(ex["context"])
    if not ctx:
        raise ValueError("Ex needs at least 1 context turn")

    reply = tokenizer.encode(ex["reply"])
    reply.append(tokenizer.special_tokens["<|end_turn|>"])

    prompt = encode_prompt(ctx, tokenizer, self_sender)
    tokens = prompt + reply

    while len(tokens) - 1 > block_size and len(ctx) > 1:
        ctx = ctx[1:]
        prompt = encode_prompt(ctx, tokenizer, self_sender)
        tokens = prompt + reply

    if len(tokens) - 1 > block_size:
        raise ExampleTooLongError("Reply & newest context turn exceed the block_size") 
    
    x = torch.tensor(tokens[:-1], dtype = torch.long)
    y = torch.tensor(tokens[1:], dtype = torch.long)

    #masking
    y[:len(prompt) - 1] = -100

    padding = block_size - len(x)
    pad_id = tokenizer.special_tokens["<|padding|>"]

    x = torch.cat([
        x,
        torch.full((padding,), pad_id, dtype=torch.long),
    ])
    y = torch.cat([
        y,
        torch.full((padding,), -100, dtype=torch.long),
    ])

    return x, y
