from minbpe import BasicTokenizer

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
    
def encode_turn(
    sender: str,
    text: str,
    tokenizer: BasicTokenizer,
) -> list[int]:
    return (encode_header(sender, tokenizer) + tokenizer.encode(text) + [tokenizer.special_tokens["<|end_turn|>"]])

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
