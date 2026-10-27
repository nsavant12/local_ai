import hashlib
from pathlib import Path

import torch
from minbpe import BasicTokenizer
from src.ModelTrainingAllBatches import get_vocab_size, select_device
from src.TransformerModel import GPT
from src.ConversationEncoding import encode_prompt

def load_for_inference(
    checkpoint_path: Path,
    device: torch.device | None = None,
) -> tuple[GPT, BasicTokenizer, str]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    config = checkpoint.get("model_config")
    info = checkpoint.get("tokenizer")
    objective = checkpoint.get("objective")

    if not isinstance(config, dict) or not isinstance(info, dict):
        raise ValueError("Checkpoint is missing model/tokenizer config settings")
    if objective not in {"flat-text", "reply"}:
        raise ValueError("Checkpoint has unsupported objective")
    if info.get("file") != "tokenizer.model":
        raise ValueError("Unsupported tokenizer snapshot filename")
    
    tokenizer_path = checkpoint_path.parent / info["file"]
    actual_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()

    if actual_hash != info.get("sha256"):
        raise ValueError("Tokenizer does not match this checkpoint")
    
    tokenizer = BasicTokenizer()
    tokenizer.load(str(tokenizer_path))

    if get_vocab_size(tokenizer) != config.get("vocab_size"):
        raise ValueError("tokenizer and model vocab sizes differ")

    if objective == "reply":
        required = {
            "<|start_turn|>",
            "<|separator|>",
            "<|end_turn|>",
            "<|padding|>",
        }
        if not required.issubset(tokenizer.special_tokens):
            raise ValueError("tokenizer is missing conversation markers")

    model = GPT(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    selected_device = device if device is not None else select_device()
    model = model.to(selected_device)
    model.eval()

    return model, tokenizer, objective

def generate_reply(
    model:GPT,
    tokenizer: BasicTokenizer,
    context: list[dict],
    max_new_tokens: int = 64,
    temp: float = 0.7,
    top_k: int | None = 50,
    self_sender: str = "Me"
) -> str:
    ctx = list(context[-6:])
    if not ctx: 
        raise ValueError("Need at least one context turn is required")
    
    prompt_ids = encode(ctx, tokenizer, self_sender)

    while len(prompt_ids) > model.block_size and len(ctx) > 1:
        ctx = ctx[1:]
        prompt_ids = encode_prompt(ctx, tokenizer, self_sender)
    
    if len(prompt_ids) > model.block_size:
        raise ValueError("Newest context turn exceeds the model window")
    
    device = next(model.parameters()).device
    input_tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    stop_id = tokenizer.special_tokens("<|end_turn|>")

    out = model.generate(
        input_tokens=input_tokens,
        max_new_tokens=max_new_tokens,
        temp=temp,
        top_k=top_k,
        stop_token_id=stop_id
    )

    reply_ids = out[0, len(prompt_ids):].tolist()
    if reply_ids and reply_ids[-1] == stop_id:
        reply_ids = reply_ids[:-1]

    return tokenizer.decode(reply_ids)
