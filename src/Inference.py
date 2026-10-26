import hashlib
from pathlib import Path

import torch
from minbpe import BasicTokenizer
from src.ModelTrainingAllBatches import get_vocab_size, select_device
from src.TransformerModel import GPT

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
