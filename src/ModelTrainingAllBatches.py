import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from minbpe import BasicTokenizer
tokenizer = BasicTokenizer()
tokenizer.load(mode_file="../output/tokenizer/my_tokenizer.model")
import torch
torch.manual_seed(42)
from transformer.model import GPT

def get_vocab_size(tokenizer: BasicTokenizer) -> int:
    vocab = tokenizer.vocab
    special_tokens = tokenizer.special_tokens
    return len(vocab) + len(special_tokens)

block_size = 256
n_embd = 512
n_head = 8
n_layer = 8