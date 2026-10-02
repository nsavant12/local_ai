import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from minbpe import BasicTokenizer
t = BasicTokenizer()
t.load(file_prefix=str(Path(__file__).resolve().parent.parent / "output" / "tokenizer" / "my_tokenizer.model"))
def get_vocab_size(t : BasicTokenizer) -> int:
    vocab = t.vocab
    special_tokens = t.special_tokens
    return len(vocab) + len(special_tokens)