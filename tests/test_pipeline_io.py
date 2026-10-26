"""Tokenizer input and checkpoint I/O using temporary synthetic files only."""

import json
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.BPE import load_training_text
from minbpe import BasicTokenizer
from src.ConversationDataset import get_conversation_loaders
from src.ConversationEncoding import encode_prompt
from src.DataCleaning import prep_examples, write_jsonl
from src.ModelTrainingAllBatches import estimate_loss, get_vocab_size, save_checkpoint
from src.TransformerModel import GPT


class PipelineIOTests(unittest.TestCase):
    def test_synthetic_sessions_train_evaluate_reload_and_generate(self):
        torch.manual_seed(42)
        records = []
        for session in range(4):
            start = datetime(2026, 1, 1, 12) + timedelta(days=session)
            turns = [
                ("Ava", f"plan-{session}"),
                ("Me", f"yes{session}"),
                ("Me", f"at noon{session}"),
                ("Bo", f"tea{session}"),
                ("Me", f"sure{session}"),
            ]
            for minute, (sender, message) in enumerate(turns):
                records.append({
                    "timestamp": start + timedelta(minutes=minute),
                    "sender": sender, "message": message,
                })
        train, val = prep_examples(
            {"invented-group": pd.DataFrame(list(reversed(records)))},
            val_fraction=0.25, seed=42,
        )
        train_sessions = {ex["context"][0]["text"] for ex in train}
        val_sessions = {ex["context"][0]["text"] for ex in val}
        self.assertEqual((len(train), len(val)), (6, 2))
        self.assertEqual((len(train_sessions), len(val_sessions)), (3, 1))
        self.assertTrue(train_sessions.isdisjoint(val_sessions))
        for example in train + val:
            session = example["context"][0]["text"][-1]
            texts = [turn["text"] for turn in example["context"]]
            texts.append(example["reply"])
            self.assertTrue(all(
                line.endswith(session) for text in texts for line in text.splitlines()
            ))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path, val_path = root / "splits/train.jsonl", root / "splits/val.jsonl"
            for examples, path in ((train, train_path), (val, val_path)):
                write_jsonl(examples, path)
                decoded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(decoded, examples)

            corpus = load_training_text(train_path)
            self.assertNotIn(val[0]["context"][0]["text"], corpus)
            tokenizer = BasicTokenizer()
            tokenizer.train(corpus, vocab_size=272, verbose=False)
            next_id = max(tokenizer.vocab) + 1
            tokenizer.register_special_tokens({
                marker: next_id + offset
                for offset, marker in enumerate((
                    "<|start_turn|>", "<|separator|>", "<|end_turn|>", "<|padding|>",
                ))
            })
            prefix = root / "synthetic-tokenizer"
            tokenizer.save(str(prefix))
            loaded_tokenizer = BasicTokenizer()
            loaded_tokenizer.load(str(prefix) + ".model")
            self.assertEqual(loaded_tokenizer.vocab, tokenizer.vocab)
            tokenizer = loaded_tokenizer
            train_loader, val_loader = get_conversation_loaders(
                train_path, val_path, tokenizer, block_size=96, batch_size=2,
            )
            for loader, examples in ((train_loader, train), (val_loader, val)):
                replies = []
                for x, y in loader:
                    self.assertEqual((x.device.type, y.device.type), ("cpu", "cpu"))
                    self.assertEqual(tuple(x.shape), tuple(y.shape))
                    self.assertEqual(x.shape[1], 96)
                    for row in y:
                        ids = row[row != -100].tolist()
                        self.assertEqual(ids[-1], tokenizer.special_tokens["<|end_turn|>"])
                        replies.append(tokenizer.decode(ids[:-1]))
                expected = [ex["reply"] for ex in examples]
                if loader is val_loader:
                    self.assertEqual(replies, expected)
                else:
                    self.assertCountEqual(replies, expected)

            config = dict(
                vocab_size=get_vocab_size(tokenizer), block_size=96,
                n_embd=16, n_head=2, n_layer=1, dropout=0.0,
            )
            model = GPT(**config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            initial = estimate_loss(model, train_loader, val_loader, eval_iters=3)
            for _ in range(8):
                for x, y in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    _, loss = model(x, y)
                    loss.backward()
                    optimizer.step()
            final = estimate_loss(model, train_loader, val_loader, eval_iters=3)
            self.assertLess(final["train"], initial["train"])
            self.assertTrue(all(torch.isfinite(torch.tensor(v)).item() for v in final.values()))
            checkpoint_path = root / "run/checkpoint.pth"
            save_checkpoint(
                model, optimizer, 7, final["train"], checkpoint_path,
                model_config=config,
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            restored = GPT(**checkpoint["model_config"])
            restored.load_state_dict(checkpoint["model_state_dict"])
            prompt = torch.tensor([encode_prompt(val[0]["context"], tokenizer)])
            settings = dict(max_new_tokens=8, temp=0.7, top_k=1,
                            stop_token_id=tokenizer.special_tokens["<|end_turn|>"])
            expected = model.generate(prompt, **settings)
            actual = restored.generate(prompt, **settings)
            torch.testing.assert_close(actual, expected)
            tokenizer.decode(actual[0, prompt.shape[1]:].tolist())

    def test_training_text_preserves_group_turns_and_multiline_reply(self):
        example = {
            "context": [
                {"sender": "Ava", "text": "Lunch?"},
                {"sender": "Bo", "text": "At noon?"},
            ],
            "reply": "Sure!\nSee you there",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_path = root / "train.jsonl"
            training_path.write_text(
                "\n" + json.dumps(example) + "\n\n", encoding="utf-8"
            )
            self.assertEqual(
                load_training_text(training_path),
                "Ava\nLunch?\nBo\nAt noon?\nMe\nSure!\nSee you there",
            )

    def test_empty_training_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.jsonl"
            path.write_text("\n  \n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_training_text(path)

    def test_checkpoint_creates_parents_and_reloads_model_and_optimizer(self):
        torch.manual_seed(42)
        config = dict(
            vocab_size=32, block_size=8, n_embd=16,
            n_head=2, n_layer=1, dropout=0.0,
        )
        model = GPT(**config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.tensor([[10, 11, 12, 20, 21]])
        y = torch.tensor([[-100, -100, 20, 21, 22]])
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new" / "nested" / "checkpoint.pth"
            self.assertFalse(path.parent.exists())
            save_checkpoint(
                model, optimizer, 2, loss.item(), path, model_config=config,
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(checkpoint["loss"], loss.item())
        restored = GPT(**checkpoint["model_config"])
        restored.load_state_dict(checkpoint["model_state_dict"])
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.5)
        restored_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.assertEqual(restored_optimizer.param_groups[0]["lr"], 1e-3)
        self.assertTrue(restored_optimizer.state)
        with torch.no_grad():
            expected, _ = model(x)
            actual, _ = restored(x)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
