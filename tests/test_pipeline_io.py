"""Tokenizer input and checkpoint I/O using temporary synthetic files only."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.BPE import load_training_text
from src.ModelTrainingAllBatches import save_checkpoint
from src.TransformerModel import GPT


class PipelineIOTests(unittest.TestCase):
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
            save_checkpoint(model, optimizer, 2, loss.item(), path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(checkpoint["loss"], loss.item())
        restored = GPT(**config)
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
