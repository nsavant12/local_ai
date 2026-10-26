"""Conversation boundaries and supervision using invented records only."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from minbpe import BasicTokenizer
from src.ConversationDataset import ConversationDataset
from src.ConversationEncoding import encode_ex, ExampleTooLongError


class ConversationDataTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = BasicTokenizer()
        self.tokenizer.register_special_tokens({
            "<|start_turn|>": 256, "<|separator|>": 257,
            "<|end_turn|>": 258, "<|padding|>": 259,
        })

    def test_group_speakers_literal_markers_shifted_reply_and_padding(self):
        example = {
            "context": [
                {"sender": "Ava", "text": "<|end_turn|>"},
                {"sender": "Bo", "text": "Lunch?"},
            ],
            "reply": "Yes\nAt noon ☕",
        }
        # No BPE merges: ordinary text must be its exact UTF-8 byte IDs.
        prompt = [
            256, *b"Ava", 257, *b"<|end_turn|>", 258,
            256, *b"Bo", 257, *b"Lunch?", 258,
            256, *b"Me", 257,
        ]
        reply = [*example["reply"].encode("utf-8"), 258]
        size = len(prompt) + len(reply) - 1
        x, y = encode_ex(example, self.tokenizer, size + 3)
        self.assertEqual(x.tolist(), (prompt + reply)[:-1] + [259] * 3)
        self.assertEqual(
            y.tolist(), [-100] * (len(prompt) - 1) + reply + [-100] * 3,
        )
        self.assertEqual(x.dtype, torch.long)
        self.assertEqual(y.dtype, torch.long)
        self.assertEqual((x.device.type, y.device.type), ("cpu", "cpu"))

    def test_window_drops_whole_old_turns_without_shortening_reply(self):
        example = {
            "context": [
                {"sender": "Ava", "text": "Earlier context. " * 10},
                {"sender": "Bo", "text": "Lunch?"},
            ],
            "reply": "Yes\nAt noon",
        }
        original = deepcopy(example)
        prompt = [256, *b"Bo", 257, *b"Lunch?", 258, 256, *b"Me", 257]
        reply = [*b"Yes\nAt noon", 258]
        size = len(prompt) + len(reply) - 1
        x, y = encode_ex(example, self.tokenizer, size)
        self.assertEqual(x.tolist(), (prompt + reply)[:-1])
        self.assertEqual(y.tolist(), [-100] * (len(prompt) - 1) + reply)
        self.assertEqual(example, original)
        with self.assertRaises(ExampleTooLongError):
            encode_ex(example, self.tokenizer, size - 1)

    def test_dataset_skips_oversize_but_surfaces_malformed_records(self):
        valid = {"context": [{"sender": "A", "text": "Q"}], "reply": "A"}
        oversized = {
            "context": [{"sender": "A", "text": "Q" * 100}], "reply": "A",
        }
        malformed = {"context": [], "reply": "A"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.jsonl"
            path.write_text(
                json.dumps(valid) + "\n\n" + json.dumps(oversized) + "\n",
                encoding="utf-8",
            )
            dataset = ConversationDataset(path, self.tokenizer, 32)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.skipped_too_long, 1)
            path.write_text(json.dumps(oversized), encoding="utf-8")
            with self.assertRaises(ValueError):
                ConversationDataset(path, self.tokenizer, 32)
            path.write_text(
                json.dumps(valid) + "\n" + json.dumps(malformed),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                ConversationDataset(path, self.tokenizer, 32)


if __name__ == "__main__":
    unittest.main()
