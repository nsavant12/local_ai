"""Generation checks using tiny random models and invented token IDs only."""

from pathlib import Path
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import io
import sys
import unittest
from unittest.mock import patch

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.TransformerModel import GPT
from minbpe import BasicTokenizer
from src.Inference import generate_reply, main as chat_main


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.model = GPT(
            vocab_size=8, block_size=4, n_embd=8,
            n_head=2, n_layer=1, dropout=0.2,
        )

    def test_temperature_scaling_and_top_k_filtering(self):
        logits = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
        with patch.object(self.model, "forward", return_value=(logits, None)), \
             patch("torch.multinomial", return_value=torch.tensor([[7]])) as sample:
            self.model.generate(
                torch.tensor([[1]]), max_new_tokens=1,
                temp=0.5, top_k=2,
            )
        probabilities = sample.call_args.args[0]
        expected = torch.tensor([
            [0, 0, 0, 0, 0, 0, 0.11920292, 0.88079708]
        ])
        torch.testing.assert_close(probabilities, expected)

    def test_stop_uses_new_token_and_preserves_prompt_while_cropping(self):
        observed_inputs = []
        predictions = iter([5, 7])

        def forward(tokens):
            observed_inputs.append(tokens.clone())
            logits = torch.full((1, tokens.shape[1], 8), -20.0)
            logits[:, -1, next(predictions)] = 20.0
            return logits, None

        prompt = torch.tensor([[1, 7, 2, 4, 6]])
        with patch.object(self.model, "forward", side_effect=forward):
            out = self.model.generate(
                prompt, max_new_tokens=10, top_k=1, stop_token_id=7,
            )
        self.assertEqual(out.tolist(), [[1, 7, 2, 4, 6, 5, 7]])
        self.assertEqual(len(observed_inputs), 2)
        self.assertEqual(observed_inputs[0].tolist(), [[7, 2, 4, 6]])
        self.assertEqual(observed_inputs[1].tolist(), [[2, 4, 6, 5]])

    def test_generation_disables_gradients_and_temporarily_disables_dropout(self):
        self.model.train()
        observed = []
        original_forward = self.model.forward

        def forward(tokens):
            observed.append((self.model.training, torch.is_grad_enabled()))
            return original_forward(tokens)

        with patch.object(self.model, "forward", side_effect=forward):
            self.model.generate(torch.tensor([[1, 2]]), 2, top_k=1)
        self.assertTrue(self.model.training)
        self.assertEqual(observed, [(False, False), (False, False)])

    def test_forward_failure_restores_previous_mode(self):
        for was_training in (True, False):
            with self.subTest(was_training=was_training):
                self.model.train(was_training)
                with patch.object(
                    self.model, "forward",
                    side_effect=RuntimeError("Synthetic failure"),
                ):
                    with self.assertRaises(RuntimeError):
                        self.model.generate(torch.tensor([[1]]), 1)
                self.assertEqual(self.model.training, was_training)

    def test_without_stop_respects_budget_and_supports_batches(self):
        prompt = torch.tensor([[1, 2], [3, 4]])
        out = self.model.generate(prompt, 3, top_k=1)
        self.assertEqual(tuple(out.shape), (2, 5))
        torch.testing.assert_close(out[:, :2], prompt)
        with self.assertRaises(ValueError):
            self.model.generate(prompt, 3, stop_token_id=7)


class ReplyGenerationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = BasicTokenizer()
        self.tokenizer.register_special_tokens({
            "<|start_turn|>": 256, "<|separator|>": 257,
            "<|end_turn|>": 258, "<|padding|>": 259,
        })

    def model(self, block_size):
        return GPT(
            vocab_size=260, block_size=block_size, n_embd=8,
            n_head=2, n_layer=1, dropout=0.0,
        )

    def test_reply_uses_latest_six_turns_and_preserves_literal_text_and_unicode(self):
        model = self.model(96)
        context = [{"sender": "A", "text": str(i)} for i in range(7)]
        original = deepcopy(context)
        expected_prompt = []
        # Speaker A is byte 65; retained bodies 1..6 are bytes 49..54.
        for body_id in range(49, 55):
            expected_prompt.extend([256, 65, 257, body_id, 258])
        expected_prompt.extend([256, *b"Me", 257])
        reply = "OK ☕ <|end_turn|>"
        reply_ids = [*reply.encode("utf-8"), 258]
        predictions = iter(reply_ids)
        observed_inputs = []

        def forward(tokens):
            observed_inputs.append(tokens.clone())
            logits = torch.full((1, tokens.shape[1], 260), -20.0)
            logits[:, -1, next(predictions)] = 20.0
            return logits, None

        with patch.object(model, "forward", side_effect=forward):
            actual = generate_reply(
                model, self.tokenizer, context,
                max_new_tokens=len(reply_ids) + 5, top_k=1,
            )
        self.assertEqual(actual, reply)
        self.assertEqual(observed_inputs[0].tolist(), [expected_prompt])
        self.assertEqual(len(observed_inputs), len(reply_ids))
        self.assertEqual(context, original)

    def test_reply_trims_whole_old_turns_and_rejects_unusable_context(self):
        context = [
            {"sender": "A", "text": "Earlier context. " * 10},
            {"sender": "B", "text": "Q"},
        ]
        expected_prompt = [256, 66, 257, 81, 258, 256, *b"Me", 257]
        model = self.model(len(expected_prompt))
        observed_inputs = []

        def forward(tokens):
            observed_inputs.append(tokens.clone())
            logits = torch.full((1, tokens.shape[1], 260), -20.0)
            logits[:, -1, 258] = 20.0
            return logits, None

        with patch.object(model, "forward", side_effect=forward):
            reply = generate_reply(model, self.tokenizer, context, top_k=1)
        self.assertEqual(reply, "")
        self.assertEqual(observed_inputs[0].tolist(), [expected_prompt])
        too_small = self.model(len(expected_prompt) - 1)
        with self.assertRaises(ValueError):
            generate_reply(too_small, self.tokenizer, context)
        with self.assertRaises(ValueError):
            generate_reply(model, self.tokenizer, [])


class ChatCommandTests(unittest.TestCase):
    def test_chat_history_is_bounded_and_reset_starts_fresh(self):
        model = torch.nn.Linear(1, 1)
        observed_contexts = []

        def reply(model, tokenizer, context, **settings):
            observed_contexts.append(deepcopy(context))
            return f"reply{len(observed_contexts)}"

        messages = [
            "", "First", "Second", "Third", "Fourth", "Fifth",
            "/reset", "New conversation", "/quit",
        ]
        with patch("sys.argv", ["Inference", "synthetic.pth", "--speaker", "Ava"]), \
             patch("src.Inference.load_for_inference", return_value=(model, BasicTokenizer(), "reply")), \
             patch("src.Inference.generate_reply", side_effect=reply), \
             patch("builtins.input", side_effect=messages), \
             redirect_stdout(io.StringIO()):
            chat_main()

        self.assertEqual(len(observed_contexts), 6)
        self.assertEqual(observed_contexts[1], [
            {"sender": "Ava", "text": "First"},
            {"sender": "Me", "text": "reply1"},
            {"sender": "Ava", "text": "Second"},
        ])
        # Six retained history turns plus the incoming message.
        self.assertLessEqual(max(map(len, observed_contexts)), 7)
        self.assertEqual(observed_contexts[4][0]["text"], "Second")
        self.assertEqual(observed_contexts[-1], [
            {"sender": "Ava", "text": "New conversation"},
        ])

    def test_chat_rejects_flat_text_checkpoint_before_reading_messages(self):
        with patch("sys.argv", ["Inference", "synthetic.pth"]), \
             patch("src.Inference.load_for_inference", return_value=(None, None, "flat-text")), \
             patch("builtins.input") as read_message, \
             patch("src.Inference.generate_reply") as reply, \
             redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exited:
                chat_main()
        self.assertEqual(exited.exception.code, 2)
        read_message.assert_not_called()
        reply.assert_not_called()


if __name__ == "__main__":
    unittest.main()
