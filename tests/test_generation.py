"""Generation checks using tiny random models and invented token IDs only."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.TransformerModel import GPT


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


if __name__ == "__main__":
    unittest.main()
