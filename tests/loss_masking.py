"""Check reply-only loss using invented token IDs and controlled batch losses."""

from pathlib import Path
import sys
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ModelTrainingAllBatches import estimate_loss
from src.TransformerModel import GPT


class LossMaskingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.model = GPT(
            vocab_size=32, block_size=8, n_embd=16,
            n_head=2, n_layer=1, dropout=0.0,
        )
        # Context: 10, 11, 12. Reply: 20, 21, 22.
        self.x = torch.tensor([[10, 11, 12, 20, 21]])
        self.y = torch.tensor([[-100, -100, 20, 21, 22]])

    def test_loss_matches_only_reply_predictions(self) -> None:
        logits, loss = self.model(self.x, self.y)
        # Independently calculate negative log-probabilities at the three reply positions.
        log_probs = logits[0, 2:].log_softmax(dim=-1)
        expected = -log_probs[torch.arange(3), torch.tensor([20, 21, 22])].mean()
        torch.testing.assert_close(loss, expected)
        self.assertEqual(tuple(logits.shape), (1, 5, 32))

    def test_masked_logits_have_no_direct_gradient_but_context_can_learn(self) -> None:
        logits, loss = self.model(self.x, self.y)
        logits.retain_grad()
        loss.backward()

        self.assertEqual(torch.count_nonzero(logits.grad[0, :2]).item(), 0)
        self.assertGreater(torch.count_nonzero(logits.grad[0, 2:]).item(), 0)
        # Reply predictions still depend on context through attention.
        context_gradient = self.model.token_embd_table.weight.grad[10]
        self.assertGreater(context_gradient.abs().sum().item(), 0)

    def test_fully_masked_batch_is_rejected_and_evaluation_restores_mode(self) -> None:
        ignored = torch.full_like(self.y, -100)
        with self.assertRaises(ValueError):
            self.model(self.x, ignored)

        loader = DataLoader(TensorDataset(self.x, ignored), batch_size=1)
        for training in (True, False):
            with self.subTest(training=training):
                self.model.train(training)
                with self.assertRaises(ValueError):
                    estimate_loss(self.model, loader, loader, eval_iters=2)
                self.assertEqual(self.model.training, training)

    def test_evaluation_weights_only_unmasked_targets(self) -> None:
        class KnownLossModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # estimate_loss uses a parameter to discover the model's device.
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(self, x, y):
                return None, x.mean()

        # First batch: one supervised token at loss 2.
        # Second batch: three supervised tokens at loss 4.
        x = torch.tensor([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]])
        y = torch.tensor([[0, -100, -100], [0, 0, 0]])
        loader = DataLoader(TensorDataset(x, y), batch_size=1)
        losses = estimate_loss(KnownLossModel(), loader, loader, eval_iters=10)
        for split in ("train", "val"):
            self.assertAlmostEqual(losses[split], 3.5)  # (1 * 2 + 3 * 4) / 4


if __name__ == "__main__":
    unittest.main(verbosity=2)
