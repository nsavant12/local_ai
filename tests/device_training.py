"""Synthetic training checks; no datasets, tokenizers, or caches are loaded.

Run from the project root:
    .venv/bin/python -B -m unittest discover -s tests -p device_training.py -v
"""

import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ModelTrainingAllBatches import estimate_loss, get_data_loaders, select_device
from src.TransformerModel import GPT


class DeviceTrainingTests(unittest.TestCase):
    def test_device_selection_prefers_cuda_then_mps_then_cpu(self) -> None:
        for cuda_available, mps_available, expected in (
            (True, True, "cuda"),
            (True, False, "cuda"),
            (False, True, "mps"),
            (False, False, "cpu"),
        ):
            with self.subTest(cuda=cuda_available, mps=mps_available):
                with patch("torch.cuda.is_available", return_value=cuda_available), \
                     patch("torch.backends.mps.is_available", return_value=mps_available):
                    self.assertEqual(select_device(), torch.device(expected))

    def _check_training_and_evaluation(self, device: torch.device) -> None:
        torch.manual_seed(42)
        train_ids = torch.randint(0, 32, (40,), device="cpu")
        val_ids = torch.randint(0, 32, (24,), device="cpu")
        train_loader, val_loader = get_data_loaders(
            train_ids, val_ids, block_size=8, batch_size=2
        )

        self.assertEqual(train_loader.dataset.data.device.type, "cpu")
        self.assertEqual(val_loader.dataset.data.device.type, "cpu")
        x, y = next(iter(train_loader))
        self.assertEqual(x.device.type, "cpu")
        self.assertEqual(y.device.type, "cpu")

        model = GPT(
            vocab_size=32,
            block_size=8,
            n_embd=16,
            n_head=2,
            n_layer=1,
            dropout=0.0,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Training transfers only the current batch to the model's device.
        x, y = x.to(device), y.to(device)
        before = model.final_ll.weight.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(x, y)
        self.assertEqual(tuple(logits.shape), (2, 8, 32))
        self.assertEqual(logits.device, next(model.parameters()).device)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
        optimizer.step()
        self.assertFalse(torch.equal(before, model.final_ll.weight))

        # Evaluation must transfer its own CPU batches and restore the caller's mode.
        for training in (True, False):
            model.train(training)
            losses = estimate_loss(model, train_loader, val_loader, eval_iters=2)
            self.assertEqual(set(losses), {"train", "val"})
            self.assertTrue(all(math.isfinite(value) for value in losses.values()))
            self.assertEqual(model.training, training)

        self.assertEqual(train_loader.dataset.data.device.type, "cpu")
        self.assertEqual(val_loader.dataset.data.device.type, "cpu")

    def test_cpu_training_and_evaluation(self) -> None:
        self._check_training_and_evaluation(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable in this environment")
    def test_cuda_training_and_evaluation(self) -> None:
        self._check_training_and_evaluation(torch.device("cuda"))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable in this environment")
    def test_mps_training_and_evaluation(self) -> None:
        self._check_training_and_evaluation(torch.device("mps"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
