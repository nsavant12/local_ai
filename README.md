This is my first (but now 2nd) commit.

## Training hardware

I used two different devices during development: a MacBook Pro (M3 Pro, 36 GB unified memory) and a
desktop with an NVIDIA RTX 3050 (6 GB VRAM), Ryzen 7 5700X and 32 GB RAM.

`src/ModelTrainingAllBatches.py` automatically selects CUDA when available,
otherwise Apple's MPS backend, otherwise CPU. It prints the selected device.
Datasets stay on CPU and each batch moves to the selected device for training
and evaluation. Mac unified memory and desktop GPU VRAM are different limits;
batch and model sizes still need to fit the machine used for the run.

Run the synthetic device checks from the project root:

```bash
.venv/bin/python -B -m unittest discover -s tests -p device_training.py -v
```

