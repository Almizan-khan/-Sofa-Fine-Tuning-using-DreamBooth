# Sofa concept fine-tuning · DreamBooth

A reproducible Stable Diffusion 1.5 workflow for a furniture concept, with image validation, training manifests and paired base-versus-fine-tuned inference.

[![checks](https://github.com/Almizan-khan/-Sofa-Fine-Tuning-using-DreamBooth/actions/workflows/checks.yml/badge.svg)](https://github.com/Almizan-khan/-Sofa-Fine-Tuning-using-DreamBooth/actions/workflows/checks.yml)


## Workflow

The maintained CLI validates local images, hashes each input, records the exact training command and delegates training to the official Diffusers DreamBooth example. The evaluation script uses identical prompts, random seeds, inference steps and guidance settings for the base and fine-tuned models.

The original training settings are retained: 512-pixel resolution, batch size 1, gradient accumulation 2, learning rate 2e-6, constant schedule, 800 steps and seed 1337. The current entry point is regular Python; notebook-only installation commands remain in the historical notebook.

## Validate the setup

Use Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

This lightweight installation checks data manifests and command construction without downloading models or requiring a GPU.

## Fine-tune

Use a CUDA environment with enough memory for full DreamBooth fine-tuning. Exact memory requirements depend on hardware and precision; this workflow does not claim an RTX 4060 fit measurement.

```bash
python -m pip install -r requirements-training.txt
git clone --depth 1 --branch v0.33.1 https://github.com/huggingface/diffusers.git external/diffusers
accelerate config
python train.py --images data/sofa --script external/diffusers/examples/dreambooth/train_dreambooth.py --dry-run
python train.py --images data/sofa --script external/diffusers/examples/dreambooth/train_dreambooth.py --output outputs/sofa-model
```

Use images you have permission to train on. Put PNG, JPEG or WebP images in `data/sofa`. Empty directories, invalid images and byte-identical duplicates are rejected. Training records image dimensions/hashes, configuration and the trainer script hash in `training_manifest.json`. The default base checkpoint is `stable-diffusion-v1-5/stable-diffusion-v1-5`; its upstream model license and usage terms apply independently.

`python sofa.main` is retained as a compatible entry point with the same arguments. Change `--prompt`, `--steps`, `--seed`, `--model` or `--precision` explicitly for a new experiment.

## Paired evaluation

```bash
python evaluate.py --fine-tuned outputs/sofa-model --prompts examples/prompts.json --seeds 17 42 73 --output outputs/comparison
```

This generates 18 images: two models × three prompts × three seeds. It saves image hashes, wall time, peak CUDA memory and safety-checker flags in `metrics.json`. The standard pipeline safety checker remains enabled. Timing includes the first inference warmup and should not be described as steady-state latency.

Inspect concept fidelity, prompt adherence, diversity and visible artifacts across the full paired set, not only a favorite image. Hold out distinct reference photos from training for an honest concept comparison. Do not infer improvement from latency or a single aesthetic example. A useful follow-up compares full fine-tuning with LoRA at a fixed data/compute budget.

## Evidence and current limits

CI verifies dataset validation and exact argument handling. No sofa dataset, trained checkpoint or generated comparison was available in this repository during this maintenance pass, so no image-quality score or GPU performance result is claimed. The reproducible training/evaluation path is provided so those results can be added with their actual artifacts.

Small concept datasets can overfit. This configuration does not enable prior-preservation training, and an 800-step budget is an experiment setting rather than a universal recommendation. Keep dataset permissions, held-out references and model provenance with each run.

See [validation](VALIDATION.md), [original experiment](notebooks/original_experiment.ipynb) and [original description](docs/original-readme.md).

## References

- [Official DreamBooth training example](https://github.com/huggingface/diffusers/tree/v0.33.1/examples/dreambooth)
- [Diffusers DreamBooth documentation](https://huggingface.co/docs/diffusers/training/dreambooth)
