import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from PIL import Image

def inspect_dataset(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError('Image directory does not exist')
    entries, hashes = [], set()
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
            continue
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes:
            raise ValueError('Duplicate training image: '+path.name)
        hashes.add(digest)
        entries.append({'file': path.name, 'width': width, 'height': height, 'sha256': digest})
    if not entries:
        raise ValueError('No supported training images found')
    return entries

def build_command(script, images, output, model, prompt, steps=800, seed=1337, precision='fp16'):
    if steps < 1 or not prompt.strip():
        raise ValueError('Training steps and prompt must be nonempty')
    script = Path(script).resolve()
    if not script.is_file() or script.name != 'train_dreambooth.py':
        raise ValueError('Provide the official Diffusers train_dreambooth.py path')
    if precision not in {'no', 'fp16', 'bf16'}:
        raise ValueError('Unsupported precision')
    return [sys.executable, '-m', 'accelerate.commands.launch', str(script),
            '--pretrained_model_name_or_path', model, '--instance_data_dir', str(Path(images).resolve()),
            '--output_dir', str(Path(output).resolve()), '--instance_prompt', prompt,
            '--resolution', '512', '--train_batch_size', '1', '--gradient_accumulation_steps', '2',
            '--learning_rate', '2e-6', '--lr_scheduler', 'constant', '--lr_warmup_steps', '0',
            '--max_train_steps', str(steps), '--mixed_precision', precision,
            '--checkpointing_steps', '100', '--seed', str(seed), '--gradient_checkpointing']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True)
    parser.add_argument('--script', required=True)
    parser.add_argument('--output', default='outputs/sofa-model')
    parser.add_argument('--model', default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    parser.add_argument('--prompt', default='a photo of a modern black sks sofa in a cozy living room')
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--precision', choices=['no', 'fp16', 'bf16'], default='fp16')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    images = inspect_dataset(args.images)
    command = build_command(args.script, args.images, args.output, args.model, args.prompt, args.steps, args.seed, args.precision)
    manifest = {'configuration': vars(args), 'images': images, 'command': command,
                'trainer_sha256': hashlib.sha256(Path(args.script).read_bytes()).hexdigest()}
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'training_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    subprocess.run(command, check=True)

if __name__ == '__main__':
    main()
