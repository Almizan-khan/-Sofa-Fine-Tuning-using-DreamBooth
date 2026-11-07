import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

def load_prompts(path):
    prompts = json.loads(Path(path).read_text())
    if not isinstance(prompts, list) or not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError('Prompts must be a nonempty JSON list of strings')
    return prompts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    parser.add_argument('--fine-tuned', required=True)
    parser.add_argument('--prompts', default='examples/prompts.json')
    parser.add_argument('--seeds', nargs='+', type=int, default=[17, 42, 73])
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='outputs/comparison')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    prompts = load_prompts(args.prompts)
    import torch
    from diffusers import StableDiffusionPipeline
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    measurements = []
    for label, checkpoint in [('base', args.base), ('fine_tuned', args.fine_tuned)]:
        pipe = StableDiffusionPipeline.from_pretrained(checkpoint, torch_dtype=torch.float16 if args.device.startswith('cuda') else torch.float32)
        pipe.to(args.device)
        pipe.enable_attention_slicing()
        for index, prompt in enumerate(prompts):
            for seed in args.seeds:
                generator = torch.Generator(device=args.device).manual_seed(seed)
                if args.device.startswith('cuda'):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                output = pipe(prompt, generator=generator, num_inference_steps=args.steps, guidance_scale=7.5, height=512, width=512)
                if args.device.startswith('cuda'):
                    torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                path = destination/f'{label}_prompt{index}_seed{seed}.png'
                output.images[0].save(path)
                measurements.append({'model': label, 'checkpoint': checkpoint, 'prompt': prompt, 'seed': seed,
                    'image': path.name, 'seconds': elapsed, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'peak_cuda_bytes': torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
                    'safety_checker_flag': output.nsfw_content_detected})
        del pipe
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    report = {'scope': 'paired visual comparison with identical prompts and seeds; latency includes first inference warmup',
              'config': vars(args), 'measurements': measurements,
              'environment': {'torch': torch.__version__, 'python': platform.python_version(), 'platform': platform.platform()}}
    (destination/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))

if __name__ == '__main__':
    main()
