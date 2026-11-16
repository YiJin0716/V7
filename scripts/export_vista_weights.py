"""Export the trained native VISTA tensors for independent component warmup."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if not saved.get('complete') or not saved.get('config', {}).get('public_only'):
        raise ValueError('Expected a completed public-only VISTA warmup')
    state = {k.removeprefix('model.'): v for k, v in saved['segmenter'].items() if k.startswith('model.')}
    if not state:
        raise ValueError('No native VISTA tensors')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    print(args.output)


if __name__ == '__main__':
    main()
