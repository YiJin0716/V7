"""Predict a held-out CT and attach its annotations for inspection."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--case', required=True, help='For example LIDC-IDRI-0259__scan259')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--object-threshold', type=float)
    parser.add_argument('--mask-threshold', type=float, default=.5)
    args = parser.parse_args()
    import torch
    from v7.inference import JointModelEvaluator, save_result
    torch.set_num_threads(2)
    torch.manual_seed(42)
    evaluator = JointModelEvaluator.from_checkpoint(args.checkpoint, args.manifest, device=args.device)
    evaluator.model.requires_grad_(False)
    result = evaluator.evaluate(args.case, object_threshold=args.object_threshold,
                                mask_threshold=args.mask_threshold)
    save_result(result, args.output, plots=False, nifti=True)
    print(args.output)


if __name__ == '__main__':
    main()
