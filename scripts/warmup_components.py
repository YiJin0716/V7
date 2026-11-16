"""Warm up DETR, the refiner and semantic extractor independently."""
from v7.cli import launch

if __name__ == '__main__':
    launch('v7._core.model_v5.train', config=True, extra=('--warmup-only',))
