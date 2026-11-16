"""Warm up the native VISTA organ and nodule prompts."""
from v7.cli import launch

if __name__ == '__main__':
    launch('v7.training.vista_warmup', config=True)
