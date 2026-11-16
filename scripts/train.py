"""V7 joint training; accepts --config and individual training options."""
from v7.cli import launch

if __name__ == '__main__':
    launch('v7.training.joint', config=True)
