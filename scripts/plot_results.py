"""Make the figures and PDF after all evaluation shards finish."""
from v7.cli import launch

if __name__ == '__main__':
    launch('v7.evaluation.report')
