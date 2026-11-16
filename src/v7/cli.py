"""Small launch helper: JSON defaults followed by command-line overrides."""
import argparse
import json
import runpy
import sys
from pathlib import Path


def config_arguments(config):
    arguments = []
    for key, value in config.items():
        if value is None or key.startswith('_'):
            continue
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                arguments.append(flag)
            else:
                arguments.append('--no-' + key.replace('_', '-'))
        elif isinstance(value, (str, float, int)):
            arguments.extend([flag, str(value)])
        elif isinstance(value, list):
            arguments.extend([flag, *map(str, value)])
        else:
            raise ValueError(f'Unsupported config value: {key}')
    return arguments


def launch(module, *, config=False, extra=()):
    arguments = sys.argv[1:]
    if config:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--config', type=Path)
        options, arguments = parser.parse_known_args(arguments)
        defaults = [] if options.config is None else config_arguments(json.loads(options.config.read_text()))
        arguments = defaults + arguments
    sys.argv = [module, *arguments, *extra]
    runpy.run_module(module, run_name='__main__')
