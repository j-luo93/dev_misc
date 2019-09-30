import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

from arglib import (Registry, add_argument, add_registry, g, get_configs,
                    parse_args, set_argument)
from trainlib import create_logger


def initiate(registries: Iterable[Registry] = None):
    """
    This function does a few things.
    1. Hook registries to arglib.
    2. Add a few default arguments: log_dir, log_level and message.
    3. Automatically set up log_dir if not already specified and mkdir.
    4. Create a logger with proper log_level and file_path.
    """
    if registries:
        for reg in registries:
            add_registry(reg)
    add_argument('log_dir', dtype=str, msg='log directory')
    add_argument('log_level', default='INFO', msg='log level')
    add_argument('message', default='', msg='message to append to the config class name')
    parse_args(known_only=True)

    # log_dir would be automatically set as follows if it is not specified manually:
    # ./log/<date>/<config_class_name>[-<message>]/<timestamp>
    if not g.log_dir:
        folder = Path('./log')
        configs = get_configs()
        identifier = '-'.join(filter(lambda x: x is not None, configs.values()))
        identifier = identifier or 'default'
        if g.message:
            identifier += f'-{g.message}'

        while True:
            now = datetime.now()
            date = now.strftime(r"%Y-%m-%d")
            timestamp = now.strftime(r"%H-%M-%S")
            log_dir = folder / date / identifier / timestamp
            if log_dir.exists():
                time.sleep(1)
            else:
                set_argument('log_dir', log_dir, force=True)
                log_dir.mkdir(parents=True)
                break

    # Create a logger.
    create_logger(file_path=Path(g.log_dir) / 'log', log_level=g.log_level)
