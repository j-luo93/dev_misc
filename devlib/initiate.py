import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

from arglib import (Registry, add_argument, add_registry, g, get_configs,
                    parse_args, set_argument)
from trainlib import create_logger


def initiate(*registries: Iterable[Registry], logger=False, log_dir=False, log_level=False, gpus=False):
    """
    This function does a few things.
    1. Hook registries to arglib.
    2. Add a few default arguments: log_dir, log_level and message.
    3. Automatically set up log_dir if not already specified and mkdir.
    4. Create a logger with proper log_level and file_path.
    """
    if registries:
        for reg in registries:
            add_registry(reg, stacklevel=2)

    if log_dir:
        add_argument('log_dir', dtype='path', msg='log directory', stacklevel=2)
        add_argument('message', default='', msg='message to append to the config class name', stacklevel=2)
    if log_level:
        add_argument('log_level', default='INFO', msg='log level', stacklevel=2)
    if gpus:
        add_argument('gpus', dtype=int, nargs='+', msg='GPUs to use', stacklevel=2)
    parse_args(known_only=True)

    # Set an environment variable.
    if gpus and g.gpus:
        # NOTE(j_luo) This environment variable is a string.
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, g.gpus))

    # log_dir would be automatically set as follows if it is not specified manually:
    # ./log/<date>/<config_class_name>[-<message>]/<timestamp>
    if log_dir and not g.log_dir:
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
    if logger:
        file_path = Path(g.log_dir) / 'log' if log_dir else None
        log_level = g.log_level if log_level else 'INFO'
        create_logger(file_path=file_path, log_level=log_level)
