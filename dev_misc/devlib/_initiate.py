import os
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

from dev_misc.arglib import (Registry, add_argument, add_registry, g,
                             get_configs, parse_args, set_argument, show_args)
from dev_misc.trainlib import create_logger, set_random_seeds


def _is_in_git_repo() -> bool:
    res = subprocess.run('git rev-parse --is-inside-work-tree', shell=True, capture_output=True)
    if res.returncode == 127:
        raise OSError('git not installed.')
    elif res.returncode != 0:
        raise OSError('Some git-related error.')
    else:
        output = res.stdout.decode('utf8').strip()
        if output == 'true':
            return True
        else:
            return False


def _get_head_commit_id() -> str:
    if _is_in_git_repo():
        res = subprocess.run('git rev-parse --short HEAD', shell=True, capture_output=True)
        if res.returncode == 0:
            return res.stdout.decode('utf8').strip()
        else:
            return ''
    return ''


class DirAlreadyExists(Exception):
    """Raise this error if the specified `log_dir` already exists."""


class Initiator:

    def __init__(self, *registries: Registry,
                 logger=False, log_dir=False, log_level=False,
                 gpus=False, random_seed=False, commit_id=False,
                 stacklevel=1):
        """
        This function does a few things.
        1. Hook registries to arglib.
        2. Add a few default arguments: log_dir, log_level, message or random_seed. Note that setting up a random seed is not done by this function since there might be multiple seeds to set up.
        3. Automatically set up log_dir if not already specified and mkdir.
        4. Create a logger with proper log_level and file_path.
        5. Add the current head commit id.
        """
        if registries:
            for reg in registries:
                add_registry(reg, stacklevel=2)

        stacklevel = stacklevel + 1
        if log_dir:
            add_argument('log_dir', dtype='path', msg='log directory', stacklevel=stacklevel)
            add_argument('message', default='', msg='message to append to the config class name', stacklevel=stacklevel)
        if log_level:
            add_argument('log_level', default='INFO', msg='log level', stacklevel=stacklevel)
        if gpus:
            add_argument('gpus', dtype=int, nargs='+', msg='GPUs to use', stacklevel=stacklevel)
        if random_seed:
            add_argument('random_seed', dtype=int, default=1234, msg='random seed to set', stacklevel=stacklevel)
        if commit_id:
            commit_id = _get_head_commit_id()
            add_argument('commit_id', dtype=str, default=commit_id,
                         msg='commit id of current head, automatically computed', stacklevel=stacklevel)

        self.log_dir = log_dir
        self.log_level = log_level
        self.logger = logger
        self.random_seed = random_seed
        self.gpus = gpus

    def run(self):
        parse_args()
        # parse_args(known_only=True)

        # Set an environment variable.
        if self.gpus and g.gpus:
            # NOTE(j_luo) This environment variable is a string.
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, g.gpus))

        # log_dir would be automatically set as follows if it is not specified manually:
        # ./log/<date>/<config_class_name>[-<message>]/<timestamp>
        if self.log_dir:
            if g.log_dir:
                try:
                    g.log_dir.mkdir(parents=True)
                except FileExistsError:
                    raise DirAlreadyExists(f'Directory {g.log_dir} already exists.')
            else:
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
                        set_argument('log_dir', log_dir, _force=True)
                        log_dir.mkdir(parents=True)
                        break

        # Create a logger.
        if self.logger:
            file_path = Path(g.log_dir) / 'log' if self.log_dir else None
            log_level = g.log_level if self.log_level else 'INFO'
            create_logger(file_path=file_path, log_level=log_level)

        # Set random seed.
        if self.random_seed:
            set_random_seeds(g.random_seed)

        show_args()
