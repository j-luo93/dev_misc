'''
Modified from MUSE
'''

import logging
import time
from datetime import timedelta
from functools import wraps
from inspect import signature

from colorlog import ColoredFormatter


class LogFormatter(ColoredFormatter):

    def __init__(self, color=False):
        self.colored = color
        if self.colored:
            fmt = '%(log_color)s%(levelname)s - %(time)s - %(elapsed)s at %(filename)s:%(lineno)d - %(message)s%(reset)s'
        else:
            fmt = '%(levelname)s - %(time)s - %(elapsed)s - %(message)s'
        super(LogFormatter, self).__init__(fmt)
        self.start_time = time.time()

    def format(self, record):
        # only need to set timestamps once -- all changes are stored in the record object
        if not hasattr(record, 'elapsed'):
            record.elapsed = timedelta(seconds=round(record.created - self.start_time))
            record.time = time.strftime('%x %X')
            if self.colored:
                prefix = "%s - %s - %s at %s:%d" % (
                    record.levelname,
                    record.time,
                    record.elapsed,
                    record.filename,
                    record.lineno
                )
            else:
                prefix = "%s - %s - %s" % (
                    record.levelname,
                    record.time,
                    record.elapsed
                )
            message = record.getMessage()
            if not message.startswith('\n'): # If a message starts with a line break, we will keep the original line break without autoindentation. 
                message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
            record.msg = message
            record.args = () # NOTE avoid evaluating the message again duing getMessage call.
        x = super(LogFormatter, self).format(record)
        return x


def create_logger(filepath=None, log_level='INFO'):
    """
    Create a logger.
    """
    # create log formatter
    colorlog_formatter = LogFormatter(color=True)
    log_formatter = LogFormatter()

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(colorlog_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(log_level)
    logger.propagate = False
    logger.addHandler(console_handler)
    if filepath:
        # create file handler and set level to debug
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger

def log_this(msg='', log_level='DEBUG', arg_list=None):
    """
    A decorator that logs the functionality, the beginning and the end of the function.
    It can optionally print out arg values in arg_list.
    """

    def decorator(func):
        new_msg = msg or func.__name__
        new_arg_list = arg_list or list()
        log_func = lambda msg: logging.log(getattr(logging, log_level), msg)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            log_func(f'*STARTING* {new_msg}')

            if new_arg_list:

                func_sig = signature(func)
                bound = func_sig.bind(*args, **kwargs)
                bound.apply_defaults()
                all_args = bound.arguments

                arg_msg = {name: all_args[name] for name in new_arg_list}
                log_func(f'*ARG_LIST* {arg_msg}')

            ret = func(*args, **kwargs)
            log_func(f'*FINISHED* {new_msg}')
            return ret

        return wrapper

    return decorator

def log_pp(obj):
    '''
    Log ``obj`` with better indentations.
    '''
    logging.info(('\n' + str(obj)).replace('\n', '\n' + ' ' * 10))
