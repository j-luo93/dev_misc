'''
Modified from MUSE
'''

from colorlog import ColoredFormatter
from datetime import timedelta
import logging
import time

class LogFormatter(ColoredFormatter):

    def __init__(self, color=False):
        if color:
            fmt = '%(log_color)s%(levelname)s - %(time)s - %(elapsed)s - %(message)s%(reset)s'
        else:
            fmt = '%(levelname)s - %(time)s - %(elapsed)s - %(message)s'
        super(LogFormatter, self).__init__(fmt)
        self.start_time = time.time()

    def format(self, record):
        # only need to set timestamps once -- all changes are stored in the record object
        if not hasattr(record, 'elapsed'):
            record.elapsed = timedelta(seconds=round(record.created - self.start_time))
            record.time = time.strftime('%x %X')
            prefix = "%s - %s - %s" % (
                record.levelname,
                record.time,
                record.elapsed
            )
            message = record.getMessage()
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
