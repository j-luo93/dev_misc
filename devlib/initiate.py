from arglib import add_argument, g, parse_args
from trainlib import create_logger


def initiate():
    add_argument('log_dir', dtype=str, msg='log directory')
    add_argument('log_level', default='INFO', msg='log level')
    add_argument('message', default='', msg='message to append to the config class name')
    parse_args()

    # log_dir would be set as follows:
    # ./log/<config_class_name>[-<message>]/<timestamp>
