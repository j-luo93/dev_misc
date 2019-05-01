from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime
import os
import sys
import random

import numpy as np
import torch

from .map import Map

_CONFIGS = dict()
def register_cfg(cls):
    global _CONFIGS
    name = cls.__name__
    assert name not in _CONFIGS, name
    _CONFIGS[name] = cls
    return cls

def get_cfg(name):
    global _CONFIGS
    return _CONFIGS[name]

def _get_log_dir(args):
    while True:
        now = datetime.now()
        date = now.strftime("%m-%d")
        timestamp = now.strftime("%H:%M:%S")
        msg = args.msg
        msg = args.config + '-' * (msg != '') + msg
        log_dir = 'log/%s/%s-%s' %(date, msg, timestamp)

        try:
            os.makedirs(log_dir)
            break
        except OSError:
            pass
    return log_dir

class _CommandNode(object):

    def __init__(self, name, cmd_name, parent=None):
        self.parent = parent
        self.name = name
        self.cmd_name = cmd_name
        self.children = list()
        self.argument_info = list()
        self.bool_flags_info = list()

    def add_command(self, command):
        self.children.append(command)

    def add_argument(self, *args, **kwargs):
        info = Map(args=args, kwargs=kwargs)
        self.argument_info.append(info)

    def add_bool_flags(self, on_name, default=False):
        info = Map(on_name=on_name, default=default)
        self.bool_flags_info.append(info)

    def is_leaf(self):
        return not self.children

    def __repr__(self):
        return self.cmd_name

class CommandException(Exception):
    pass

class ArgParser(object):
    '''
    A customized class for handling CLI. It makes heavy use of argparse, but the main class ArgumentParser is initialized lazily.
    It works by building nested commands as a tree, and associate each leaf node with a combination of node-specific options,
    and options inherited from its parent.
    '''

    def __init__(self, parent=None, root=None):
        if parent is None:
            name = sys.argv[0]
            self.root_command = _CommandNode(name, '')
            self.name2command = {name: self.root_command}
        else:
            self.name2command = parent.name2command
            self.root_command = root

    def add_command(self, name):
        '''
        Return a ArgParser instance with different root node every time a new (sub)command is added.
        '''
        new_name = self.root_command.name + '=>' + name
        command = _CommandNode(new_name, name, parent=self.root_command)
        self.name2command[new_name] = command
        self.root_command.add_command(command)
        return ArgParser(parent=self, root=command)

    def add_argument(self, *args, **kwargs):
        self.root_command.add_argument(*args, **kwargs)

    def add_bool_flags(self, on_name, default=False):
        self.root_command.add_bool_flags(on_name, default=default)

    def parse_args(self, to_log=True):
        # get config args first
        _base_parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, add_help=False)
        _base_parser.add_argument('--config', '-cfg', type=str, help='Configure file')
        config_arg, _remaining_args = _base_parser.parse_known_args()
        defaults = {'config': config_arg.config}
        if config_arg.config:
            config_cls = get_cfg(config_arg.config)
            cfg = config_cls()
            defaults.update(cfg)

        # first parse the subcommand structure
        command = self.root_command
        chain = [command]
        i = 1 # NOTE the first argument is sys.argv is also the python script
        try:
            while not command.is_leaf():
                name = command.name + '=>' + sys.argv[i]
                command = self.name2command[name]
                chain.append(command)
                i += 1
        except (KeyError, IndexError):
            raise CommandException('This is not a leaf command. Possible subcommands are \n%r' %command.children)

        # now construct an ArgumentParser
        info = list()
        bool_info = list()
        for command in chain:
            info.extend(command.argument_info)
            bool_info.extend(command.bool_flags_info)
        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, parents=[_base_parser])
        for node_info in info:
            parser.add_argument(*node_info.args, **node_info.kwargs)
        for node_info in bool_info:
            on_name = node_info.on_name
            group = parser.add_mutually_exclusive_group(required=False)
            group.add_argument('--' + on_name, dest=on_name, action='store_true')
            group.add_argument('--no_' + on_name, dest=on_name, action='store_false')
            parser.set_defaults(**{on_name: node_info.default})
        parser.set_defaults(**defaults)
        args = parser.parse_args(sys.argv[i:])
        args.mode = '-'.join(chain[-1].name.split('=>')[1:])
        if to_log:
            args.log_dir = _get_log_dir(args)
        else:
            args.log_dir = None
        return args
        
######################################## parse_args method ########################################

def _check_args(args):
    assert args.loss_mode in ['embedding', 'distance', 'distance_layerwise', 'embedding_layerwise']
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not args.random:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    def check_monotonic(lst):
        for x, y in zip(lst[:-1], lst[1:]):
            assert x <= y
            
    check_monotonic(args.cur_pair_ckpts)
    check_monotonic(args.cur_op_ckpts)
    check_monotonic(args.cur_layer_ckpts)

def get_default_parser():
    parser = ArgParser()
    # global args
    parser.add_argument('--anneal_ends', type=int, nargs='+', default=[5], help='end of annealing (in epoch)')
    parser.add_argument('--anneal_schedule', type=str, default='dummy', help='anneal schedule mode')
    parser.add_argument('--anneal_starts', type=int, nargs='+', default=[0], help='start of annealing (in epoch)')
    parser.add_argument('--autoencoder_name', '-an', default='fixed-pairs', type=str, help='name of the autoencoder class')
    parser.add_argument('--basis_size', '-B', type=int, default=100, help='dimensionality of bases')
    parser.add_argument('--batch_size', '-bs', type=int, default=1280, help='how many words per batch')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--gpu', '-g', type=str, help='which gpu to choose')
    parser.add_argument('--input_size', '-is', type=int, default=300, help='dimensionality of inputs')
    parser.add_argument('--log_level', default='INFO', type=str, help='log level')
    parser.add_argument('--loss_mode', type=str, default='embedding', help='loss mode')
    parser.add_argument('--msg', '-M', default='', type=str, help='message')
    parser.add_argument('--num_fillers_per_role', '-nf', type=int, default=50, help='number of fillers per role')
    parser.add_argument('--num_pairs', '-np', type=int, default=2, help='number of role-filler pairs')
    parser.add_argument('--num_roles', '-nr', type=int, default=2, help='number of roles')
    parser.add_argument('--pretrained_embedding_path', '-pep', type=str, help='path to pretrained embeddings')
    parser.add_argument('--random', action='store_true', help='random, ignore seed')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--track', action='store_true', help='track log_dir so that shell can use it')
    parser.add_bool_flags('freeze_bases')
    # args for train
    train_parser = parser.add_command('train')
    train_parser.add_argument('--dec_layer_no', type=int, default='-1', help='layer number to approximate as the decoded layer.')
    train_parser.add_argument('--enc_layer_no', type=int, default='-1', help='layer number to approximate as the encoder layer.')
    train_parser.add_argument('--hidden_size', type=int, default='300', help='number of hidden units in LSTM')
    train_parser.add_argument('--include_labels', action='store_true', help='include labels for the stream')
    train_parser.add_argument('--layer_no', type=int, default='-1', help='layer number to approximate. -1 means all. -2 means nothing at all.')
    train_parser.add_argument('--lr_init', type=float, default=0.001, help='initial learning rate')
    train_parser.add_argument('--num_epochs', '-ne', type=int, default=5, help='number of epochs')
    train_parser.add_argument('--num_layers', '-nl', type=int, default=12, help='number of layers')
    train_parser.add_argument('--num_ops', '-nop', type=int, default=2, help='number of ops per hrr layer')
    train_parser.add_argument('--reg_hyper', type=float, default=0.0, help='regularization hyperparameter for orthonormality')
    train_parser.add_argument('--ent_hyper', type=float, default=0.0, help='regularization hyperparameter for entropy minimization')
    train_parser.add_argument('--input_role_range', nargs=2, type=int, default=[0, 0], help='role range for input')
    train_parser.add_argument('--output_role_range', nargs=2, type=int, default=[0, 0], help='role range for output')
    train_parser.add_argument('--save_all', action='store_true', help='flag to save all models')
    train_parser.add_argument('--saved_path', '-sp', type=str, help='path to saved states: continue training')
    train_parser.add_argument('--cur_pair_ckpts', nargs='*', default=[0], type=int, help='curriculum pair ckpts')
    train_parser.add_argument('--cur_op_ckpts', nargs='*', default=[0], type=int, help='curriculum op ckpts')
    train_parser.add_argument('--cur_layer_ckpts', nargs='+', default=[0], type=int, help='curriculum layer ckpts')
    train_parser.add_bool_flags('use_zero_bases', default=True)
    train_parser.add_bool_flags('use_gumbel', default=False)
    
    return parser

def parse_args(parser=None, to_log=True):
    if parser is None:
        parser = get_default_parser()
    args = parser.parse_args(to_log=to_log)
    _check_args(args)

    return Map(**vars(args)) # NOTE Return a map, not a Namespace.
