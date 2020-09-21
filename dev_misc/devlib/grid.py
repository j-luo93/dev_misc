import configparser as cfgp
from argparse import ArgumentParser
from itertools import product
from typing import Any, ClassVar, Dict, List, Optional, Tuple

Arg2value = Dict[str, Any]


def _is_enclosed(arg: str) -> bool:
    return arg[0] == '(' and arg[-1] == ')'


class Command:

    base_cmd: ClassVar[str] = ''
    base_arg2value: ClassVar[Arg2value] = dict()  # These do NOT include flags.

    def __init__(self, arg_list: List[Tuple[str, Any]]):
        self._arg2value = dict()
        self._cmd = ''
        for arg, value in arg_list:
            self._set_arg_value(arg, value)

    def _set_arg_value(self, arg: str, value: Any):
        enclosed = _is_enclosed(arg)
        if enclosed:
            arg = arg[1:-1]
        self._arg2value[arg] = value

        if value is True:
            new_arg = f'--{arg}'
        elif value is False:
            new_arg = f'--no_{arg}'
        else:
            new_arg = f'--{arg} {value}'
        # Do not include this in the command if it is enclosed by parenthese.
        if not enclosed:
            self._cmd += ' ' + new_arg

    @classmethod
    def add_base_arg(cls, arg: str, value: Any):
        enclosed = _is_enclosed(arg)
        if enclosed:
            arg = arg[1:-1]

        if value is None:
            # `base_arg2value` does NOT record the arg-value pair because for flags, we do not know if they are True or False.
            new_arg = f'--{arg}'
        else:
            new_arg = f'--{arg} {value}'
            cls.base_arg2value[arg] = value

        if not enclosed:
            cls.base_cmd += ' ' + new_arg

    def interpolate(self, arg: str, fmt: str) -> str:
        arg2value = self.base_arg2value.copy()
        arg2value.update(self._arg2value)
        value = fmt.format(**arg2value)
        self._set_arg_value(arg, value)

    def __str__(self):
        return (self.base_cmd + self._cmd).strip()


def make_grid(file_path: str, script: Optional[str] = None) -> List[str]:
    """Return a list of command line strings (with optional script name)."""
    parser = cfgp.ConfigParser(allow_no_value=True, interpolation=cfgp.ExtendedInterpolation())
    parser.read(file_path)

    # Only three sections are allowed:
    # 1. shared: these arguments are shared among all commands.
    # 2. grid: these arguments might take different values, separated by commas.
    # 3. deps: these arguments are dependent on (combinations of) grid values. For instance,
    #          the log directionary might be dependent on both run id and the model name.
    #          They use Python's format string pattern for interpolation.
    # For all argument names, if they are enclosed by parentheses "()", then they would not show up as part of the command line,
    # though they are still useful for interpolation purposes.
    assert set(parser.sections()) <= {'shared', 'grid', 'deps'}

    # Obtain base command with shared configurations. If `script` is not provided, only keep the arguments.
    if script:
        base_cmd = f'python {script}'
    else:
        base_cmd = ''

    # Add shared arguments to the command.
    if 'shared' in parser:
        for arg, value in parser['shared'].items():
            Command.add_base_arg(arg, value)

    # Grid search over all grid variables.
    cmdl = list()
    if 'grid' in parser:
        grid = list()
        for arg, value in parser['grid'].items():
            if value is None:
                grid.append([(arg, True), (arg, False)])
            else:
                grid.append([(arg, vv.strip()) for vv in value.split(',')])
        for combo in product(*grid):
            cmd = Command(combo)
            cmdl.append(cmd)
    else:
        cmdl.append(base_cmd)

    # Parse the dependent values.
    if 'deps' in parser:
        for arg, fmt in parser['deps'].items():
            for cmd in cmdl:
                cmd.interpolate(arg, fmt)

    return [str(cmd) for cmd in cmdl]


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument('file_path', type=str, help='Path to the config file.')
    arg_parser.add_argument('--script', type=str, help='Script file')
    args = arg_parser.parse_args()

    cmdl = make_grid(args.file_path, script=args.script)

    for cmd in cmdl:
        print(cmd)
