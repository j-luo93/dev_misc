import configparser as cfgp
import sys
from itertools import product

if __name__ == "__main__":
    file_path = sys.argv[1]
    parser = cfgp.ConfigParser(allow_no_value=True, interpolation=cfgp.ExtendedInterpolation())
    parser.read(file_path)

    assert 'shared' in parser
    assert 'script' in parser['shared']
    assert set(parser.sections()) <= {'grid', 'shared'}

    # Obtain base command with shared configurations.
    script = parser['shared']['script']
    base_cmd = f'python {script}'
    for k, v in parser['shared'].items():
        if v is None:
            base_cmd += f' --{k}'
        else:
            base_cmd += f' --{k} {v}'

    # Grid search over all grid variables.
    cmdl = list()
    if 'grid' in parser:
        grid = list()
        for k, v in parser['grid'].items():
            if v is None:
                grid.append([f'--{k}', f'--no_{k}'])
            else:
                grid.append([f'--{k} {vv.strip()}' for vv in v.split(',')])
        for combo in product(*grid):
            cmdl.append(base_cmd + ' ' + ' '.join(combo))
    else:
        cmdl.append(base_cmd)

    for cmd in cmdl:
        print(cmd)
