# Installation
1. `pip install -r requirements.txt` to install dependency.
1. Install [pytorch](https://pytorch.org/get-started/locally/).
1. `pip install .` to install this package.

# arglib
## Goals
* The ability to specify arguments in a file, and then can be optionally overridden by CLI.
* The ability for fast prototyping by declaring arguments locally.
* The ability for access the params everywhere (i.e., a global entry).

## Mechanism
Arguments should be declared where they belong. For instance, arguments related to training should be declared in a Trainer class.

# devlib
A library built upon arglib and trainlib which helps develop ml models.

# trainlib
Just a collection of useful functions and classes to deal with training.
