# Goals
* The ability to specify arguments in a file, and then can be optionally overridden by CLI.
* The ability for fast prototyping by declaring arguments locally.
* The ability for access the params everywhere (i.e., a global entry).

# Mechanism
Arguments should be declared where they belong. For instance, arguments related to training should be declared in a Trainer class.
