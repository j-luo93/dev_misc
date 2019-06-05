# Motivation
The main mechanism to specify an argument or option to a class is to pass it down to its constructor. This is a top-down approach, and also a centralized approach. The advantage of this approach is that you can see all options easily in one file (typically where an ArgumentParser is constructed). But the disadvantage is that if you want to add one argument, you have to do it in two places -- the main argument file, and the file when the model is defined and the added argument is actually used. Another disadvantage is that normally I have to pass the arguments through `**kwargs` in `__init__`, which is not pretty in my opinion. And what if for fast prototyping, you have to add something to `kwargs`? It would be better if there is a global entry like the logging module that enables the classes to gain access to any argument. 

Is it possible to have a bottom-up approach? This is essentially the philosophy behind `register_xxx` decorators -- you register everything in many files, and have one file import every relevant file. But you lose the ability to have a centralized view every argument possible, and importing every relevant file is a bit weird for me because you are not only importing the params file and also the actual class definitions (unless these two are divided into two files). 

In sum, these are the features we need for quality argument parser.
* The ability to specify arguments in a file, and then can be optionally overridden by CLI.
* The ability to nest arguments, and also share arguments across commands (even if they are not descendents and ancestors).
* The ability for fast prototyping by allowing some ad-hoc arguments to be added and used without triggering errors in CLI (but warnings nonetheless). 
* The ability for access the params everywhere (i.e., a global entry)

# TODO
[x] Framework and global entry.
[x] Use file to set up default values.
[x] Ad-hoc arguments and help.
[x] Use trie to find arguments.
[x] `has_property` decorator.
[x] nargs.
[x] bool.
