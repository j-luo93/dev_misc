from setuptools import setup, find_packages

setup(
    name='dev_misc',
    version='0.2',
    zip_safe=False,
    package_data={"dev_misc": ["py.typed"]},
    packages=find_packages())
