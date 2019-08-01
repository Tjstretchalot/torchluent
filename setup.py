"""Uses setuptools to install the torchluent module"""
import setuptools
import os

setuptools.setup(
    name='torchluent',
    version='0.0.4',
    author='Timothy Moore',
    author_email='mtimothy984@gmail.com',
    description='Build pytorch models in a fluent interface',
    license='CC0',
    keywords='torch fluent models machinelearning',
    url='https://github.com/tjstretchalot/torchluent',
    packages=['torchluent'],
    long_description=open(
        os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    long_description_content_type='text/markdown',
    install_requires=['torch>=1.1.0', 'torchvision>=0.3.0', 'numpy', 'pytypeutils'],
    classifiers=(
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication',
        'Topic :: Utilities'),
    python_requires='>=3.6',
)
