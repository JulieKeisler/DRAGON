from setuptools import setup


setup(
    name="evodags",
    version="1.0",
    description='Implementation of the algorithmic framework from <An algorithmic framework for the optimization of deep neural networks architectures and hyperparameters>',
    packages=['evodags'],
    author='Julie Keisler',
    author_email='julie.keisler@edf.fr',
    install_requires=[
        'numpy',
        'pandas',
        'zellij',
        'torch'
        'glutons[torch,pro]'
    ],
)
