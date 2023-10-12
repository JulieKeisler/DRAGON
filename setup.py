from setuptools import setup, find_packages


setup(
    name="evodags",
    version="1.0",
    description='Implementation of the algorithmic framework from <An algorithmic framework for the optimization of deep neural networks architectures and hyperparameters>',
    url="https://github.com/JulieKeisler/SmoothAutoDL.git",
    package_dir={"": "lib"},
    packages=find_packages("lib"),
    author='Julie Keisler',
    author_email='julie.keisler.rfo@gmail.com',
    install_requires=[
        'numpy',
        'pandas',
        'zellij @ git+ssh://git@github.com/TFirmin/zellig@dag',
        'torch',
        'gluonts[torch,pro]',
        'graphviz'
    ],
    extras_require={
        "mpi": ["mpi4py>=3.1.2"]
    }
)
