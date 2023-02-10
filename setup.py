from setuptools import setup


setup(
    name="evodags",
    version="1.0",
    description='Implementation of the algorithmic framework from <An algorithmic framework for the optimization of deep neural networks architectures and hyperparameters>',
    url="https://github.com/JulieKeisler/SmoothAutoDL.git",
    package_dir={"evodags":"lib/evodags"},
    packages=['evodags'],
    author='Julie Keisler',
    author_email='julie.keisler@edf.fr',
    install_requires=[
        'numpy',
        'pandas',
        'zellij',
        'torch',
        'gluonts[torch,pro]'
    ],
    extras_require={
        "mpi": ["mpi4py>=3.1.2"]
    }
)
