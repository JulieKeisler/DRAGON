from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

docs_extras = [
    "Sphinx >= 3.0.0",  # Force RTD to use >= 3.0.0
    "docutils",
    "pylons-sphinx-themes >= 1.0.8",  # Ethical Ads
    "pylons_sphinx_latesturl",
    "repoze.sphinx.autointerface",
    "sphinxcontrib-autoprogram",
    "sphinx-copybutton",
    "sphinx-tabs",
    "sphinx-panels",
    "sphinx-rtd-theme",
    "pillow>=6.2.0",
    "openml",
    "optuna",
    "matplotlib",
    "skorch"
]


setup(
    name="dragon_autodl",
    version="1.0",
    description='Implementation of the algorithmic framework from <An algorithmic framework for the optimization of deep neural networks architectures and hyperparameters>',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: CeCILL-C Free Software License Agreement (CECILL-C)",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "deep learning",
        "neural networks",
        "automl",
        "neural architecture search",
        "hyperparameters optimization",
        "metaheuristics",
    ],
    url="https://github.com/JulieKeisler/DRAGON",
    project_urls={
        "Bug Tracker": "https://github.com/JulieKeisler/DRAGON/issues",
    },
    packages=find_packages("lib"),
    package_dir={"": "lib"},
    author='Julie Keisler',
    author_email='julie.keisler.rfo@gmail.com',
    install_requires=[
        'numpy<2.0.0',
        'pandas',
        'torch',
        'graphviz'
    ],
    extras_require={
        "mpi": ["mpi4py>=3.1.2"],
        "docs": docs_extras
    },
    python_requires=">=3.9",
)
