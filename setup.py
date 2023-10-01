#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="FixText",
    version="0.5.0",
    description="Fix Text is a semi supervised training method, inspired by Fix Match, which is designed to work for text based inputs.",
    author="Andrei Dumitrescu",
    author_email="andreidumitrescu99@gmail.com",
    url="https://github.com/AndreiDumitrescu99/FixText",
    install_requires=[
        "bitarray",
        "fairseq",
        "fastBPE",
        "hydra-core",
        "nltk",
        "pytorch-lightning",
        "pytorch_transformers",
        "regex",
        "sacrebleu",
        "sacremoses",
        "scikit-learn",
        "tensorboard",
        "tensorboardX",
        "torch",
        "tqdm",
    ],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
