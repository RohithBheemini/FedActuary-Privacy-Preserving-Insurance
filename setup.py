from setuptools import setup, find_packages

setup(
    name="fedactuary",
    version="2.0.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "fedactuary=src.cli:main",
        ],
    },
)
