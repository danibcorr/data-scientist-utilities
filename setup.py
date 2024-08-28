import os
from setuptools import setup, find_packages
import codecs

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = "0.1.1"
DESCRIPTION = "Essential utilities for data scientists"
LONG_DESCRIPTION = """
    A package of essential tools and utilities for streamlining data science tasks like manipulation, 
    augmentation, visualization, among others, enhancing daily workflows.
"""

setup(
    name="datasu",
    version=VERSION,
    author="danibcorr (Daniel Bazo)",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danibcorr/data-scientist-utilities",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "numba",
        "ipywidgets",
        "matplotlib",
        "seaborn",
        "pandas",
        "scikit-learn",
    ],
    keywords=[
        "python",
        "data science",
        "machine learning",
        "deep learning",
        "artificial intelligence",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
