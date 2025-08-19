#!/usr/bin/env python3
"""
Setup script for GRAID Human Evaluation Tool
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="graid-human-eval",
    version="1.0.0",
    author="GRAID Project",
    description="Human evaluation tool for GRAID-generated visual question-answering datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/graid-human-eval",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "graid-eval=gradio_eval_app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
