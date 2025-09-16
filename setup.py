#!/usr/bin/env python3
"""
Setup script for Fluid Image Boundary Simulation package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Lattice Boltzmann Method Fluid Simulation with GUI"

# Read requirements
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return [
            "numpy>=1.19.0",
            "matplotlib>=3.3.0", 
            "pillow>=8.0.0",
            "scipy>=1.6.0"
        ]

setup(
    name="fluid-image-boundary-sim",
    version="2.0.0",
    author="STOKEDMODELLER",
    description="Lattice Boltzmann Method Fluid Simulation with GUI",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "gui": [],  # tkinter is usually included with Python
        "dev": ["pytest", "pytest-cov", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "fluid-sim-gui=fluid_sim.gui.main_window:main",
        ],
    },
    include_package_data=True,
    package_data={
        "fluid_sim": ["config/*.json"],
    },
)