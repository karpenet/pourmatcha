#!/usr/bin/env python3
"""
Setup script for lerobot_serl_bridge package.
"""

from setuptools import setup, find_packages
import os

# Read the README file


def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Read requirements


def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_path):
        with open(req_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()
                    if line.strip() and not line.startswith("#")]
    return []


setup(
    name="lerobot_serl_bridge",
    version="0.1.0",
    author="SERL-LeRobot Integration Team",
    author_email="",
    description="Integration library for SERL online RL with LeRobot ACT policies",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lerobot_serl_bridge",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
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
    install_requires=[
        # Core dependencies
        "numpy>=1.21.0",
        "torch>=1.11.0",
        "jax>=0.4.20",
        "jaxlib>=0.4.20",
        "flax>=0.7.0",
        "optax>=0.1.7",

        # LeRobot dependencies (assumed to be installed separately)
        # These would be listed as optional or in requirements.txt

        # Optional SERL dependencies
        # "serl_launcher",  # This needs to be installed separately

        # Utility dependencies
        "wandb>=0.15.0",
        "tensorboard>=2.8.0",
    ],
    extras_require={
        "cuda11": ["jax[cuda11_pip]"],
        "cuda12": ["jax[cuda12_pip]"],
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=4.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lerobot-serl-train=lerobot_serl_bridge.scripts.train_online:main",
        ],
    },
    include_package_data=True,
    package_data={
        "lerobot_serl_bridge": ["docs/*.md"],
    },
    zip_safe=False,
)
