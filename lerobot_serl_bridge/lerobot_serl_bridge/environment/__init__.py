"""
Environment module for SERL-LeRobot integration.

This module provides wrappers and adapters for integrating
LeRobot robots with SERL training algorithms.
"""

from .robot_wrapper import SERLRobotEnvironment

__all__ = [
    "SERLRobotEnvironment",
]
