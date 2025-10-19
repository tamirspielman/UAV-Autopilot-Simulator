"""
AI-Based UAV Autopilot Simulator
================================
A comprehensive flight control system combining classical control theory,
sensor fusion, and reinforcement learning for autonomous UAV operation.
"""

__version__ = "1.0.0"
__author__ = "UAV Autopilot Team"

# Import only what's necessary, avoid circular imports
from .utils import FlightMode

__all__ = [
    'FlightMode'
]