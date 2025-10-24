import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlightMode(Enum):
    """Enumeration of available flight modes"""
    MANUAL = "manual"
    STABILIZE = "stabilize"
    ALTITUDE_HOLD = "altitude_hold"
    POSITION_HOLD = "position_hold"
    AUTO = "auto"
    RTL = "return_to_launch"
    LAND = "land"
    AI_PILOT = "ai_pilot"
    TAKEOFF = "takeoff"

def normalize_angles(angles: np.ndarray) -> np.ndarray:
    """Normalize angles to [-π, π] range"""
    return np.arctan2(np.sin(angles), np.cos(angles))

def wrap_angle(angle: float) -> float:
    """Wrap a single angle to [-π, π] range"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def rotation_matrix(angles: np.ndarray) -> np.ndarray:
    """Create 3D rotation matrix from Euler angles (roll, pitch, yaw)"""
    roll, pitch, yaw = angles
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])
    return R

def check_imports() -> Dict[str, bool]:
    """Check availability of optional dependencies for enhanced features"""
    imports: Dict[str, bool] = {}
    imports['numpy'] = True
    try:
        import torch
        import torch.nn as nn
        imports['torch'] = True
    except ImportError:
        imports['torch'] = False
        logger.info("PyTorch not installed. RL features will be limited.")
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        imports['matplotlib'] = True
    except ImportError:
        imports['matplotlib'] = False
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        imports['plotly'] = True
    except ImportError:
        imports['plotly'] = False
    try:
        import dash
        from dash import dcc, html, Input, Output
        imports['dash'] = True
    except ImportError:
        imports['dash'] = False
    try:
        import dash_bootstrap_components as dbc
        imports['dbc'] = True
    except ImportError:
        imports['dbc'] = False
        logger.info("dash_bootstrap_components not available - using basic styling")
    return imports