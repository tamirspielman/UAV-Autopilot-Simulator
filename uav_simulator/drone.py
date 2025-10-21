"""
Drone class - contains drone physical properties and state
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Union
import numpy as np
from .utils import logger

@dataclass
class DroneState:
    """Complete state of the drone"""
    # Position in NED (North-East-Down) coordinates
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    
    # Velocity in world frame (m/s)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Orientation: [roll, pitch, yaw] in radians
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Angular velocity: [p, q, r] in rad/s
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Acceleration in world frame (m/s^2)
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Timestamp
    timestamp: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'orientation': self.orientation.tolist(),
            'angular_velocity': self.angular_velocity.tolist(),
            'acceleration': self.acceleration.tolist(),
            'timestamp': self.timestamp
        }

class Drone:
    """
    Drone class containing physical properties and state
    """
    
    def __init__(self, mass: float = 1.5, arm_length: float = 0.25):
        # Physical parameters
        self.mass = mass                    # kg
        self.arm_length = arm_length        # meters
        
        # Motor/propeller specifications - INCREASED RPM RANGE FOR MORE THRUST
        self.motor_speeds: np.ndarray = np.array([3000.0, 3000.0, 3000.0, 3000.0])  # RPM
        self.min_rpm = 2000
        self.max_rpm = 10000  # Increased max RPM
        
        # INCREASED THRUST COEFFICIENT - critical fix!
        self.thrust_coeff = 1.5e-5     # Thrust = k_t * omega^2 (INCREASED FROM 6e-8)
        self.torque_coeff = 3e-7     # Torque = k_q * omega^2 (INCREASED FROM 1e-9)
        
        # State
        self.true_state = DroneState()
        self.estimated_state = DroneState()
        
        # Aerodynamic properties
        self.drag_coeff = 0.02
        
        logger.info(f"✓ Drone initialized (mass={mass}kg, arm={arm_length}m)")
    
    def get_motor_speeds(self) -> np.ndarray:
        """Get current motor speeds"""
        return self.motor_speeds.copy()
    
    def set_motor_speeds(self, speeds: np.ndarray):
        """Set motor speeds with limits"""
        self.motor_speeds = np.clip(speeds, self.min_rpm, self.max_rpm)
    
    def calculate_thrust(self) -> float:
        """Calculate total thrust from motor speeds"""
        motor_rads = self.motor_speeds * (2 * np.pi / 60)
        return float(self.thrust_coeff * np.sum(motor_rads ** 2))
    
    def calculate_torques(self) -> np.ndarray:
        """Calculate torques from motor speeds"""
        l = self.arm_length
        kf = self.thrust_coeff
        
        # Square motor speeds for force calculation
        m0_sq = self.motor_speeds[0] ** 2
        m1_sq = self.motor_speeds[1] ** 2  
        m2_sq = self.motor_speeds[2] ** 2
        m3_sq = self.motor_speeds[3] ** 2
        
        # Roll torque (difference between right and left motors)
        roll_torque = l * kf * ((m0_sq + m3_sq) - (m1_sq + m2_sq))
        
        # Pitch torque (difference between front and rear motors)
        pitch_torque = l * kf * ((m0_sq + m1_sq) - (m2_sq + m3_sq))
        
        # Yaw torque (difference between CW and CCW motors)
        yaw_torque = self.torque_coeff * (-m0_sq + m1_sq - m2_sq + m3_sq)
        
        return np.array([roll_torque, pitch_torque, yaw_torque])
    
    def get_telemetry(self) -> Dict[str, Any]:
        """Get telemetry data for display"""
        return {
            'position': self.estimated_state.position.tolist(),
            'velocity': self.estimated_state.velocity.tolist(),
            'orientation': self.estimated_state.orientation.tolist(),
            'motor_speeds': self.motor_speeds.tolist(),
            'altitude': float(-self.estimated_state.position[2])  # Convert NED to altitude
        }