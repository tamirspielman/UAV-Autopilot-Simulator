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
    Based on realistic UAV performance metrics
    """
    
    def __init__(self, mass: float = 1.0, arm_length: float = 0.25):
        # Physical parameters - 1kg quadcopter (consumer drone)
        self.mass = mass                    # kg (typical 0.5-2kg for consumer drones)
        self.arm_length = arm_length        # meters
        
        # Motor/propeller specifications
        # Target: 2:1 thrust-to-weight ratio for stable flight
        # Total thrust needed: 2 * 1kg = 2kg = 19.6N
        # Per motor: 0.5kg = 4.9N at max throttle
        self.motor_speeds: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0])  # RPM
        self.min_rpm = 3000      # Idle speed
        self.max_rpm = 9000      # Max speed for 6-inch props (realistic for 1kg drone)
        
        # Thrust coefficient calibrated for 2:1 thrust-to-weight ratio
        # At max RPM (9000), total thrust should be ~2kg = 19.6N
        # thrust_per_motor = k_t * omega^2
        # omega_max = 9000 * (2*pi/60) = 942.5 rad/s
        # 4.9N = k_t * (942.5)^2
        # k_t = 4.9 / 888806 ≈ 5.5e-6
        self.thrust_coeff = 5.5e-6     # Thrust = k_t * omega^2 (N/(rad/s)^2)
        self.torque_coeff = 1.1e-7     # Torque = k_q * omega^2 (Nm/(rad/s)^2)
        
        # State
        self.true_state = DroneState()
        self.estimated_state = DroneState()
        
        # Aerodynamic properties
        self.drag_coeff = 0.04  # Increased for more realistic drag
        
        # Battery properties (for reference)
        self.battery_voltage = 11.1  # 3S LiPo
        self.motor_kv = 1000  # RPM per volt (typical for 6" props)
        
        logger.info(f"✓ Drone initialized (mass={mass}kg, arm={arm_length}m, thrust-to-weight=2:1)")
    
    def get_motor_speeds(self) -> np.ndarray:
        """Get current motor speeds"""
        return self.motor_speeds.copy()
    
    def set_motor_speeds(self, speeds: np.ndarray):
        """Set motor speeds with limits"""
        self.motor_speeds = np.clip(speeds, self.min_rpm, self.max_rpm)
    
    def calculate_thrust(self) -> float:
        """Calculate total thrust from motor speeds"""
        # Convert RPM to rad/s
        motor_rads = self.motor_speeds * (2 * np.pi / 60)
        # Sum thrust from all motors
        total_thrust = float(self.thrust_coeff * np.sum(motor_rads ** 2))
        return total_thrust
    
    def calculate_torques(self) -> np.ndarray:
        """Calculate torques from motor speeds"""
        l = self.arm_length
        
        # Convert RPM to rad/s and square for thrust calculation
        motor_rads = self.motor_speeds * (2 * np.pi / 60)
        motor_thrusts = self.thrust_coeff * (motor_rads ** 2)
        
        # Roll torque (difference between right and left motors)
        # Motors: 0=FR, 1=FL, 2=RL, 3=RR
        roll_torque = l * ((motor_thrusts[0] + motor_thrusts[3]) - (motor_thrusts[1] + motor_thrusts[2]))
        
        # Pitch torque (difference between front and rear motors)
        pitch_torque = l * ((motor_thrusts[0] + motor_thrusts[1]) - (motor_thrusts[2] + motor_thrusts[3]))
        
        # Yaw torque (difference between CW and CCW motors)
        # Motors 0 and 2 spin CW, motors 1 and 3 spin CCW
        yaw_torque = self.torque_coeff * (
            -(motor_rads[0]**2) + (motor_rads[1]**2) - (motor_rads[2]**2) + (motor_rads[3]**2)
        )
        
        return np.array([roll_torque, pitch_torque, yaw_torque])
    
    def get_hover_throttle(self) -> float:
        """
        Calculate theoretical hover throttle with better accuracy
        """
        weight = self.mass * 9.80665  
        hover_thrust_per_motor = weight / 4.0
        omega_rads = np.sqrt(hover_thrust_per_motor / self.thrust_coeff)
        rpm = omega_rads * (60 / (2 * np.pi))
        throttle = (rpm - self.min_rpm) / (self.max_rpm - self.min_rpm)
        return float(np.clip(throttle, 0.45, 0.55)) 
    def get_telemetry(self) -> Dict[str, Any]:
        """Get telemetry data for display"""
        return {
            'position': self.estimated_state.position.tolist(),
            'velocity': self.estimated_state.velocity.tolist(),
            'orientation': self.estimated_state.orientation.tolist(),
            'motor_speeds': self.motor_speeds.tolist(),
            'altitude': float(-self.estimated_state.position[2])  # Convert NED to altitude
        }