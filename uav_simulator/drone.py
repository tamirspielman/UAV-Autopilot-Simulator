from dataclasses import dataclass, field
from typing import Dict, Any, List, Union
import numpy as np
from .utils import logger
@dataclass
class DroneState:
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    timestamp: float = 0.0
    def to_dict(self) -> Dict:
        return {
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'orientation': self.orientation.tolist(),
            'angular_velocity': self.angular_velocity.tolist(),
            'acceleration': self.acceleration.tolist(),
            'timestamp': self.timestamp
        }
class Drone:
    def __init__(self, mass: float = 1.0, arm_length: float = 0.25):
        self.mass = mass 
        self.arm_length = arm_length  
        self.motor_speeds: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0]) 
        self.min_rpm = 3000      
        self.max_rpm = 9000    
        self.thrust_coeff = 5.5e-6   
        self.torque_coeff = 1.1e-7     
        self.true_state = DroneState()
        self.estimated_state = DroneState()
        self.drag_coeff = 0.04 
        self.battery_voltage = 11.1  
        self.motor_kv = 1000 
        logger.info(f"✓ Drone initialized (mass={mass}kg, arm={arm_length}m, thrust-to-weight=2:1)")
    def get_motor_speeds(self) -> np.ndarray:
        return self.motor_speeds.copy()
    def set_motor_speeds(self, speeds: np.ndarray):
        self.motor_speeds = np.clip(speeds, self.min_rpm, self.max_rpm)
    def calculate_thrust(self) -> float:
        motor_rads = self.motor_speeds * (2 * np.pi / 60)
        effective_k_t = 7.0e-6
        thrust_per_motor = effective_k_t * (motor_rads ** 2)
        total_thrust = float(np.sum(thrust_per_motor))
        return total_thrust
    def calculate_torques(self) -> np.ndarray:
        l = self.arm_length
        motor_rads = self.motor_speeds * (2 * np.pi / 60)
        motor_thrusts = self.thrust_coeff * (motor_rads ** 2)
        roll_torque = l * ((motor_thrusts[0] + motor_thrusts[3]) - (motor_thrusts[1] + motor_thrusts[2]))
        pitch_torque = l * ((motor_thrusts[0] + motor_thrusts[1]) - (motor_thrusts[2] + motor_thrusts[3]))
        yaw_torque = self.torque_coeff * (
            -(motor_rads[0]**2) + (motor_rads[1]**2) - (motor_rads[2]**2) + (motor_rads[3]**2)
        )
        return np.array([roll_torque, pitch_torque, yaw_torque])
    def get_hover_throttle(self) -> float:
        weight = self.mass * 9.80665  
        hover_thrust_per_motor = weight / 4.0
        omega_hover = np.sqrt(hover_thrust_per_motor / 7.0e-6)
        rpm_hover = omega_hover * (60 / (2 * np.pi))
        omega_rads = np.sqrt(hover_thrust_per_motor / self.thrust_coeff)
        rpm = omega_rads * (60 / (2 * np.pi))
        throttle = (rpm_hover - self.min_rpm) / (self.max_rpm - self.min_rpm)
        return float(np.clip(throttle, 0.55, 0.6))
    def get_telemetry(self) -> Dict[str, Any]:
        return {
            'position': self.estimated_state.position.tolist(),
            'velocity': self.estimated_state.velocity.tolist(),
            'orientation': self.estimated_state.orientation.tolist(),
            'motor_speeds': self.motor_speeds.tolist(),
            'altitude': float(-self.estimated_state.position[2])  
        }