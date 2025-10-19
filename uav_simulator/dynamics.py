"""
UAV Dynamics and State Definitions
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict
from .utils import rotation_matrix, normalize_angles

@dataclass
class UAVState:
    """Complete state representation of the UAV"""
    # Position (NED coordinates - North, East, Down)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Velocity (body frame)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Orientation (roll, pitch, yaw) in radians
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Angular velocity (p, q, r) in rad/s
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Motor speeds (4 motors for quadcopter)
    motor_speeds: np.ndarray = field(default_factory=lambda: np.zeros(4))
    # Acceleration
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Timestamp
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'orientation': self.orientation.tolist(),
            'angular_velocity': self.angular_velocity.tolist(),
            'motor_speeds': self.motor_speeds.tolist(),
            'acceleration': self.acceleration.tolist(),
            'timestamp': self.timestamp
        }

class UAVDynamics:
    """
    High-fidelity 6-DOF UAV dynamics model
    """
    
    def __init__(self, mass: float = 1.5, arm_length: float = 0.25):
        # Physical parameters
        self.mass = mass  # kg
        self.arm_length = arm_length  # meters
        self.gravity = 9.81  # m/s^2
        
        # Inertia matrix (simplified for quadcopter)
        self.inertia = np.diag([0.01, 0.01, 0.02])  # kg*m^2
        self.inertia_inv = np.linalg.inv(self.inertia)
        
        # Aerodynamic coefficients
        self.drag_coeff = 0.1
        self.thrust_coeff = 1.5e-5
        self.torque_coeff = 3e-7
        
        # Motor parameters
        self.max_rpm = 10000
        self.motor_time_constant = 0.05  # seconds
        
    def update(self, state: UAVState, control_input: np.ndarray, dt: float) -> UAVState:
        """
        Update UAV state using Runge-Kutta 4th order integration
        control_input: [throttle, roll, pitch, yaw] commands
        """
        # Convert control input to motor speeds
        motor_speeds = self._mixer(control_input)
        
        # RK4 integration
        k1 = self._state_derivative(state, motor_speeds)
        k2 = self._state_derivative(self._add_derivative(state, k1, dt/2), motor_speeds)
        k3 = self._state_derivative(self._add_derivative(state, k2, dt/2), motor_speeds)
        k4 = self._state_derivative(self._add_derivative(state, k3, dt), motor_speeds)
        
        # Combine RK4 steps
        derivative = self._combine_derivatives(k1, k2, k3, k4)
        
        # Update state
        new_state = UAVState()
        new_state.position = state.position + derivative['position'] * dt
        new_state.velocity = state.velocity + derivative['velocity'] * dt
        new_state.orientation = state.orientation + derivative['orientation'] * dt
        new_state.angular_velocity = state.angular_velocity + derivative['angular_velocity'] * dt
        new_state.motor_speeds = motor_speeds
        new_state.acceleration = derivative['velocity']
        new_state.timestamp = state.timestamp + dt
        
        # Normalize angles
        new_state.orientation = normalize_angles(new_state.orientation)
        
        return new_state
    
    def _mixer(self, control_input: np.ndarray) -> np.ndarray:
        """
        Motor mixing algorithm (converts control to motor speeds)
        """
        throttle, roll, pitch, yaw = control_input
        
        # Quadcopter X configuration
        motor_speeds = np.array([
            throttle + roll + pitch - yaw,  # Front-right
            throttle - roll + pitch + yaw,  # Front-left
            throttle - roll - pitch - yaw,  # Rear-left
            throttle + roll - pitch + yaw   # Rear-right
        ])
        
        # Saturate motor speeds
        motor_speeds = np.clip(motor_speeds, 0, self.max_rpm)
        return motor_speeds
    
    def _state_derivative(self, state: UAVState, motor_speeds: np.ndarray) -> Dict:
        """Calculate state derivatives for dynamics integration"""
        # Calculate forces and torques
        thrust = self._calculate_thrust(motor_speeds)
        torques = self._calculate_torques(motor_speeds)
        drag = -self.drag_coeff * state.velocity * np.linalg.norm(state.velocity)
        
        # Rotation matrix from body to world frame
        R = rotation_matrix(state.orientation)
        
        # Linear dynamics
        forces_body = np.array([0, 0, -thrust])
        forces_world = R @ forces_body + np.array([0, 0, self.mass * self.gravity]) + drag
        linear_accel = forces_world / self.mass
        
        # Angular dynamics (using Euler's equations)
        angular_accel = self.inertia_inv @ (torques - np.cross(state.angular_velocity, 
                                                                self.inertia @ state.angular_velocity))
        
        return {
            'position': state.velocity,
            'velocity': linear_accel,
            'orientation': state.angular_velocity,
            'angular_velocity': angular_accel
        }
    
    def _calculate_thrust(self, motor_speeds: np.ndarray) -> float:
        """Calculate total thrust from motor speeds"""
        return self.thrust_coeff * np.sum(motor_speeds ** 2)
    
    def _calculate_torques(self, motor_speeds: np.ndarray) -> np.ndarray:
        """Calculate torques from motor speeds"""
        # Torque contributions from each motor
        roll_torque = self.arm_length * self.thrust_coeff * (
            motor_speeds[0]**2 + motor_speeds[3]**2 - 
            motor_speeds[1]**2 - motor_speeds[2]**2
        )
        pitch_torque = self.arm_length * self.thrust_coeff * (
            motor_speeds[0]**2 + motor_speeds[1]**2 - 
            motor_speeds[2]**2 - motor_speeds[3]**2
        )
        yaw_torque = self.torque_coeff * (
            -motor_speeds[0]**2 + motor_speeds[1]**2 - 
            motor_speeds[2]**2 + motor_speeds[3]**2
        )
        return np.array([roll_torque, pitch_torque, yaw_torque])
    
    def _add_derivative(self, state: UAVState, deriv: Dict, dt: float) -> UAVState:
        """Helper for RK4 integration"""
        new_state = UAVState()
        new_state.position = state.position + deriv['position'] * dt
        new_state.velocity = state.velocity + deriv['velocity'] * dt
        new_state.orientation = state.orientation + deriv['orientation'] * dt
        new_state.angular_velocity = state.angular_velocity + deriv['angular_velocity'] * dt
        return new_state
    
    def _combine_derivatives(self, k1: Dict, k2: Dict, k3: Dict, k4: Dict) -> Dict:
        """Combine RK4 derivatives"""
        combined = {}
        for key in k1.keys():
            combined[key] = (k1[key] + 2*k2[key] + 2*k3[key] + k4[key]) / 6
        return combined