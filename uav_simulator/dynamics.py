"""
UAV Dynamics - COMPLETE REWRITE
Simple, physically accurate quadcopter dynamics
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict
from .utils import rotation_matrix, normalize_angles, logger


@dataclass
class UAVState:
    """Complete state of the UAV"""
    # Position in NED (North-East-Down) coordinates
    # Down is positive Z, so negative Z = positive altitude
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -10.0]))
    
    # Velocity in world frame (m/s)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Orientation: [roll, pitch, yaw] in radians
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Angular velocity: [p, q, r] in rad/s
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Motor speeds (RPM) for 4 motors
    motor_speeds: np.ndarray = field(default_factory=lambda: np.array([5000.0, 5000.0, 5000.0, 5000.0]))
    
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
            'motor_speeds': self.motor_speeds.tolist(),
            'acceleration': self.acceleration.tolist(),
            'timestamp': self.timestamp
        }


class UAVDynamics:
    """
    Quadcopter dynamics model
    Simple and physically accurate
    """

    def __init__(self, mass: float = 1.5, arm_length: float = 0.25):
        # Physical parameters
        self.mass = mass                    # kg
        self.arm_length = arm_length        # meters
        self.gravity = 9.81                 # m/s^2
        
        # Moment of inertia (kg*m^2)
        self.inertia = np.diag([0.01, 0.01, 0.02])
        self.inertia_inv = np.linalg.inv(self.inertia)
        
        # Aerodynamic drag coefficient
        self.drag_coeff = 0.02
        
        # Motor/propeller coefficients
        self.thrust_coeff = 6e-8     # Thrust = k_t * omega^2
        self.torque_coeff = 1e-9       # Torque = k_q * omega^2
        
        # Motor limits
        self.min_rpm = 2000
        self.max_rpm = 8000
        
        logger.info(f"✓ UAVDynamics initialized (mass={mass}kg, arm={arm_length}m)")

    def update(self, state: UAVState, control_input: np.ndarray, dt: float) -> UAVState:
        """
        Update dynamics for one time step using RK4 integration
        
        Args:
            state: Current UAV state
            control_input: [throttle, roll_cmd, pitch_cmd, yaw_cmd]
            dt: Time step in seconds
        
        Returns:
            New UAV state
        """
        # Validate inputs
        if not np.all(np.isfinite(control_input)):
            logger.warning("Invalid control input, using safe defaults")
            control_input = np.array([0.55, 0.0, 0.0, 0.0])
        
        # Convert control inputs to motor speeds
        motor_speeds = self._control_to_motors(control_input)
        
        try:
            # RK4 integration for smooth, accurate dynamics
            k1 = self._state_derivative(state, motor_speeds)
            k2 = self._state_derivative(self._add_state(state, k1, dt/2), motor_speeds)
            k3 = self._state_derivative(self._add_state(state, k2, dt/2), motor_speeds)
            k4 = self._state_derivative(self._add_state(state, k3, dt), motor_speeds)
            
            # Combine derivatives: (k1 + 2*k2 + 2*k3 + k4) / 6
            derivative = {}
            for key in k1.keys():
                derivative[key] = (k1[key] + 2*k2[key] + 2*k3[key] + k4[key]) / 6
            
        except Exception as e:
            logger.error(f"Dynamics integration error: {e}")
            return state
        
        # Create new state
        new_state = UAVState()
        
        # Update position
        new_state.position = state.position + derivative['position'] * dt
        
        # Update velocity
        new_state.velocity = state.velocity + derivative['velocity'] * dt
        
        # Update orientation
        new_state.orientation = state.orientation + derivative['orientation'] * dt
        new_state.orientation = normalize_angles(new_state.orientation)
        
        # Update angular velocity
        new_state.angular_velocity = state.angular_velocity + derivative['angular_velocity'] * dt
        
        # Update motor speeds
        new_state.motor_speeds = motor_speeds
        
        # Update acceleration
        new_state.acceleration = derivative['velocity']
        
        # Update timestamp
        new_state.timestamp = state.timestamp + dt
        
        # CRITICAL: Ground protection - prevent going below ground
        # In NED: position[2] = 0 is ground level
        # Negative Z = above ground, Positive Z = below ground (bad!)
        if new_state.position[2] > 0.0:
            new_state.position[2] = 0.0
            # Stop downward velocity
            if new_state.velocity[2] > 0:
                new_state.velocity[2] = 0.0
            logger.warning("⚠️ Ground collision prevented")
        
        return new_state
    
    def _control_to_motors(self, control: np.ndarray) -> np.ndarray:
        throttle = np.clip(control[0], 0.0, 1.0)
        roll = np.clip(control[1], -0.5, 0.5)
        pitch = np.clip(control[2], -0.5, 0.5)
        yaw = np.clip(control[3], -0.3, 0.3)
        
        m0 = throttle + pitch + roll - yaw
        m1 = throttle + pitch - roll + yaw
        m2 = throttle - pitch - roll - yaw
        m3 = throttle - pitch + roll + yaw
        # Normalize to avoid exceeding limits
        motor_commands = np.array([m0, m1, m2, m3])
        max_cmd = np.max(np.abs(motor_commands))
        if max_cmd > 1.0:
            motor_commands = motor_commands / max_cmd
    
        # Convert to RPM
        rpm_range = self.max_rpm - self.min_rpm
        motor_speeds = motor_commands * rpm_range + self.min_rpm
    
        return np.clip(motor_speeds, self.min_rpm, self.max_rpm)
    
    def _state_derivative(self, state: UAVState, motor_speeds: np.ndarray) -> Dict:
        """
        Calculate state derivatives (rates of change)
        
        Returns:
            Dictionary with derivatives of position, velocity, orientation, angular_velocity
        """
        # Calculate forces and torques from motors
        thrust = self._calculate_thrust(motor_speeds)
        torques = self._calculate_torques(motor_speeds)
        
        # Aerodynamic drag (opposes velocity)
        drag = -self.drag_coeff * state.velocity * np.linalg.norm(state.velocity)
        
        # Rotation matrix: body frame -> world frame
        R = rotation_matrix(state.orientation)
        
        # Forces in body frame (thrust acts upward in body Z)
        # In NED: positive Z is down, so thrust is negative in world Z
        forces_body = np.array([0.0, 0.0, -thrust])
        
        # Transform forces to world frame and add gravity + drag
        # Gravity acts downward in world frame (positive Z in NED)
        forces_world = R @ forces_body + np.array([0.0, 0.0, self.mass * self.gravity]) + drag
        
        # Linear acceleration (world frame)
        acceleration = forces_world / self.mass
        
        # Angular acceleration (body frame)
        # Euler's rotation equation: I * dw/dt + w × (I * w) = torque
        inertia_times_omega = self.inertia @ state.angular_velocity
        angular_acceleration = self.inertia_inv @ (
            torques - np.cross(state.angular_velocity, inertia_times_omega)
        )
        
        # Return derivatives
        return {
            'position': state.velocity,           # dp/dt = v
            'velocity': acceleration,             # dv/dt = a
            'orientation': state.angular_velocity, # dΘ/dt = ω (simplified for small angles)
            'angular_velocity': angular_acceleration  # dω/dt = α
        }
    
    def _calculate_thrust(self, motor_speeds: np.ndarray) -> float:
        """
        Calculate total thrust from motor speeds
        
        Args:
            motor_speeds: [M0, M1, M2, M3] in RPM
            
        Returns:
            Total thrust force in Newtons
        """
        # Thrust = coefficient * sum(RPM^2)
        return self.thrust_coeff * np.sum(motor_speeds ** 2)
    
    def _calculate_torques(self, motor_speeds: np.ndarray) -> np.ndarray:
        """
        Calculate torques from motor speeds (roll, pitch, yaw)
        
        Args:
            motor_speeds: [M0, M1, M2, M3] in RPM
            
        Returns:
            Torque vector [roll, pitch, yaw] in N·m
        """
        l = self.arm_length
        kf = self.thrust_coeff
        km = self.torque_coeff
        
        # Square motor speeds for force calculation
        m0_sq = motor_speeds[0] ** 2
        m1_sq = motor_speeds[1] ** 2  
        m2_sq = motor_speeds[2] ** 2
        m3_sq = motor_speeds[3] ** 2
        
        # Roll torque (difference between right and left motors)
        roll_torque = l * kf * ((m0_sq + m3_sq) - (m1_sq + m2_sq))
        
        # Pitch torque (difference between front and rear motors)
        pitch_torque = l * kf * ((m0_sq + m1_sq) - (m2_sq + m3_sq))
        
        # Yaw torque (difference between CW and CCW motors)
        yaw_torque = km * (-m0_sq + m1_sq - m2_sq + m3_sq)
        
        return np.array([roll_torque, pitch_torque, yaw_torque])
    
    def _add_state(self, state: UAVState, derivative: Dict, dt: float) -> UAVState:
        """
        Create intermediate state for RK4 integration
        
        Args:
            state: Current state
            derivative: State derivatives
            dt: Time step
            
        Returns:
            New state advanced by derivative * dt
        """
        new_state = UAVState()
        
        # Advance each state variable
        new_state.position = state.position + derivative['position'] * dt
        new_state.velocity = state.velocity + derivative['velocity'] * dt
        new_state.orientation = state.orientation + derivative['orientation'] * dt
        new_state.angular_velocity = state.angular_velocity + derivative['angular_velocity'] * dt
        
        # Copy other fields
        new_state.motor_speeds = state.motor_speeds.copy()
        new_state.acceleration = derivative['velocity']  # Current acceleration
        new_state.timestamp = state.timestamp
        
        return new_state