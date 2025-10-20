"""
UAV Dynamics - STABILIZED VERSION
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict
from .utils import rotation_matrix, normalize_angles, logger


@dataclass
class UAVState:
    """Complete state representation of the UAV"""
    # Position (NED coordinates - North, East, Down)
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -10.0]))  # Start 10m ABOVE ground (Down negative)
    # Velocity (body frame)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Orientation (roll, pitch, yaw) in radians
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Angular velocity (p, q, r) in rad/s
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Motor speeds (4 motors for quadcopter)
    motor_speeds: np.ndarray = field(default_factory=lambda: np.array([5000, 5000, 5000, 5000]))  # Near-hover
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
    """Stabilized UAV dynamics model (hover-calibrated and ground-safe)"""

    def __init__(self, mass: float = 1.5, arm_length: float = 0.25):
        # Physical constants
        self.mass = mass
        self.arm_length = arm_length
        self.gravity = 9.81

        # Inertia
        self.inertia = np.diag([0.01, 0.01, 0.02])
        self.inertia_inv = np.linalg.inv(self.inertia)

        # Aerodynamic & motor coefficients (empirically balanced)
        self.drag_coeff = 0.02
        self.thrust_coeff = 1.5e-7     # tuned for 5000 RPM hover
        self.torque_coeff = 3e-9
        self.max_rpm = 10000
        self.motor_time_constant = 0.05

        logger.info("UAVDynamics initialized with stable parameters (hover-calibrated)")

    # ===============================================================
    def update(self, state: UAVState, control_input: np.ndarray, dt: float) -> UAVState:
        """Run one integration step with hover and ground stability"""
        if not np.all(np.isfinite(control_input)):
            control_input = np.array([0.55, 0, 0, 0])

        motor_speeds = self._mixer(control_input)

        try:
            k1 = self._state_derivative(state, motor_speeds)
            k2 = self._state_derivative(self._add_derivative(state, k1, dt/2), motor_speeds)
            k3 = self._state_derivative(self._add_derivative(state, k2, dt/2), motor_speeds)
            k4 = self._state_derivative(self._add_derivative(state, k3, dt), motor_speeds)
            deriv = self._combine_derivatives(k1, k2, k3, k4)
        except Exception as e:
            logger.error(f"Dynamics error: {e}")
            return state

        new_state = UAVState()
        for attr in ["position", "velocity", "orientation", "angular_velocity"]:
            val = getattr(state, attr) + deriv[attr] * dt
            if attr == "position":
                # Clamp horizontal spread, ground at z >= 0
                val[0:2] = np.clip(val[0:2], -1000, 1000)
                val[2] = max(0.0, val[2])
            setattr(new_state, attr, val)

        new_state.motor_speeds = motor_speeds
        new_state.acceleration = deriv["velocity"]
        new_state.timestamp = state.timestamp + dt
        new_state.orientation = normalize_angles(new_state.orientation)
        return new_state

    # ===============================================================
    def _mixer(self, control_input: np.ndarray) -> np.ndarray:
        """Motor mixing (X config, RPM output)"""
        throttle, roll, pitch, yaw = np.clip(control_input, [0, -0.5, -0.5, -0.3], [1, 0.5, 0.5, 0.3])
        motor_cmd = np.array([
            throttle + 0.5*pitch + 0.5*roll - 0.3*yaw,  # front-right
            throttle + 0.5*pitch - 0.5*roll + 0.3*yaw,  # front-left
            throttle - 0.5*pitch - 0.5*roll - 0.3*yaw,  # rear-left
            throttle - 0.5*pitch + 0.5*roll + 0.3*yaw   # rear-right
        ])
        motor_speeds = motor_cmd * 3000 + 5000
        return np.clip(motor_speeds, 2000, 8000)

    # ===============================================================
    def _state_derivative(self, state: UAVState, motor_speeds: np.ndarray) -> Dict:
        thrust = self._calculate_thrust(motor_speeds)
        torques = self._calculate_torques(motor_speeds)
        drag = -self.drag_coeff * state.velocity * np.linalg.norm(state.velocity)
        R = rotation_matrix(state.orientation)

        # Body → world, NED convention: Down = +Z
        forces_body = np.array([0, 0, -thrust])          # thrust up
        forces_world = R @ forces_body - np.array([0, 0, self.mass * self.gravity]) + drag
        accel_world = forces_world / self.mass

        ang_accel = self.inertia_inv @ (torques - np.cross(state.angular_velocity, self.inertia @ state.angular_velocity))
        return {
            "position": state.velocity,
            "velocity": accel_world,
            "orientation": state.angular_velocity,
            "angular_velocity": ang_accel
        }

    def _calculate_thrust(self, motor_speeds: np.ndarray) -> float:
        return self.thrust_coeff * np.sum(motor_speeds ** 2)

    def _calculate_torques(self, motor_speeds: np.ndarray) -> np.ndarray:
        l, kf, km = self.arm_length, self.thrust_coeff, self.torque_coeff
        roll_torque = l * kf * ((motor_speeds[0]**2 + motor_speeds[3]**2)
                               - (motor_speeds[1]**2 + motor_speeds[2]**2))
        pitch_torque = l * kf * ((motor_speeds[0]**2 + motor_speeds[1]**2)
                                - (motor_speeds[2]**2 + motor_speeds[3]**2))
        yaw_torque = km * (-motor_speeds[0]**2 + motor_speeds[1]**2
                           - motor_speeds[2]**2 + motor_speeds[3]**2)
        return np.array([roll_torque, pitch_torque, yaw_torque])

    def _add_derivative(self, state: UAVState, deriv: Dict, dt: float) -> UAVState:
        s = UAVState()
        s.position = state.position + deriv["position"] * dt
        s.velocity = state.velocity + deriv["velocity"] * dt
        s.orientation = state.orientation + deriv["orientation"] * dt
        s.angular_velocity = state.angular_velocity + deriv["angular_velocity"] * dt
        return s

    def _combine_derivatives(self, k1, k2, k3, k4):
        return {k: (k1[k] + 2*k2[k] + 2*k3[k] + k4[k]) / 6 for k in k1}
