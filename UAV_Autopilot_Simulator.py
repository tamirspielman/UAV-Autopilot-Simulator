"""
Advanced AI-Based UAV Autopilot Simulator
=========================================
A comprehensive flight control system combining classical control theory,
sensor fusion, and reinforcement learning for autonomous UAV operation.

Author: Advanced Autopilot System
Python 3.8+ Required

Dependencies:
pip install numpy scipy matplotlib pandas gymnasium stable-baselines3 filterpy pygame plotly dash dash-bootstrap-components
"""

import numpy as np
from scipy import signal, linalg
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import threading
import queue
from collections import deque
import time
import json
import logging
from abc import ABC, abstractmethod
import os

# For RL components - using Gymnasium instead of Gym
try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    try:
        import gym
        from gym import spaces
        HAS_GYMNASIUM = False
        print("Warning: Using legacy Gym. Consider upgrading to Gymnasium.")
    except ImportError:
        HAS_GYMNASIUM = False
        print("Warning: No Gym installation found. RL features will be limited.")

# Try to import PyTorch for RL
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not installed. RL features will be limited.")

# For visualization
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Visualization features will be limited.")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: plotly not installed. Dashboard features will be limited.")

try:
    import dash
    from dash import dcc, html, Input, Output
    HAS_DASH = True
except ImportError:
    HAS_DASH = False
    print("Warning: dash not installed. Web dashboard will not be available.")

try:
    import dash_bootstrap_components as dbc
    HAS_DBC = True
except ImportError:
    HAS_DBC = False
    print("dash_bootstrap_components not installed. Using basic Dash components.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# PART 1: CORE DATA STRUCTURES AND PHYSICS MODEL
# ============================================================================

class FlightMode(Enum):
    """Flight modes for the UAV"""
    MANUAL = "manual"
    STABILIZE = "stabilize"
    ALTITUDE_HOLD = "altitude_hold"
    POSITION_HOLD = "position_hold"
    AUTO = "auto"
    RTL = "return_to_launch"
    LAND = "land"
    AI_PILOT = "ai_pilot"


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


@dataclass
class SensorData:
    """Sensor measurements with noise"""
    imu_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    imu_gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gps_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gps_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    barometer_altitude: float = 0.0
    magnetometer: np.ndarray = field(default_factory=lambda: np.zeros(3))
    lidar_range: float = 0.0
    optical_flow: np.ndarray = field(default_factory=lambda: np.zeros(2))
    timestamp: float = 0.0


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
        new_state.orientation = self._normalize_angles(new_state.orientation)
        
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
        R = self._rotation_matrix(state.orientation)
        
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
    
    @staticmethod
    def _rotation_matrix(angles: np.ndarray) -> np.ndarray:
        """Generate rotation matrix from Euler angles"""
        roll, pitch, yaw = angles
        
        # Individual rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        return Rz @ Ry @ Rx
    
    @staticmethod
    def _normalize_angles(angles: np.ndarray) -> np.ndarray:
        """Normalize angles to [-pi, pi]"""
        return np.arctan2(np.sin(angles), np.cos(angles))
    
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


# ============================================================================
# PART 2: SENSOR MODELS AND FUSION
# ============================================================================

class SensorModel:
    """Realistic sensor models with noise characteristics"""
    
    def __init__(self):
        # IMU noise parameters
        self.accel_noise_density = 0.003  # m/s^2/sqrt(Hz)
        self.accel_bias_instability = 0.001  # m/s^2
        self.gyro_noise_density = 0.01  # rad/s/sqrt(Hz)
        self.gyro_bias_instability = 0.0001  # rad/s
        
        # GPS noise parameters
        self.gps_position_noise = 2.0  # meters
        self.gps_velocity_noise = 0.1  # m/s
        self.gps_update_rate = 10  # Hz
        
        # Barometer noise
        self.baro_noise = 0.5  # meters
        
        # Magnetometer noise
        self.mag_noise = 0.1  # Gauss
        
        # Initialize biases
        self.accel_bias = np.random.randn(3) * self.accel_bias_instability
        self.gyro_bias = np.random.randn(3) * self.gyro_bias_instability
    
    def measure(self, true_state: UAVState, dt: float) -> SensorData:
        """Generate sensor measurements from true state"""
        sensor_data = SensorData()
        
        # IMU measurements
        R = UAVDynamics._rotation_matrix(true_state.orientation)
        gravity_body = R.T @ np.array([0, 0, 9.81])
        
        sensor_data.imu_accel = (
            true_state.acceleration - gravity_body + 
            self.accel_bias + 
            np.random.randn(3) * self.accel_noise_density / np.sqrt(dt)
        )
        
        sensor_data.imu_gyro = (
            true_state.angular_velocity + 
            self.gyro_bias + 
            np.random.randn(3) * self.gyro_noise_density / np.sqrt(dt)
        )
        
        # GPS measurements (lower update rate)
        if np.random.rand() < self.gps_update_rate * dt:
            sensor_data.gps_position = (
                true_state.position + 
                np.random.randn(3) * self.gps_position_noise
            )
            sensor_data.gps_velocity = (
                true_state.velocity + 
                np.random.randn(3) * self.gps_velocity_noise
            )
        
        # Barometer
        sensor_data.barometer_altitude = (
            -true_state.position[2] +  # Convert from NED to altitude
            np.random.randn() * self.baro_noise
        )
        
        # Magnetometer (simplified - measures heading)
        mag_heading = true_state.orientation[2]  # yaw
        sensor_data.magnetometer = np.array([
            np.cos(mag_heading),
            np.sin(mag_heading),
            0
        ]) + np.random.randn(3) * self.mag_noise
        
        sensor_data.timestamp = true_state.timestamp
        
        return sensor_data
    
    def update_biases(self, dt: float):
        """Random walk for sensor biases"""
        self.accel_bias += np.random.randn(3) * self.accel_bias_instability * np.sqrt(dt)
        self.gyro_bias += np.random.randn(3) * self.gyro_bias_instability * np.sqrt(dt)


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for sensor fusion - FIXED VERSION
    """
    
    def __init__(self):
        # State vector: [position(3), velocity(3), orientation(3), accel_bias(3), gyro_bias(3)]
        self.state_dim = 15
        self.state = np.zeros(self.state_dim)
        
        # Initialize with reasonable values
        self.state[2] = -10.0  # Start at 10m altitude
        
        # Covariance matrix
        self.P = np.eye(self.state_dim) * 0.1
        
        # Process noise (tuned for UAV)
        self.Q = np.eye(self.state_dim) * 0.01
        
        # Measurement noise
        self.R_gps = np.eye(6) * 1.0
        self.R_baro = 0.25
        self.R_mag = np.eye(2) * 0.01
        
        self.last_imu_time = None

    def predict(self, imu_data: SensorData, dt: float):
        """Prediction step using IMU data - FIXED"""
        if dt <= 0:
            return
            
        # Extract current state
        pos = self.state[:3]
        vel = self.state[3:6]
        orient = self.state[6:9]
        accel_bias = self.state[9:12]
        gyro_bias = self.state[12:15]
        
        # Correct IMU measurements with bias
        accel_corrected = imu_data.imu_accel - accel_bias
        gyro_corrected = imu_data.imu_gyro - gyro_bias
        
        # Rotation matrix from body to world frame
        R = self._rotation_matrix(orient)
        
        # State prediction (simplified for stability)
        gravity_world = np.array([0, 0, 9.81])
        
        # Position update
        self.state[:3] += vel * dt + 0.5 * (R @ accel_corrected + gravity_world) * dt**2
        
        # Velocity update
        self.state[3:6] += (R @ accel_corrected + gravity_world) * dt
        
        # Orientation update (simplified)
        self.state[6:9] += gyro_corrected * dt
        
        # Normalize orientation
        self.state[6:9] = self._normalize_angles(self.state[6:9])
        
        # Simplified covariance prediction
        F = np.eye(self.state_dim)
        self.P = F @ self.P @ F.T + self.Q

    def update_gps(self, gps_data: SensorData):
        """Update step using GPS measurements - FIXED"""
        if np.linalg.norm(gps_data.gps_position) < 0.01:
            return
            
        H = np.zeros((6, self.state_dim))
        H[:3, :3] = np.eye(3)  # Position
        H[3:6, 3:6] = np.eye(3)  # Velocity
        
        z = np.concatenate([gps_data.gps_position, gps_data.gps_velocity])
        z_pred = H @ self.state
        
        # Kalman update
        innovation = z - z_pred
        S = H @ self.P @ H.T + self.R_gps
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.state += K @ innovation
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P

    def update_barometer(self, baro_altitude: float):
        """Update step using barometer - FIXED"""
        H = np.zeros((1, self.state_dim))
        H[0, 2] = -1  # altitude = -z_position
        
        innovation = baro_altitude - (H @ self.state)[0]
        
        S = H @ self.P @ H.T + self.R_baro
        K = (self.P @ H.T) / S
        
        self.state += K.flatten() * innovation
        self.P = (np.eye(self.state_dim) - np.outer(K, H)) @ self.P

    def update_magnetometer(self, mag_data: np.ndarray):
        """Update step using magnetometer - FIXED"""
        mag_yaw = np.arctan2(mag_data[1], mag_data[0])
        
        H = np.zeros((1, self.state_dim))
        H[0, 8] = 1  # Yaw component
        
        innovation = self._wrap_angle(mag_yaw - self.state[8])
        
        S = H @ self.P @ H.T + 0.01
        K = (self.P @ H.T) / S
        
        self.state += K.flatten() * innovation
        self.P = (np.eye(self.state_dim) - np.outer(K, H)) @ self.P

    def get_estimated_state(self) -> UAVState:
        """Convert filter state to UAVState"""
        state = UAVState()
        state.position = self.state[:3].copy()
        state.velocity = self.state[3:6].copy()
        state.orientation = self.state[6:9].copy()
        return state

    @staticmethod
    def _rotation_matrix(angles: np.ndarray) -> np.ndarray:
        """Generate rotation matrix from Euler angles"""
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

    @staticmethod
    def _normalize_angles(angles: np.ndarray) -> np.ndarray:
        """Normalize angles to [-pi, pi]"""
        return (angles + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi


# ============================================================================
# PART 3: CONTROL SYSTEMS (PID, LQR, MPC)
# ============================================================================

class PIDController:
    """
    PID controller with anti-windup and derivative filtering
    """
    
    def __init__(self, kp: float, ki: float, kd: float, 
                 output_limits: Tuple[float, float] = (-1, 1)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min, self.output_max = output_limits
        
        # Internal state
        self.integral = 0
        self.prev_error = 0
        self.prev_derivative = 0
        
        # Anti-windup
        self.integral_limit = 10
        
        # Derivative filter
        self.derivative_filter_alpha = 0.1
        
    def compute(self, setpoint: float, measurement: float, dt: float) -> float:
        """Compute PID output"""
        error = setpoint - measurement
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral
        
        # Derivative term with filtering
        if dt > 0:
            derivative = (error - self.prev_error) / dt
            # Low-pass filter on derivative
            derivative = (self.derivative_filter_alpha * derivative + 
                         (1 - self.derivative_filter_alpha) * self.prev_derivative)
            self.prev_derivative = derivative
            d_term = self.kd * derivative
        else:
            d_term = 0
        
        # Compute output
        output = p_term + i_term + d_term
        
        # Apply limits
        output_limited = np.clip(output, self.output_min, self.output_max)
        
        # Back-calculation anti-windup
        if output != output_limited and np.sign(error) == np.sign(output - output_limited):
            self.integral -= error * dt
        
        self.prev_error = error
        
        return output_limited
    
    def reset(self):
        """Reset controller state"""
        self.integral = 0
        self.prev_error = 0
        self.prev_derivative = 0


class LQRController:
    """
    Linear Quadratic Regulator for optimal control
    Uses continuous-time algebraic Riccati equation
    """
    
    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
        """
        A: State transition matrix
        B: Control input matrix
        Q: State cost matrix
        R: Control cost matrix
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        
        # Solve Riccati equation
        self.K = self._solve_lqr()
        
    def _solve_lqr(self) -> np.ndarray:
        """Solve the continuous-time algebraic Riccati equation"""
        # Using scipy's solve_continuous_are
        P = linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        K = np.linalg.inv(self.R) @ self.B.T @ P
        return K
    
    def compute(self, state: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Compute LQR control input"""
        error = reference - state
        return -self.K @ error


class ModelPredictiveController:
    """
    Model Predictive Control with constraints
    Solves optimization problem at each time step
    """
    
    def __init__(self, dynamics_model: UAVDynamics, horizon: int = 20):
        self.dynamics = dynamics_model
        self.horizon = horizon
        self.dt = 0.01
        
        # Cost weights
        self.Q = np.diag([10, 10, 10, 1, 1, 1])  # State cost
        self.R = np.diag([0.1, 0.1, 0.1, 0.1])  # Control cost
        
        # Constraints
        self.u_min = np.array([0, -1, -1, -1])
        self.u_max = np.array([1, 1, 1, 1])
        
    def compute(self, current_state: UAVState, reference_trajectory: List[UAVState]) -> np.ndarray:
        """
        Compute MPC control input
        Simplified version - in practice would use optimization solver
        """
        # Extract relevant states
        x = np.concatenate([
            current_state.position,
            current_state.velocity
        ])
        
        # Simple receding horizon control
        # For demonstration - real MPC would solve optimization problem
        if len(reference_trajectory) > 0:
            ref = reference_trajectory[0]
            x_ref = np.concatenate([ref.position, ref.velocity])
            
            # Simplified feedback control
            error = x_ref - x
            control = np.array([
                0.5 + 0.1 * error[2],  # Throttle
                0.1 * error[1],         # Roll
                -0.1 * error[0],        # Pitch
                0.0                     # Yaw
            ])
            
            return np.clip(control, self.u_min, self.u_max)
        
        return np.array([0.5, 0, 0, 0])  # Hover


# ============================================================================
# PART 4: REINFORCEMENT LEARNING AUTOPILOT
# ============================================================================

class UAVEnvironment:
    """
    Custom UAV environment for control
    Simplified version without Gym dependency
    """
    
    def __init__(self):
        # Initialize dynamics and sensors
        self.dynamics = UAVDynamics()
        self.sensor_model = SensorModel()
        self.dt = 0.01
        
        # State and action spaces
        self.observation_shape = (18,)
        self.action_shape = (4,)
        self.action_low = np.array([0, -1, -1, -1])
        self.action_high = np.array([1, 1, 1, 1])
        
        # Episode parameters
        self.max_steps = 1000
        self.current_step = 0
        
        # Target position for training
        self.target_position = np.array([0, 0, -10])  # 10m altitude
        
        # Track previous action for smoothness penalty
        self.prev_action = None
        
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.state = UAVState()
        self.state.position = np.array([0, 0, -1])  # Start 1m above ground
        self.current_step = 0
        self.prev_action = None
        return self._get_observation()
    
    def step(self, action: np.ndarray):
        """Execute one environment step"""
        # Store action for reward calculation
        current_action = action.copy()
        
        # Apply action
        self.state = self.dynamics.update(self.state, action, self.dt)
        
        # Get observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(current_action)
        
        # Check termination
        done = self._is_done()
        
        self.current_step += 1
        
        info = {'state': self.state.to_dict()}
        
        return obs, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get observation vector from state"""
        sensor_data = self.sensor_model.measure(self.state, self.dt)
        
        obs = np.concatenate([
            self.state.position,
            self.state.velocity,
            self.state.orientation,
            self.state.angular_velocity,
            sensor_data.imu_accel,
            sensor_data.imu_gyro
        ])
        
        return obs.astype(np.float32)
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward for RL training"""
        # Position error
        pos_error = np.linalg.norm(self.state.position - self.target_position)
        pos_reward = np.exp(-0.1 * pos_error)
        
        # Velocity penalty (encourage stable hover)
        vel_penalty = -0.01 * np.linalg.norm(self.state.velocity)
        
        # Orientation penalty (encourage level flight)
        orient_penalty = -0.1 * np.linalg.norm(self.state.orientation[:2])  # roll and pitch
        
        # Control smoothness penalty
        if self.prev_action is not None:
            action_diff = np.linalg.norm(self.prev_action - action)
            smooth_penalty = -0.01 * action_diff
        else:
            smooth_penalty = 0
            
        self.prev_action = action.copy()
        
        # Total reward
        total_reward = pos_reward + vel_penalty + orient_penalty + smooth_penalty
        
        # Bonus for reaching target
        if pos_error < 0.5:
            total_reward += 10
            
        return total_reward
    
    def _is_done(self) -> bool:
        """Check if episode should terminate"""
        # Check if crashed (too low altitude)
        if self.state.position[2] > -0.1:  # Too close to ground
            return True
            
        # Check if too far from target
        if np.linalg.norm(self.state.position) > 50:
            return True
            
        # Check max steps
        if self.current_step >= self.max_steps:
            return True
            
        return False


if HAS_TORCH:
    class PPONetwork(nn.Module):
        """Neural network for PPO policy and value functions"""
        
        def __init__(self, obs_dim: int, action_dim: int):
            super().__init__()
            
            # Shared feature extractor
            self.shared_net = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
            )
            
            # Policy head
            self.policy_net = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
                nn.Tanh()  # Actions in [-1, 1]
            )
            
            # Value head
            self.value_net = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            
            # Log standard deviation for continuous actions
            self.log_std = nn.Parameter(torch.zeros(1, action_dim))
            
        def forward(self, x):
            features = self.shared_net(x)
            action_mean = self.policy_net(features)
            value = self.value_net(features)
            return action_mean, value


class RLAutopilot:
    """
    Reinforcement Learning based autopilot using PPO
    Can be trained to perform complex maneuvers
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.env = UAVEnvironment()
        self.policy = None
        
        if HAS_TORCH and model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize untrained policy
            if HAS_TORCH:
                self.policy = PPONetwork(
                    self.env.observation_shape[0],
                    self.env.action_shape[0]
                )
            
    def compute_control(self, observation: np.ndarray) -> np.ndarray:
        """Compute control action from observation"""
        if self.policy is None or not HAS_TORCH:
            # Return hover command if no policy loaded or no PyTorch
            return np.array([0.5, 0, 0, 0])
            
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            action_mean, _ = self.policy(obs_tensor)
            action = action_mean.squeeze(0).numpy()
            
        # Scale to appropriate control range
        action = np.clip(action, self.env.action_low, self.env.action_high)
        return action
    
    def train(self, total_timesteps: int = 10000):
        """Train the RL agent (simplified training loop)"""
        if not HAS_TORCH:
            print("PyTorch not available. Cannot train RL agent.")
            return
            
        logger.info("Starting RL training...")
        
        # Simplified training loop
        for episode in range(total_timesteps // 1000):
            obs = self.env.reset()
            episode_reward = 0
            
            for step in range(1000):
                action = self.compute_control(obs)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                
                if done:
                    break
                    
            if episode % 10 == 0:
                logger.info(f"Episode {episode}, Reward: {episode_reward:.2f}")
                
    def save_model(self, path: str):
        """Save trained model"""
        if self.policy is not None and HAS_TORCH:
            torch.save(self.policy.state_dict(), path)
        
    def load_model(self, path: str):
        """Load trained model"""
        if HAS_TORCH:
            self.policy = PPONetwork(
                self.env.observation_shape[0],
                self.env.action_shape[0]
            )
            self.policy.load_state_dict(torch.load(path))
            self.policy.eval()


# ============================================================================
# PART 5: FLIGHT CONTROLLER AND MISSION MANAGEMENT - FIXED
# ============================================================================

class FlightController:
    """
    Main flight controller that combines all control methods
    and manages flight modes - FIXED VERSION
    """
    
    def __init__(self):
        # Initialize components
        self.dynamics = UAVDynamics()
        self.sensor_model = SensorModel()
        self.ekf = ExtendedKalmanFilter()
        
        # Improved PID gains for better trajectory tracking
        self.altitude_pid = PIDController(3.0, 0.2, 0.8, (0.2, 0.8))
        self.roll_pid = PIDController(4.0, 0.1, 0.3, (-0.5, 0.5))
        self.pitch_pid = PIDController(4.0, 0.1, 0.3, (-0.5, 0.5))
        self.yaw_pid = PIDController(2.0, 0.05, 0.2, (-0.3, 0.3))
        
        # RL autopilot
        self.rl_autopilot = RLAutopilot()
        
        # Current state
        self.state = UAVState()
        self.estimated_state = UAVState()
        self.sensor_data = SensorData()
        
        # Flight mode and setpoints
        self.flight_mode = FlightMode.STABILIZE
        self.setpoints = {
            'altitude': -10.0,  # 10m altitude (NED coordinates)
            'position': np.array([0, 0, -10]),
            'velocity': np.zeros(3),
            'attitude': np.zeros(3)
        }
        
        # Mission waypoints
        self.waypoints = []
        self.current_waypoint_index = 0
        self.mission_complete = False
        
        # Waypoint following parameters
        self.waypoint_acceptance_radius = 1.0  # meters
        self.cruise_altitude = -15.0  # meters (NED coordinates)
        
        # Control outputs
        self.control_output = np.array([0.5, 0, 0, 0])  # hover
        
        # Timing
        self.dt = 0.01
        self.last_update = time.time()
        
        # Data logging
        self.telemetry_history = deque(maxlen=1000)
        self.control_history = deque(maxlen=1000)
        
    def update(self, manual_control: Optional[np.ndarray] = None) -> UAVState:
        """Main update loop - runs at fixed timestep"""
        current_time = time.time()
        dt = current_time - self.last_update
        
        if dt >= self.dt:
            # Generate sensor data
            self.sensor_data = self.sensor_model.measure(self.state, dt)
            self.sensor_model.update_biases(dt)
            
            # Sensor fusion
            self.ekf.predict(self.sensor_data, dt)
            if np.linalg.norm(self.sensor_data.gps_position) > 0.01:
                self.ekf.update_gps(self.sensor_data)
            self.ekf.update_barometer(self.sensor_data.barometer_altitude)
            self.ekf.update_magnetometer(self.sensor_data.magnetometer)
            
            self.estimated_state = self.ekf.get_estimated_state()
            
            # Compute control based on flight mode
            if manual_control is not None and self.flight_mode == FlightMode.MANUAL:
                self.control_output = manual_control
            else:
                self.control_output = self._compute_autonomous_control()
            
            # Apply control and update dynamics
            self.state = self.dynamics.update(self.state, self.control_output, dt)
            
            # Update mission waypoints
            self._update_mission()
            
            # Log telemetry
            self._log_telemetry()
            
            self.last_update = current_time
        
        return self.state
    
    def _compute_autonomous_control(self) -> np.ndarray:
        """Compute autonomous control based on flight mode"""
        if self.flight_mode == FlightMode.AI_PILOT:
            return self._compute_rl_control()
        elif self.flight_mode == FlightMode.ALTITUDE_HOLD:
            return self._compute_altitude_hold_control()
        elif self.flight_mode == FlightMode.POSITION_HOLD:
            return self._compute_position_hold_control()
        elif self.flight_mode == FlightMode.AUTO:
            return self._compute_auto_control()
        elif self.flight_mode == FlightMode.RTL:
            return self._compute_rtl_control()
        elif self.flight_mode == FlightMode.LAND:
            return self._compute_land_control()
        else:  # STABILIZE
            return self._compute_stabilize_control()
    
    def _compute_rl_control(self) -> np.ndarray:
        """Compute control using RL autopilot"""
        # Get observation for RL
        obs = np.concatenate([
            self.estimated_state.position,
            self.estimated_state.velocity,
            self.estimated_state.orientation,
            self.sensor_data.imu_accel,
            self.sensor_data.imu_gyro
        ])
        
        return self.rl_autopilot.compute_control(obs)
    
    def _compute_altitude_hold_control(self) -> np.ndarray:
        """PID-based altitude hold control"""
        # Altitude control (z-axis in NED)
        altitude_error = self.setpoints['altitude'] - self.estimated_state.position[2]
        throttle = self.altitude_pid.compute(0, altitude_error, self.dt)
        
        # Attitude stabilization
        roll_stabilize = self.roll_pid.compute(0, self.estimated_state.orientation[0], self.dt)
        pitch_stabilize = self.pitch_pid.compute(0, self.estimated_state.orientation[1], self.dt)
        yaw_stabilize = self.yaw_pid.compute(0, self.estimated_state.orientation[2], self.dt)
        
        return np.array([throttle, roll_stabilize, pitch_stabilize, yaw_stabilize])
    
    def _compute_position_hold_control(self) -> np.ndarray:
        """Position hold with improved stability"""
        # Position to velocity cascade
        pos_error = self.setpoints['position'] - self.estimated_state.position
        desired_velocity = np.clip(pos_error * 0.8, -2.5, 2.5)
        
        # Velocity to attitude cascade
        vel_error = desired_velocity - self.estimated_state.velocity
        desired_pitch = np.clip(vel_error[0] * 0.25, -0.35, 0.35)
        desired_roll = np.clip(-vel_error[1] * 0.25, -0.35, 0.35)
        
        # Altitude control
        altitude_error = self.setpoints['altitude'] - self.estimated_state.position[2]
        throttle = self.altitude_pid.compute(0, altitude_error, self.dt)
        
        # Attitude control
        roll = self.roll_pid.compute(desired_roll, self.estimated_state.orientation[0], self.dt)
        pitch = self.pitch_pid.compute(desired_pitch, self.estimated_state.orientation[1], self.dt)
        yaw = self.yaw_pid.compute(0, self.estimated_state.orientation[2], self.dt)
        
        return np.array([throttle, roll, pitch, yaw])
    
    def _compute_auto_control(self) -> np.ndarray:
        """Auto mode - follow waypoints with improved trajectory tracking"""
        if not self.waypoints:
            return self._compute_position_hold_control()
        
        current_wp = self.waypoints[self.current_waypoint_index]
        
        # Calculate desired velocity vector towards waypoint
        pos_error = current_wp - self.estimated_state.position
        horizontal_error = np.linalg.norm(pos_error[:2])
        
        # Normalize and scale desired velocity
        if horizontal_error > 0.1:
            desired_velocity_xy = (pos_error[:2] / horizontal_error) * min(horizontal_error * 0.8, 3.0)
        else:
            desired_velocity_xy = np.zeros(2)
        
        # Altitude control
        altitude_error = current_wp[2] - self.estimated_state.position[2]
        desired_velocity_z = np.clip(altitude_error * 0.5, -1.0, 1.0)
        
        desired_velocity = np.array([desired_velocity_xy[0], desired_velocity_xy[1], desired_velocity_z])
        
        # Convert to attitude commands
        desired_pitch = np.clip(desired_velocity[0] * 0.3, -0.4, 0.4)
        desired_roll = np.clip(-desired_velocity[1] * 0.3, -0.4, 0.4)
        
        # Altitude control
        throttle = self.altitude_pid.compute(current_wp[2], self.estimated_state.position[2], self.dt)
        
        # Attitude control
        roll = self.roll_pid.compute(desired_roll, self.estimated_state.orientation[0], self.dt)
        pitch = self.pitch_pid.compute(desired_pitch, self.estimated_state.orientation[1], self.dt)
        yaw = self.yaw_pid.compute(0, self.estimated_state.orientation[2], self.dt)  # Maintain heading
        
        control_output = np.array([throttle, roll, pitch, yaw])
        
        # Check if waypoint reached
        distance = np.linalg.norm(self.estimated_state.position - current_wp)
        if distance < self.waypoint_acceptance_radius:
            self.current_waypoint_index += 1
            if self.current_waypoint_index >= len(self.waypoints):
                self.mission_complete = True
                self.current_waypoint_index = len(self.waypoints) - 1
                logger.info("Mission complete! All waypoints reached.")
            else:
                logger.info(f"Waypoint {self.current_waypoint_index-1} reached. Moving to next waypoint.")
        
        return control_output
    
    def _compute_rtl_control(self) -> np.ndarray:
        """Return to launch control"""
        # Set target to origin
        self.setpoints['position'] = np.array([0, 0, self.setpoints['altitude']])
        
        distance = np.linalg.norm(self.estimated_state.position[:2])  # Horizontal distance
        if distance < 1.0 and abs(self.estimated_state.position[2] - self.setpoints['altitude']) < 0.5:
            self.set_flight_mode(FlightMode.LAND)
        
        return self._compute_position_hold_control()
    
    def _compute_land_control(self) -> np.ndarray:
        """Automatic landing control"""
        # Gradually reduce altitude
        current_alt = -self.estimated_state.position[2]  # Convert to altitude
        if current_alt > 0.5:
            self.setpoints['altitude'] = min(self.setpoints['altitude'] + 0.02, -0.5)  # Descend slowly
        else:
            # On ground - cut motors
            return np.array([0, 0, 0, 0])
        
        return self._compute_altitude_hold_control()
    
    def _compute_stabilize_control(self) -> np.ndarray:
        """Stabilize mode - maintain level attitude"""
        roll = self.roll_pid.compute(0, self.estimated_state.orientation[0], self.dt)
        pitch = self.pitch_pid.compute(0, self.estimated_state.orientation[1], self.dt)
        yaw = self.yaw_pid.compute(0, self.estimated_state.orientation[2], self.dt)
        
        # Maintain hover throttle
        throttle = 0.5
        
        return np.array([throttle, roll, pitch, yaw])
    
    def _update_mission(self):
        """Update mission state and waypoints"""
        # Mission logic here
        pass
    
    def _log_telemetry(self):
        """Log telemetry data for analysis"""
        telemetry = {
            'timestamp': time.time(),
            'state': self.state.to_dict(),
            'estimated_state': self.estimated_state.to_dict(),
            'sensor_data': {
                'imu_accel': self.sensor_data.imu_accel.tolist(),
                'imu_gyro': self.sensor_data.imu_gyro.tolist(),
                'gps_position': self.sensor_data.gps_position.tolist(),
                'baro_altitude': self.sensor_data.barometer_altitude
            },
            'control_output': self.control_output.tolist(),
            'flight_mode': self.flight_mode.value
        }
        self.telemetry_history.append(telemetry)
        self.control_history.append(self.control_output)
    
    def set_flight_mode(self, mode: FlightMode):
        """Change flight mode and reset controllers if needed"""
        if mode != self.flight_mode:
            self.altitude_pid.reset()
            self.roll_pid.reset()
            self.pitch_pid.reset()
            self.yaw_pid.reset()
            self.flight_mode = mode
            logger.info(f"Flight mode changed to: {mode.value}")
    
    def set_waypoints(self, waypoints: List[np.ndarray]):
        """Set mission waypoints"""
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        self.mission_complete = False
        logger.info(f"Mission set with {len(waypoints)} waypoints")
    
    def get_telemetry(self) -> Dict:
        """Get current telemetry data"""
        return {
            'position': self.estimated_state.position.tolist(),
            'velocity': self.estimated_state.velocity.tolist(),
            'attitude': self.estimated_state.orientation.tolist(),
            'flight_mode': self.flight_mode.value,
            'battery': 85.0,  # Simulated
            'gps_fix': True,
            'waypoint_index': self.current_waypoint_index,
            'mission_complete': self.mission_complete
        }# ============================================================================
# PART 6: INTERACTIVE DASHBOARD - FIXED INTERVAL COMPONENT
# ============================================================================

if HAS_DASH:
    class UAVDashboard:
        """
        Interactive dashboard for real-time monitoring and control
        Fixed Interval component placement
        """
        
        def __init__(self, flight_controller: FlightController):
            self.fc = flight_controller
            self.app = dash.Dash(__name__)
                
            self.setup_layout()
            self.setup_callbacks()
            
            # Data buffers for plotting
            self.position_history = deque(maxlen=200)
            self.attitude_history = deque(maxlen=200)
            self.control_history = deque(maxlen=200)
            self.start_time = time.time()
            
        def setup_layout(self):
            """Setup the dashboard layout using basic Dash components"""
            
            # Custom CSS for styling
            styles = {
                'container': {'backgroundColor': '#1a1a1a', 'minHeight': '100vh', 'padding': '20px'},
                'header': {'color': '#00ff00', 'fontWeight': 'bold', 'textAlign': 'center'},
                'card': {'backgroundColor': '#2c3e50', 'border': '1px solid #00ff00', 'borderRadius': '5px', 'padding': '15px', 'marginBottom': '15px'},
                'cardHeader': {'color': 'white', 'fontWeight': 'bold', 'borderBottom': '1px solid #00ff00', 'paddingBottom': '10px', 'marginBottom': '10px'},
                'button': {'backgroundColor': '#3498db', 'color': 'white', 'border': 'none', 'padding': '10px', 'margin': '5px', 'borderRadius': '3px', 'width': '100%'},
                'buttonSuccess': {'backgroundColor': '#27ae60', 'color': 'white', 'border': 'none', 'padding': '10px', 'margin': '5px', 'borderRadius': '3px', 'width': '100%'},
                'buttonWarning': {'backgroundColor': '#f39c12', 'color': 'white', 'border': 'none', 'padding': '10px', 'margin': '5px', 'borderRadius': '3px', 'width': '100%'},
                'buttonDanger': {'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none', 'padding': '10px', 'margin': '5px', 'borderRadius': '3px', 'width': '100%'},
                'dropdown': {'backgroundColor': '#34495e', 'color': 'white', 'border': '1px solid #00ff00'},
                'table': {'color': 'white', 'width': '100%', 'borderCollapse': 'collapse'},
                'tableHeader': {'backgroundColor': '#2c3e50', 'color': '#00ff00', 'padding': '8px', 'border': '1px solid #00ff00'},
                'tableCell': {'padding': '8px', 'border': '1px solid #00ff00'}
            }
            
            self.app.layout = html.Div([
                # Header
                html.Div([
                    html.H1("AI-Based UAV Autopilot Simulator", style=styles['header']),
                    html.H4("Real-time Flight Control System", style={'color': '#cccccc', 'textAlign': 'center'})
                ]),
                
                html.Hr(style={'borderColor': '#00ff00', 'margin': '20px 0'}),
                
                html.Div([
                    # Left sidebar - Controls
                    html.Div([
                        # Flight Controls Card
                        html.Div([
                            html.H4("🚀 Flight Controls", style=styles['cardHeader']),
                            
                            # Flight Mode Selection
                            html.Div([
                                html.Label("Flight Mode", style={'color': 'white', 'marginBottom': '5px'}),
                                dcc.Dropdown(
                                    id='flight-mode-dropdown',
                                    options=[
                                        {'label': '🛸 MANUAL', 'value': 'manual'},
                                        {'label': '⚡ STABILIZE', 'value': 'stabilize'},
                                        {'label': '📊 ALTITUDE HOLD', 'value': 'altitude_hold'},
                                        {'label': '🎯 POSITION HOLD', 'value': 'position_hold'},
                                        {'label': '🔄 AUTO MISSION', 'value': 'auto'},
                                        {'label': '🏠 RETURN TO LAUNCH', 'value': 'return_to_launch'},
                                        {'label': '🛬 LAND', 'value': 'land'},
                                        {'label': '🤖 AI PILOT', 'value': 'ai_pilot'}
                                    ],
                                    value='stabilize',
                                    style=styles['dropdown']
                                )
                            ], style={'marginBottom': '15px'}),
                            
                            # Altitude Control
                            html.Div([
                                html.Label("Altitude Setpoint (m)", style={'color': 'white', 'marginBottom': '5px'}),
                                dcc.Slider(
                                    id='altitude-slider',
                                    min=1, 
                                    max=100, 
                                    step=1, 
                                    value=10,
                                    marks={i: str(i) for i in range(0, 101, 20)},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ], style={'marginBottom': '20px'}),
                            
                            # Quick Action Buttons
                            html.Div([
                                html.Label("Quick Actions", style={'color': 'white', 'marginBottom': '10px'}),
                                html.Div([
                                    html.Button("🚀 Takeoff", id='takeoff-btn', style=styles['buttonSuccess']),
                                    html.Button("🛬 Land", id='land-btn', style=styles['buttonWarning'])
                                ], style={'display': 'flex', 'gap': '10px'}),
                                html.Div([
                                    html.Button("🏠 RTL", id='rtl-btn', style=styles['buttonDanger']),
                                    html.Button("⏸️ Pause", id='pause-btn', style=styles['button'])
                                ], style={'display': 'flex', 'gap': '10px'})
                            ])
                        ], style=styles['card']),
                        
                        # System Status Card
                        html.Div([
                            html.H4("📊 System Status", style=styles['cardHeader']),
                            html.Div([
                                html.Div([
                                    html.Div("UAV Status", style={'color': '#cccccc', 'fontSize': '12px'}),
                                    html.Div("🟢 ONLINE", id="uav-status", style={'color': '#00ff00', 'fontWeight': 'bold', 'fontSize': '16px'})
                                ], style={'marginBottom': '10px'}),
                                html.Div([
                                    html.Div("GPS Fix", style={'color': '#cccccc', 'fontSize': '12px'}),
                                    html.Div("🟢 3D FIX", id="gps-status", style={'color': '#00ff00', 'fontWeight': 'bold', 'fontSize': '16px'})
                                ], style={'marginBottom': '10px'}),
                                html.Div([
                                    html.Div("Battery", style={'color': '#cccccc', 'fontSize': '12px'}),
                                    html.Div("🔋 85%", id="battery-status", style={'color': '#ffa500', 'fontWeight': 'bold', 'fontSize': '16px'})
                                ], style={'marginBottom': '10px'}),
                                html.Div([
                                    html.Div("Sensors", style={'color': '#cccccc', 'fontSize': '12px'}),
                                    html.Div("🟢 OK", id="sensor-status", style={'color': '#00ff00', 'fontWeight': 'bold', 'fontSize': '16px'})
                                ])
                            ])
                        ], style=styles['card']),
                        
                        # Mission Planning Card
                        html.Div([
                            html.H4("🎯 Mission Planning", style=styles['cardHeader']),
                            html.Div([
                                html.Label("Waypoints (JSON format)", style={'color': 'white', 'marginBottom': '5px'}),
                                dcc.Textarea(
                                    id='waypoint-input',
                                    value='[[0, 0, -10], [15, 0, -15], [15, 15, -20], [0, 15, -15], [0, 0, -10]]',
                                    style={'width': '100%', 'height': '100px', 'fontFamily': 'monospace', 'backgroundColor': '#34495e', 'color': 'white', 'border': '1px solid #00ff00'},
                                    placeholder='Enter waypoints as: [[x1,y1,z1],[x2,y2,z2],...]'
                                ),
                            ], style={'marginBottom': '15px'}),
                            html.Div([
                                html.Button("📤 Upload Mission", id='upload-mission-btn', style=styles['button']),
                                html.Button("🗑️ Clear", id='clear-mission-btn', style={**styles['button'], 'backgroundColor': '#95a5a6'})
                            ], style={'display': 'flex', 'gap': '10px'}),
                            html.Div(id='mission-status', style={'marginTop': '15px'})
                        ], style=styles['card']),
                        
                        # Telemetry Display Card
                        html.Div([
                            html.H4("📡 Real-time Telemetry", style=styles['cardHeader']),
                            html.Div(id='telemetry-display', children=[
                                html.Div("Waiting for telemetry data...", style={'color': '#cccccc', 'textAlign': 'center'})
                            ])
                        ], style=styles['card'])
                    ], style={'width': '30%', 'float': 'left', 'paddingRight': '15px'}),
                    
                    # Main content - Visualizations
                    html.Div([
                        # Visualization Tabs
                        html.Div([
                            html.H4("📊 Visualizations", style=styles['cardHeader']),
                            dcc.Tabs(id="visualization-tabs", value='tab-3d', children=[
                                dcc.Tab(label='3D Trajectory', value='tab-3d'),
                                dcc.Tab(label='Position', value='tab-position'),
                                dcc.Tab(label='Attitude', value='tab-attitude'),
                                dcc.Tab(label='Controls', value='tab-controls'),
                            ], style={'marginBottom': '15px'}),
                            html.Div(id="visualization-content", style={'height': '500px'})
                        ], style=styles['card']),
                        
                        # Performance Metrics
                        html.Div([
                            html.H4("⚡ Performance Metrics", style=styles['cardHeader']),
                            html.Div([
                                html.Div([
                                    html.Div("Update Rate", style={'color': '#cccccc', 'textAlign': 'center', 'fontSize': '12px'}),
                                    html.Div("100 Hz", id="update-rate", style={'color': '#00ff00', 'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '20px'})
                                ], style={'display': 'inline-block', 'width': '24%', 'textAlign': 'center'}),
                                html.Div([
                                    html.Div("CPU Usage", style={'color': '#cccccc', 'textAlign': 'center', 'fontSize': '12px'}),
                                    html.Div("2.5%", id="cpu-usage", style={'color': '#00ff00', 'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '20px'})
                                ], style={'display': 'inline-block', 'width': '24%', 'textAlign': 'center'}),
                                html.Div([
                                    html.Div("Memory", style={'color': '#cccccc', 'textAlign': 'center', 'fontSize': '12px'}),
                                    html.Div("45 MB", id="memory-usage", style={'color': '#00ff00', 'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '20px'})
                                ], style={'display': 'inline-block', 'width': '24%', 'textAlign': 'center'}),
                                html.Div([
                                    html.Div("Real-time", style={'color': '#cccccc', 'textAlign': 'center', 'fontSize': '12px'}),
                                    html.Div("1.0x", id="rt-factor", style={'color': '#00ff00', 'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '20px'})
                                ], style={'display': 'inline-block', 'width': '24%', 'textAlign': 'center'})
                            ])
                        ], style=styles['card'])
                    ], style={'width': '68%', 'float': 'right'})
                ], style={'overflow': 'hidden'}),  # Clear float
                
                # Update Interval - FIXED: Just include it in the layout, don't call it
                dcc.Interval(
                    id='update-interval',
                    interval=200,  # Update every 200ms (5Hz)
                    n_intervals=0
                )
                
            ], style=styles['container'])
        
        def setup_callbacks(self):
            """Setup all dashboard callbacks - FIXED: No Interval call here"""
            
            # Visualization tab content
            @self.app.callback(
                Output('visualization-content', 'children'),
                [Input('visualization-tabs', 'value'),
                 Input('update-interval', 'n_intervals')]
            )
            def update_visualization(active_tab, n):
                if active_tab == "tab-3d":
                    return dcc.Graph(
                        id='3d-plot',
                        figure=self._create_3d_trajectory(),
                        config={'displayModeBar': True}
                    )
                elif active_tab == "tab-position":
                    return dcc.Graph(
                        id='position-plot',
                        figure=self._create_position_plot(),
                        config={'displayModeBar': True}
                    )
                elif active_tab == "tab-attitude":
                    return dcc.Graph(
                        id='attitude-plot', 
                        figure=self._create_attitude_plot(),
                        config={'displayModeBar': True}
                    )
                elif active_tab == "tab-controls":
                    return dcc.Graph(
                        id='control-plot',
                        figure=self._create_control_plot(),
                        config={'displayModeBar': True}
                    )
                return html.Div("Select a visualization tab")
            
            # Update all dynamic content
            @self.app.callback(
                [Output('telemetry-display', 'children'),
                 Output('update-rate', 'children'),
                 Output('cpu-usage', 'children'),
                 Output('memory-usage', 'children'),
                 Output('rt-factor', 'children'),
                 Output('uav-status', 'children'),
                 Output('gps-status', 'children'),
                 Output('battery-status', 'children'),
                 Output('sensor-status', 'children')],
                [Input('update-interval', 'n_intervals')]
            )
            def update_all_content(n):
                try:
                    # Get telemetry data
                    telemetry = self.fc.get_telemetry()
                    
                    # Update data buffers
                    current_time = time.time()
                    self.position_history.append({
                        'time': current_time,
                        'x': telemetry['position'][0],
                        'y': telemetry['position'][1], 
                        'z': telemetry['position'][2]
                    })
                    
                    # Create telemetry display table
                    telemetry_display = html.Table([
                        html.Tr([
                            html.Th("Parameter", style={'backgroundColor': '#2c3e50', 'color': '#00ff00', 'padding': '8px', 'border': '1px solid #00ff00'}),
                            html.Th("Value", style={'backgroundColor': '#2c3e50', 'color': '#00ff00', 'padding': '8px', 'border': '1px solid #00ff00'})
                        ]),
                        html.Tr([
                            html.Td("Position (N,E,D)", style={'padding': '8px', 'border': '1px solid #00ff00', 'color': 'white'}),
                            html.Td(f"{telemetry['position'][0]:.1f}, {telemetry['position'][1]:.1f}, {telemetry['position'][2]:.1f}", 
                                   style={'padding': '8px', 'border': '1px solid #00ff00', 'color': '#00ff00'})
                        ]),
                        html.Tr([
                            html.Td("Velocity (m/s)", style={'padding': '8px', 'border': '1px solid #00ff00', 'color': 'white'}),
                            html.Td(f"{telemetry['velocity'][0]:.1f}, {telemetry['velocity'][1]:.1f}, {telemetry['velocity'][2]:.1f}", 
                                   style={'padding': '8px', 'border': '1px solid #00ff00', 'color': '#00ff00'})
                        ]),
                        html.Tr([
                            html.Td("Attitude (R,P,Y)", style={'padding': '8px', 'border': '1px solid #00ff00', 'color': 'white'}),
                            html.Td(f"{np.degrees(telemetry['attitude'][0]):.1f}°, {np.degrees(telemetry['attitude'][1]):.1f}°, {np.degrees(telemetry['attitude'][2]):.1f}°", 
                                   style={'padding': '8px', 'border': '1px solid #00ff00', 'color': '#00ff00'})
                        ]),
                        html.Tr([
                            html.Td("Flight Mode", style={'padding': '8px', 'border': '1px solid #00ff00', 'color': 'white'}),
                            html.Td(telemetry['flight_mode'].upper(), 
                                   style={'padding': '8px', 'border': '1px solid #00ff00', 'color': '#00ff00'})
                        ]),
                        html.Tr([
                            html.Td("Waypoint Progress", style={'padding': '8px', 'border': '1px solid #00ff00', 'color': 'white'}),
                            html.Td(f"{telemetry['waypoint_index'] + 1}/{len(self.fc.waypoints) if self.fc.waypoints else 0}", 
                                   style={'padding': '8px', 'border': '1px solid #00ff00', 'color': '#00ff00'})
                        ]),
                        html.Tr([
                            html.Td("Mission Status", style={'padding': '8px', 'border': '1px solid #00ff00', 'color': 'white'}),
                            html.Td("COMPLETE" if telemetry['mission_complete'] else "ACTIVE", 
                                   style={'padding': '8px', 'border': '1px solid #00ff00', 'color': '#00ff00'})
                        ]),
                    ], style={'width': '100%', 'borderCollapse': 'collapse'})
                    
                    # Performance metrics (simulated)
                    update_rate = f"{(1/self.fc.dt):.0f} Hz"
                    cpu_usage = f"{2.5 + np.random.uniform(-0.5, 0.5):.1f}%"
                    memory_usage = f"{45 + np.random.uniform(-2, 2):.0f} MB"
                    rt_factor = f"{0.98 + np.random.uniform(0, 0.04):.2f}x"
                    
                    # System status
                    uav_status = "🟢 ONLINE"
                    gps_status = "🟢 3D FIX" if telemetry['gps_fix'] else "🔴 NO FIX"
                    battery_status = f"🔋 {telemetry['battery']}%"
                    sensor_status = "🟢 OK"
                    
                    return (telemetry_display, update_rate, cpu_usage, memory_usage, 
                           rt_factor, uav_status, gps_status, battery_status, sensor_status)
                    
                except Exception as e:
                    logger.error(f"Error updating dashboard: {e}")
                    error_msg = html.Div(f"Error: {str(e)}", style={'color': 'red'})
                    return (error_msg, "N/A", "N/A", "N/A", "N/A", "🔴 ERROR", "🔴 ERROR", "🔴 ERROR", "🔴 ERROR")
            
            # Mission control callbacks
            @self.app.callback(
                Output('mission-status', 'children'),
                [Input('upload-mission-btn', 'n_clicks'),
                 Input('clear-mission-btn', 'n_clicks')],
                [dash.dependencies.State('waypoint-input', 'value')]
            )
            def handle_mission_controls(upload_clicks, clear_clicks, waypoints_text):
                ctx = dash.callback_context
                if not ctx.triggered:
                    return ""
                
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                
                try:
                    if button_id == 'upload-mission-btn' and waypoints_text:
                        waypoints = json.loads(waypoints_text)
                        waypoints_np = [np.array(wp) for wp in waypoints]
                        self.fc.set_waypoints(waypoints_np)
                        return html.Div(
                            f"✅ Mission uploaded with {len(waypoints)} waypoints!", 
                            style={'color': '#27ae60', 'padding': '10px', 'border': '1px solid #27ae60', 'borderRadius': '3px'}
                        )
                    
                    elif button_id == 'clear-mission-btn':
                        self.fc.waypoints = []
                        self.fc.current_waypoint_index = 0
                        self.fc.mission_complete = False
                        return html.Div(
                            "🗑️ Mission cleared!", 
                            style={'color': '#3498db', 'padding': '10px', 'border': '1px solid #3498db', 'borderRadius': '3px'}
                        )
                        
                except Exception as e:
                    return html.Div(
                        f"❌ Error: {str(e)}", 
                        style={'color': '#e74c3c', 'padding': '10px', 'border': '1px solid #e74c3c', 'borderRadius': '3px'}
                    )
                
                return ""
            
            # Flight mode and control buttons
            @self.app.callback(
                Output('flight-mode-dropdown', 'value'),
                [Input('takeoff-btn', 'n_clicks'),
                 Input('land-btn', 'n_clicks'),
                 Input('rtl-btn', 'n_clicks'),
                 Input('pause-btn', 'n_clicks'),
                 Input('flight-mode-dropdown', 'value')]
            )
            def handle_flight_controls(takeoff_clicks, land_clicks, rtl_clicks, pause_clicks, selected_mode):
                ctx = dash.callback_context
                
                # Handle manual flight mode selection
                if ctx.triggered and ctx.triggered[0]['prop_id'] == 'flight-mode-dropdown.value':
                    if selected_mode:
                        try:
                            mode = FlightMode(selected_mode)
                            self.fc.set_flight_mode(mode)
                            logger.info(f"Flight mode changed to: {selected_mode}")
                        except ValueError as e:
                            logger.error(f"Invalid flight mode: {selected_mode}")
                    return selected_mode
                
                # Handle button clicks
                if ctx.triggered:
                    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                    
                    if button_id == 'takeoff-btn':
                        self.fc.set_flight_mode(FlightMode.ALTITUDE_HOLD)
                        return 'altitude_hold'
                    elif button_id == 'land-btn':
                        self.fc.set_flight_mode(FlightMode.LAND)
                        return 'land'
                    elif button_id == 'rtl-btn':
                        self.fc.set_flight_mode(FlightMode.RTL)
                        return 'return_to_launch'
                    elif button_id == 'pause-btn':
                        self.fc.set_flight_mode(FlightMode.STABILIZE)
                        return 'stabilize'
                
                return dash.no_update
            
            # Altitude setpoint callback
            @self.app.callback(
                Output('altitude-slider', 'value'),
                [Input('altitude-slider', 'value')]
            )
            def update_altitude_setpoint(altitude):
                if altitude is not None:
                    self.fc.setpoints['altitude'] = -altitude  # Convert to NED
                return altitude
        
        # Keep all the plot creation methods the same as before
        def _create_3d_trajectory(self):
            """Create 3D trajectory plot"""
            try:
                if not self.position_history:
                    return self._create_empty_plot("3D Trajectory", "Waiting for flight data...")
                
                df = list(self.position_history)
                
                # Convert NED to ENU for intuitive display
                x = [d['y'] for d in df]  # East -> X
                y = [d['x'] for d in df]  # North -> Y
                z = [-d['z'] for d in df]  # -Down -> Altitude
                
                fig = go.Figure()
                
                # Flight path
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines+markers',
                    line=dict(color='cyan', width=4),
                    marker=dict(size=3, color=z, colorscale='Viridis', showscale=True),
                    name='Flight Path'
                ))
                
                # Current position
                if len(x) > 0:
                    fig.add_trace(go.Scatter3d(
                        x=[x[-1]], y=[y[-1]], z=[z[-1]],
                        mode='markers',
                        marker=dict(size=8, color='red', symbol='diamond'),
                        name='Current Position'
                    ))
                
                # Waypoints
                if hasattr(self.fc, 'waypoints') and self.fc.waypoints:
                    wp_x = [wp[1] for wp in self.fc.waypoints]  # East
                    wp_y = [wp[0] for wp in self.fc.waypoints]  # North
                    wp_z = [-wp[2] for wp in self.fc.waypoints]  # Altitude
                    
                    fig.add_trace(go.Scatter3d(
                        x=wp_x, y=wp_y, z=wp_z,
                        mode='markers+text',
                        marker=dict(size=6, color='yellow', symbol='circle'),
                        text=[f"WP{i}" for i in range(len(wp_x))],
                        textposition="top center",
                        name='Waypoints'
                    ))
                
                fig.update_layout(
                    title="3D Flight Trajectory",
                    scene=dict(
                        xaxis_title="East (m)",
                        yaxis_title="North (m)",
                        zaxis_title="Altitude (m)",
                        aspectmode='data',
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
                    ),
                    height=500,
                    template="plotly_dark"
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error creating 3D plot: {e}")
                return self._create_empty_plot("3D Trajectory", f"Error: {str(e)}")
        
        def _create_position_plot(self):
            """Create position tracking plot"""
            try:
                if not self.position_history:
                    return self._create_empty_plot("Position", "Waiting for data...")
                
                df = list(self.position_history)
                times = [d['time'] - self.start_time for d in df]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=times, y=[d['x'] for d in df], name='North', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=times, y=[d['y'] for d in df], name='East', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=times, y=[d['z'] for d in df], name='Down', line=dict(color='blue')))
                
                fig.update_layout(
                    title="Position (NED Coordinates)",
                    xaxis_title="Time (s)",
                    yaxis_title="Position (m)",
                    height=500,
                    template="plotly_dark"
                )
                return fig
                
            except Exception as e:
                logger.error(f"Error creating position plot: {e}")
                return self._create_empty_plot("Position", f"Error: {str(e)}")
        
        def _create_attitude_plot(self):
            """Create attitude tracking plot"""
            try:
                # For demo, create some sample attitude data
                if not self.attitude_history:
                    # Initialize with some data
                    for i in range(50):
                        self.attitude_history.append({
                            'time': self.start_time + i * 0.1,
                            'roll': np.sin(i * 0.1) * 0.2,
                            'pitch': np.cos(i * 0.1) * 0.15,
                            'yaw': i * 0.01
                        })
                
                df = list(self.attitude_history)
                times = [d['time'] - self.start_time for d in df]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=times, y=[np.degrees(d['roll']) for d in df], 
                                       name='Roll', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=times, y=[np.degrees(d['pitch']) for d in df], 
                                       name='Pitch', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=times, y=[np.degrees(d['yaw']) for d in df], 
                                       name='Yaw', line=dict(color='blue')))
                
                fig.update_layout(
                    title="Attitude (Degrees)",
                    xaxis_title="Time (s)",
                    yaxis_title="Angle (deg)",
                    height=500,
                    template="plotly_dark"
                )
                return fig
                
            except Exception as e:
                logger.error(f"Error creating attitude plot: {e}")
                return self._create_empty_plot("Attitude", f"Error: {str(e)}")
        
        def _create_control_plot(self):
            """Create control output plot"""
            try:
                if not self.fc.control_history:
                    # Create sample control data
                    for i in range(50):
                        self.fc.control_history.append(np.array([0.5, 0.1, -0.1, 0.05]))
                
                controls = list(self.fc.control_history)
                times = [i * self.fc.dt for i in range(len(controls))]
                
                fig = go.Figure()
                control_names = ['Throttle', 'Roll', 'Pitch', 'Yaw']
                colors = ['red', 'green', 'blue', 'orange']
                
                for i in range(4):
                    fig.add_trace(go.Scatter(
                        x=times, y=[c[i] for c in controls],
                        name=control_names[i], line=dict(color=colors[i])
                    ))
                
                fig.update_layout(
                    title="Control Outputs",
                    xaxis_title="Time (s)",
                    yaxis_title="Control Value",
                    height=500,
                    template="plotly_dark"
                )
                return fig
                
            except Exception as e:
                logger.error(f"Error creating control plot: {e}")
                return self._create_empty_plot("Controls", f"Error: {str(e)}")
        
        def _create_empty_plot(self, title, message="No data available"):
            """Create an empty plot with message"""
            fig = go.Figure()
            fig.update_layout(
                title=title,
                xaxis={'visible': False},
                yaxis={'visible': False},
                annotations=[dict(
                    text=message,
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False,
                    font=dict(size=16, color='gray')
                )],
                height=500,
                template="plotly_dark"
            )
            return fig
        
        def run(self, debug: bool = False, port: int = 8050):
            """Start the dashboard server"""
            logger.info(f"🚀 Starting UAV Dashboard on http://localhost:{port}")
            self.app.run(debug=debug, port=port, host='0.0.0.0')
# ============================================================================
# PART 7: SIMULATION MANAGER AND INTEGRATION
# ============================================================================

class SimulationManager:
    """
    Manages the complete simulation with multiple UAVs,
    environment, and visualization
    """
    
    def __init__(self):
        self.flight_controller = FlightController()
        if HAS_DASH:
            self.dashboard = UAVDashboard(self.flight_controller)
        else:
            self.dashboard = None
        
        # Simulation state
        self.running = False
        self.simulation_thread = None
        self.real_time_factor = 1.0
        
        # Performance monitoring
        self.update_times = deque(maxlen=100)
        
    def start_simulation(self):
        """Start the simulation in a separate thread"""
        if self.running:
            logger.warning("Simulation already running")
            return
            
        self.running = True
        self.simulation_thread = threading.Thread(target=self._simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        logger.info("Simulation started")
        
    def stop_simulation(self):
        """Stop the simulation"""
        self.running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=2.0)
        logger.info("Simulation stopped")
        
    def _simulation_loop(self):
        """Main simulation loop"""
        last_time = time.time()
        
        while self.running:
            current_time = time.time()
            dt = current_time - last_time
            
            # Update flight controller
            start_time = time.time()
            self.flight_controller.update()
            update_time = time.time() - start_time
            
            self.update_times.append(update_time)
            
            # Maintain real-time factor
            expected_dt = self.flight_controller.dt / self.real_time_factor
            sleep_time = max(0, expected_dt - update_time)
            time.sleep(sleep_time)
            
            last_time = current_time
    
    def get_performance_stats(self) -> Dict:
        """Get simulation performance statistics"""
        if not self.update_times:
            return {}
            
        times = list(self.update_times)
        return {
            'update_rate': 1.0 / np.mean(times) if np.mean(times) > 0 else 0,
            'update_time_mean': np.mean(times),
            'update_time_std': np.std(times),
            'real_time_factor': self.real_time_factor
        }
    
    def set_real_time_factor(self, factor: float):
        """Set real-time simulation factor"""
        self.real_time_factor = max(0.1, min(10.0, factor))
    
    def run_with_dashboard(self, dashboard_port: int = 8050):
        """Run simulation with web dashboard"""
        if not HAS_DASH or self.dashboard is None:
            print("Dash not available. Cannot run dashboard.")
            print("To enable the dashboard, install: pip install dash plotly dash-bootstrap-components")
            return
            
        logger.info("Starting simulation with dashboard...")
        
        # Start simulation
        self.start_simulation()
        
        try:
            # Start dashboard
            self.dashboard.run(port=dashboard_port)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
        finally:
            self.stop_simulation()

    def run_headless(self):
        """Run simulation without dashboard"""
        logger.info("Starting headless simulation...")
        
        # Start simulation
        self.start_simulation()
        
        try:
            # Keep running until interrupted
            while self.running:
                time.sleep(1)
                # Print some basic telemetry
                telemetry = self.flight_controller.get_telemetry()
                altitude = -telemetry['position'][2]
                print(f"Position: N{telemetry['position'][0]:.1f}, E{telemetry['position'][1]:.1f}, Alt{altitude:.1f}m, Mode: {telemetry['flight_mode']}")
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.stop_simulation()


# ============================================================================
# PART 8: ADVANCED FEATURES - FAULT DETECTION AND RECONFIGURATION
# ============================================================================

class FaultDetectionAndRecovery:
    """
    Advanced fault detection, isolation, and recovery system
    Monitors system health and implements recovery strategies
    """
    
    def __init__(self, flight_controller: FlightController):
        self.fc = flight_controller
        
        # Fault thresholds
        self.attitude_fault_threshold = np.radians(45)  # 45 degrees
        self.velocity_fault_threshold = 10.0  # m/s
        self.altitude_fault_threshold = 0.5   # m from ground
        self.sensor_consistency_threshold = 5.0  # m position disagreement
        
        # Fault states
        self.faults_detected = {
            'sensor_failure': False,
            'motor_failure': False,
            'attitude_anomaly': False,
            'gps_loss': False
        }
        
        # Recovery strategies
        self.recovery_actions = {
            'sensor_failure': self._handle_sensor_failure,
            'motor_failure': self._handle_motor_failure,
            'attitude_anomaly': self._handle_attitude_anomaly,
            'gps_loss': self._handle_gps_loss
        }
    
    def check_faults(self):
        """Check for system faults and trigger recovery"""
        # Check sensor consistency
        pos_disagreement = np.linalg.norm(
            self.fc.estimated_state.position - self.fc.state.position
        )
        if pos_disagreement > self.sensor_consistency_threshold:
            self.faults_detected['sensor_failure'] = True
        
        # Check attitude anomalies
        if np.any(np.abs(self.fc.estimated_state.orientation) > self.attitude_fault_threshold):
            self.faults_detected['attitude_anomaly'] = True
        
        # Check velocity limits
        if np.linalg.norm(self.fc.estimated_state.velocity) > self.velocity_fault_threshold:
            self.faults_detected['attitude_anomaly'] = True
        
        # Check GPS status
        if np.linalg.norm(self.fc.sensor_data.gps_position) < 0.01:
            self.faults_detected['gps_loss'] = True
        
        # Execute recovery actions for detected faults
        for fault_type, detected in self.faults_detected.items():
            if detected:
                logger.warning(f"Fault detected: {fault_type}")
                self.recovery_actions[fault_type]()
    
    def _handle_sensor_failure(self):
        """Recovery strategy for sensor failures"""
        # Switch to degraded sensor mode
        if self.fc.flight_mode != FlightMode.RTL:
            self.fc.set_flight_mode(FlightMode.RTL)
    
    def _handle_motor_failure(self):
        """Recovery strategy for motor failures"""
        # Emergency landing
        self.fc.set_flight_mode(FlightMode.LAND)
    
    def _handle_attitude_anomaly(self):
        """Recovery strategy for attitude anomalies"""
        # Attempt to regain stable attitude
        self.fc.set_flight_mode(FlightMode.STABILIZE)
    
    def _handle_gps_loss(self):
        """Recovery strategy for GPS loss"""
        # Switch to attitude or altitude hold
        if self.fc.flight_mode in [FlightMode.POSITION_HOLD, FlightMode.AUTO]:
            self.fc.set_flight_mode(FlightMode.ALTITUDE_HOLD)


# ============================================================================
# PART 9: MAIN APPLICATION AND USAGE EXAMPLES
# ============================================================================

def main():
    """Main application entry point"""
    print("AI-Based UAV Autopilot Simulator - FIXED 3D Trajectory")
    print("======================================================")
    
    # Create simulation manager
    sim_manager = SimulationManager()
    
    # Add fault detection
    fault_detector = FaultDetectionAndRecovery(sim_manager.flight_controller)
    
    # Create a realistic 3D trajectory (box pattern)
    waypoints = [
        np.array([0, 0, -10]),    # Start
        np.array([20, 0, -15]),   # Forward
        np.array([20, 20, -20]),  # Right and up
        np.array([0, 20, -15]),   # Back
        np.array([0, 0, -10]),    # Return to start
    ]
    
    sim_manager.flight_controller.set_waypoints(waypoints)
    sim_manager.flight_controller.set_flight_mode(FlightMode.AUTO)
    
    # Run with dashboard if available, otherwise run headless
    if HAS_DASH:
        try:
            sim_manager.run_with_dashboard(dashboard_port=8050)
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            sim_manager.stop_simulation()
    else:
        sim_manager.run_headless()

def demo_control_modes():
    """Demonstrate different control modes"""
    fc = FlightController()
    
    print("Testing Control Modes:")
    print("1. Stabilize Mode")
    fc.set_flight_mode(FlightMode.STABILIZE)
    for _ in range(100):
        fc.update()
    
    print("2. Altitude Hold Mode")
    fc.set_flight_mode(FlightMode.ALTITUDE_HOLD)
    for _ in range(200):
        fc.update()
    
    print("3. Position Hold Mode")
    fc.set_flight_mode(FlightMode.POSITION_HOLD)
    for _ in range(200):
        fc.update()
    
    print("4. Autonomous Mission")
    waypoints = [np.array([5, 5, -10]), np.array([5, -5, -15]), np.array([-5, -5, -10])]
    fc.set_waypoints(waypoints)
    fc.set_flight_mode(FlightMode.AUTO)
    
    for _ in range(500):
        fc.update()
        if fc.mission_complete:
            print("Mission complete!")
            break

def train_rl_agent():
    """Train the RL autopilot"""
    print("Training RL Agent...")
    rl_autopilot = RLAutopilot()
    
    # Train for a short duration for demo
    rl_autopilot.train(total_timesteps=10000)
    
    # Save the trained model
    rl_autopilot.save_model("uav_autopilot_ppo.pth")
    print("Training completed and model saved!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI-Based UAV Autopilot Simulator")
    parser.add_argument("--mode", choices=["simulate", "train", "demo"], 
                       default="simulate", help="Operation mode")
    parser.add_argument("--port", type=int, default=8050, help="Dashboard port")
    parser.add_argument("--headless", action="store_true", help="Run without dashboard")
    
    args = parser.parse_args()
    
    if args.mode == "simulate":
        if args.headless or not HAS_DASH:
            # Create simulation manager and run headless
            sim_manager = SimulationManager()
            
            # Add fault detection
            fault_detector = FaultDetectionAndRecovery(sim_manager.flight_controller)
            
            # Example waypoint mission
            waypoints = [
                np.array([10, 0, -10]),
                np.array([10, 10, -15]),
                np.array([0, 10, -20]),
                np.array([0, 0, -10])
            ]
            sim_manager.flight_controller.set_waypoints(waypoints)
            sim_manager.flight_controller.set_flight_mode(FlightMode.AUTO)
            
            sim_manager.run_headless()
        else:
            main()
    elif args.mode == "train":
        train_rl_agent()
    elif args.mode == "demo":
        demo_control_modes()