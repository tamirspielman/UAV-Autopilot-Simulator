"""
Sensor Models and Fusion Algorithms
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict
from scipy import linalg
from .dynamics import UAVState
from .utils import rotation_matrix, normalize_angles, wrap_angle, logger

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
        R = rotation_matrix(true_state.orientation)
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
        
        sensor_data.barometer_altitude = true_state.position[2] + np.random.randn() * self.baro_noise
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

"""
Extended Kalman Filter - FIXED to start at ground level
"""
import numpy as np
from .dynamics import UAVState
from .sensor_model import SensorData
from .utils import rotation_matrix, normalize_angles, wrap_angle


class ExtendedKalmanFilter:
    def __init__(self):
        # State vector: [position(3), velocity(3), orientation(3), accel_bias(3), gyro_bias(3)]
        self.state_dim = 15
        self.state = np.zeros(self.state_dim)
        
        # FIXED: Initialize at ground level [0, 0, 0]
        self.state[2] = 0.0  # Ground level in NED
        
        # Covariance matrix
        self.P = np.eye(self.state_dim) * 0.01
        
        # Process noise (tuned for UAV)
        self.Q = np.eye(self.state_dim) * 0.001
        
        # Measurement noise
        self.R_gps = np.eye(6) * 0.5
        self.R_baro = 0.2
        self.R_mag = np.eye(2) * 0.01
        
        self.last_imu_time = None

    def predict(self, imu_data: SensorData, dt: float):
        """Prediction step using IMU data"""
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
        R = rotation_matrix(orient)
        
        # State prediction
        gravity_world = np.array([0, 0, 9.81])
        
        # Position update
        self.state[:3] += vel * dt + 0.5 * (R @ accel_corrected + gravity_world) * dt**2
        
        # Velocity update
        self.state[3:6] += (R @ accel_corrected + gravity_world) * dt
        
        # Orientation update
        self.state[6:9] += gyro_corrected * dt
        
        # Normalize orientation
        self.state[6:9] = normalize_angles(self.state[6:9])
        
        # Simplified covariance prediction
        F = np.eye(self.state_dim)
        self.P = F @ self.P @ F.T + self.Q

    def update_gps(self, gps_data: SensorData):
        """Update step using GPS measurements"""
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
        """Update step using barometer"""
        H = np.zeros((1, self.state_dim))
        H[0, 2] = -1  # altitude = -z_position
        
        innovation = baro_altitude - (H @ self.state)[0]
        
        S = H @ self.P @ H.T + self.R_baro
        K = (self.P @ H.T) / S
        
        self.state += K.flatten() * innovation
        self.P = (np.eye(self.state_dim) - np.outer(K, H)) @ self.P

    def update_magnetometer(self, mag_data: np.ndarray):
        """Update step using magnetometer"""
        mag_yaw = np.arctan2(mag_data[1], mag_data[0])
        
        H = np.zeros((1, self.state_dim))
        H[0, 8] = 1  # Yaw component
        
        innovation = wrap_angle(mag_yaw - self.state[8])
        
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