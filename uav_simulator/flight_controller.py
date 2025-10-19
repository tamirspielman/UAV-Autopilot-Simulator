"""
Flight Controller Logic and Setpoint Management
"""
import numpy as np
import time
from typing import List, Optional, Dict
from collections import deque

from .utils import FlightMode, logger
from .dynamics import UAVState, UAVDynamics
from .sensor_model import SensorData, SensorModel, ExtendedKalmanFilter
from .autopilot import PIDController, RLAutopilot
from .datalogger import DataLogger

class FlightController:
    """
    Main flight controller with integrated data logging
    """
    
    def __init__(self):
        # Initialize components
        self.dynamics = UAVDynamics()
        self.sensor_model = SensorModel()
        self.ekf = ExtendedKalmanFilter()
        
        # Controllers for different axes
        self.altitude_pid = PIDController(2.0, 0.1, 0.5, (0, 1))
        self.roll_pid = PIDController(3.0, 0.05, 0.2, (-1, 1))
        self.pitch_pid = PIDController(3.0, 0.05, 0.2, (-1, 1))
        self.yaw_pid = PIDController(1.0, 0.01, 0.1, (-1, 1))
        
        # RL autopilot
        self.rl_autopilot = RLAutopilot()
        
        # Current state
        self.state = UAVState()
        self.estimated_state = UAVState()
        self.sensor_data = SensorData()
        
        # Flight mode and setpoints
        self.flight_mode = FlightMode.STABILIZE
        self.setpoints = {
            'altitude': -10.0,
            'position': np.array([0, 0, -10]),
            'velocity': np.zeros(3),
            'attitude': np.zeros(3)
        }
        
        # Mission waypoints
        self.waypoints = []
        self.current_waypoint_index = 0
        self.mission_complete = False
        
        # Control outputs
        self.control_output = np.array([0.5, 0, 0, 0])
        
        # Timing
        self.dt = 0.01
        self.last_update = time.time()
        self.last_log_time = time.time()
        self.log_interval = 0.1  # Log every 0.1 seconds (10 Hz)
        
        # Data logging
        self.telemetry_history = deque(maxlen=1000)
        self.control_history = deque(maxlen=1000)
        
        # Waypoint acceptance radius
        self.waypoint_acceptance_radius = 2.0
        
        # Data logger
        self.data_logger = DataLogger()
        self.data_logger.start_new_log()
        
    def update(self, manual_control: Optional[np.ndarray] = None) -> UAVState:
        """Main update loop with data logging"""
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
            
            # Data logging (at reduced rate to avoid excessive file I/O)
            if current_time - self.last_log_time >= self.log_interval:
                self.data_logger.log_data(self)
                self.last_log_time = current_time
            
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
        """Compute control using RL autopilot - CORRECTED OBSERVATION"""
        # Get observation for RL - CORRECTED: Use only 15 features
        obs = np.concatenate([
            self.estimated_state.position,           # 3 features
            self.estimated_state.velocity,           # 3 features  
            self.estimated_state.orientation,        # 3 features
            self.estimated_state.angular_velocity,   # 3 features
            self.sensor_data.imu_accel[:3]           # 3 features (only first 3)
        ])
        # Total: 15 features that match the neural network input
        
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
        """
        Autonomous mission following control.
        Robust to waypoint list size, index limits, and mission completion.
        """
        if not self.waypoints:
            return self._compute_position_hold_control()

        # Clamp waypoint index
        idx = max(0, min(self.current_waypoint_index, len(self.waypoints) - 1))
        current_wp = np.array(self.waypoints[idx], dtype=float)

        # Set active setpoint for logger
        self.setpoints['position'] = current_wp

        # Position control
        pos_err_xy = current_wp[:2] - self.estimated_state.position[:2]
        dist_xy = np.linalg.norm(pos_err_xy)
        desired_velocity_xy = np.clip(pos_err_xy * 0.3, -2.0, 2.0) if dist_xy > 0.1 else np.zeros(2)

        # Altitude control
        altitude_error = current_wp[2] - self.estimated_state.position[2]
        desired_velocity_z = np.clip(altitude_error * 0.5, -1.0, 1.0)
        desired_velocity = np.array([desired_velocity_xy[0], desired_velocity_xy[1], desired_velocity_z])

        # Convert desired velocities → attitude setpoints
        desired_pitch = np.clip(desired_velocity[0] * 0.3, -0.4, 0.4)
        desired_roll = np.clip(-desired_velocity[1] * 0.3, -0.4, 0.4)

        # Compute throttle & attitude commands
        throttle = self.altitude_pid.compute(current_wp[2], self.estimated_state.position[2], self.dt)
        roll = self.roll_pid.compute(desired_roll, self.estimated_state.orientation[0], self.dt)
        pitch = self.pitch_pid.compute(desired_pitch, self.estimated_state.orientation[1], self.dt)
        yaw = self.yaw_pid.compute(0.0, self.estimated_state.orientation[2], self.dt)

        control_output = np.array([throttle, roll, pitch, yaw])

        # Waypoint reached check
        distance = np.linalg.norm(self.estimated_state.position - current_wp)
        if distance < self.waypoint_acceptance_radius:
            if self.current_waypoint_index < len(self.waypoints) - 1:
                logger.info(f"Waypoint {self.current_waypoint_index} reached ({distance:.2f} m).")
                self.current_waypoint_index += 1
            else:
                logger.info("Final waypoint reached. Mission complete.")
                self.mission_complete = True

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
        }
    
    def stop_logging(self):
        """Stop data logging"""
        self.data_logger.stop_logging()
    
    def get_flight_report(self):
        """Get a flight analysis report"""
        return self.data_logger.generate_report()
    
    def get_recent_logs(self):
        """Get list of recent flight logs"""
        return self.data_logger.get_recent_logs()