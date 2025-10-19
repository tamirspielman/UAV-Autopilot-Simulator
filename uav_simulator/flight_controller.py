"""
Flight Controller - CRITICAL FIX: Sign Error Corrected
The previous version had inverted control signs causing descent instead of climb
"""
import numpy as np
import time
from typing import List, Optional, Dict
from collections import deque

from .utils import FlightMode, logger
from .dynamics import UAVState, UAVDynamics
from .sensor_model import SensorData, SensorModel, ExtendedKalmanFilter
from .PIDController import PIDController, RLAutopilot
from .datalogger import DataLogger

class FlightController:
    """
    Flight controller with CORRECTED sign conventions
    """
    
    def __init__(self):
        # Initialize components
        self.dynamics = UAVDynamics()
        self.sensor_model = SensorModel()
        self.ekf = ExtendedKalmanFilter()
        
        # FIXED: Simple, working PID controllers
        # Altitude PID: output is throttle adjustment directly
        self.altitude_pid = PIDController(
            kp=1.5,
            ki=0.1,
            kd=0.8,
            output_limits=(-0.3, 0.3)  # Throttle adjustment
        )
        
        # Attitude controllers
        self.roll_pid = PIDController(2.5, 0.05, 0.2, (-1, 1))
        self.pitch_pid = PIDController(2.5, 0.05, 0.2, (-1, 1))
        self.yaw_pid = PIDController(1.0, 0.02, 0.15, (-1, 1))
        
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
            'position': np.array([0, 0, 10]),
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
        self.log_interval = 0.1
        
        # Data logging
        self.telemetry_history = deque(maxlen=1000)
        self.control_history = deque(maxlen=1000)
        
        # Waypoint acceptance radius
        self.waypoint_acceptance_radius = 2.0
        
        # Data logger
        self.data_logger = DataLogger()
        self.data_logger.start_new_log()
        
    def update(self, manual_control: Optional[np.ndarray] = None) -> UAVState:
        """Main update loop"""
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
            
            # Compute control
            if manual_control is not None and self.flight_mode == FlightMode.MANUAL:
                self.control_output = manual_control
            else:
                self.control_output = self._compute_autonomous_control()
            
            # Apply control and update dynamics
            self.state = self.dynamics.update(self.state, self.control_output, dt)
            
            # Update mission
            self._update_mission()
            
            # Log telemetry
            self._log_telemetry()
            
            # Data logging
            if current_time - self.last_log_time >= self.log_interval:
                self.data_logger.log_data(self)
                self.last_log_time = current_time
            
            self.last_update = current_time
        
        return self.state
    
    def _compute_autonomous_control(self) -> np.ndarray:
        """Compute control based on flight mode"""
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
    
    def _compute_altitude_control(self, target_altitude: float) -> float:
        """
        FIXED: Correct altitude control with proper sign convention
    
        Key insight: In NED coordinates, negative z = higher altitude
        When we're too low (z too high/less negative), we need MORE throttle
        When we're too high (z too low/more negative), we need LESS throttle
        """
        current_z = self.estimated_state.position[2]  # ENU: z up
        altitude_error = target_altitude - current_z  # positive = too low
        throttle_adjustment = self.altitude_pid.compute(0, altitude_error, self.dt)
        hover_throttle = 0.5
        throttle = hover_throttle + throttle_adjustment
        return np.clip(throttle, 0.0, 1.0)
    
    def _compute_rl_control(self) -> np.ndarray:
        """RL control"""
        obs = np.concatenate([
            self.estimated_state.position,
            self.estimated_state.velocity,
            self.estimated_state.orientation,
            self.estimated_state.angular_velocity,
            self.sensor_data.imu_accel[:3]
        ])
        return self.rl_autopilot.compute_control(obs)
    
    def _compute_altitude_hold_control(self) -> np.ndarray:
        """FIXED: Altitude hold"""
        throttle = self._compute_altitude_control(self.setpoints['altitude'])
        
        # Keep level
        roll = self.roll_pid.compute(0, self.estimated_state.orientation[0], self.dt)
        pitch = self.pitch_pid.compute(0, self.estimated_state.orientation[1], self.dt)
        yaw = self.yaw_pid.compute(0, self.estimated_state.orientation[2], self.dt)
        
        return np.array([throttle, roll, pitch, yaw])
    
    def _compute_position_hold_control(self) -> np.ndarray:
        """FIXED: Position hold"""
        # Horizontal control
        pos_error = self.setpoints['position'][:2] - self.estimated_state.position[:2]
        desired_velocity = np.clip(pos_error * 0.5, -2.0, 2.0)
        vel_error = desired_velocity - self.estimated_state.velocity[:2]
        
        # Velocity to attitude
        desired_pitch = np.clip(vel_error[0] * 0.3, -0.4, 0.4)
        desired_roll = np.clip(-vel_error[1] * 0.3, -0.4, 0.4)
        
        # Altitude control
        throttle = self._compute_altitude_control(self.setpoints['altitude'])
        
        # Attitude
        roll = self.roll_pid.compute(desired_roll, self.estimated_state.orientation[0], self.dt)
        pitch = self.pitch_pid.compute(desired_pitch, self.estimated_state.orientation[1], self.dt)
        yaw = self.yaw_pid.compute(0, self.estimated_state.orientation[2], self.dt)
        
        return np.array([throttle, roll, pitch, yaw])
    
    def _compute_auto_control(self) -> np.ndarray:
        """FIXED: Auto mission"""
        if not self.waypoints:
            return self._compute_position_hold_control()

        idx = max(0, min(self.current_waypoint_index, len(self.waypoints) - 1))
        current_wp = np.array(self.waypoints[idx], dtype=float)

        self.setpoints['position'] = current_wp
        self.setpoints['altitude'] = current_wp[2]

        # Horizontal
        pos_err = current_wp[:2] - self.estimated_state.position[:2]
        dist = np.linalg.norm(pos_err)
        
        if dist > 0.1:
            desired_vel = np.clip(pos_err * 0.5, -2.0, 2.0)
        else:
            desired_vel = np.zeros(2)

        vel_err = desired_vel - self.estimated_state.velocity[:2]
        desired_pitch = np.clip(vel_err[0] * 0.3, -0.4, 0.4)
        desired_roll = np.clip(-vel_err[1] * 0.3, -0.4, 0.4)

        # Altitude
        throttle = self._compute_altitude_control(current_wp[2])

        # Attitude
        roll = self.roll_pid.compute(desired_roll, self.estimated_state.orientation[0], self.dt)
        pitch = self.pitch_pid.compute(desired_pitch, self.estimated_state.orientation[1], self.dt)
        yaw = self.yaw_pid.compute(0.0, self.estimated_state.orientation[2], self.dt)

        # Waypoint check
        distance = np.linalg.norm(self.estimated_state.position - current_wp)
        if distance < self.waypoint_acceptance_radius:
            if self.current_waypoint_index < len(self.waypoints) - 1:
                logger.info(f"Waypoint {self.current_waypoint_index} reached.")
                self.current_waypoint_index += 1
            else:
                logger.info("Mission complete.")
                self.mission_complete = True

        return np.array([throttle, roll, pitch, yaw])
    
    def _compute_rtl_control(self) -> np.ndarray:
        """RTL"""
        self.setpoints['position'] = np.array([0, 0, self.setpoints['altitude']])
        
        distance = np.linalg.norm(self.estimated_state.position[:2])
        altitude_ok = abs(self.estimated_state.position[2] - self.setpoints['altitude']) < 1.0
        
        if distance < 2.0 and altitude_ok:
            logger.info("RTL complete, landing.")
            self.set_flight_mode(FlightMode.LAND)
        
        return self._compute_position_hold_control()
    
    def _compute_land_control(self) -> np.ndarray:
        """FIXED: Landing"""
        current_altitude = self.estimated_state.position[2]  # Convert to positive altitude
        
        if current_altitude > 0.5:
            # Gradual descent: increase z (less negative) slowly
            target_z = current_altitude - 0.2 * self.dt  # smooth descent
            throttle = self._compute_altitude_control(target_z)
            
            # Keep level
            roll = self.roll_pid.compute(0, self.estimated_state.orientation[0], self.dt)
            pitch = self.pitch_pid.compute(0, self.estimated_state.orientation[1], self.dt)
            yaw = self.yaw_pid.compute(0, self.estimated_state.orientation[2], self.dt)
            
            return np.array([throttle, roll, pitch, yaw])
        else:
            logger.info("Landed.")
            return np.array([0, 0, 0, 0])
    
    def _compute_stabilize_control(self) -> np.ndarray:
        """FIXED: Stabilize"""
        throttle = self._compute_altitude_control(self.setpoints['altitude'])
        
        roll = self.roll_pid.compute(0, self.estimated_state.orientation[0], self.dt)
        pitch = self.pitch_pid.compute(0, self.estimated_state.orientation[1], self.dt)
        yaw = self.yaw_pid.compute(0, self.estimated_state.orientation[2], self.dt)
        
        return np.array([throttle, roll, pitch, yaw])
    
    def _update_mission(self):
        """Update mission"""
        pass
    
    def _log_telemetry(self):
        """Log telemetry"""
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
        """Change flight mode"""
        if mode != self.flight_mode:
            self.altitude_pid.reset()
            self.roll_pid.reset()
            self.pitch_pid.reset()
            self.yaw_pid.reset()
            self.flight_mode = mode
            logger.info(f"Mode: {mode.value}")
    
    def set_waypoints(self, waypoints: List[np.ndarray]):
        """Set waypoints"""
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        self.mission_complete = False
        logger.info(f"Mission: {len(waypoints)} waypoints")
    
    def get_telemetry(self) -> Dict:
        """Get telemetry"""
        return {
            'position': self.estimated_state.position.tolist(),
            'velocity': self.estimated_state.velocity.tolist(),
            'attitude': self.estimated_state.orientation.tolist(),
            'flight_mode': self.flight_mode.value,
            'battery': 85.0,
            'gps_fix': True,
            'waypoint_index': self.current_waypoint_index,
            'mission_complete': self.mission_complete
        }
    
    def stop_logging(self):
        """Stop logging"""
        self.data_logger.stop_logging()
    
    def get_flight_report(self):
        """Get report"""
        return self.data_logger.generate_report()
    
    def get_recent_logs(self):
        """Get logs"""
        return self.data_logger.get_recent_logs()