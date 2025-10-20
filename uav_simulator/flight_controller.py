"""
Flight Controller - MINIMAL FIXED VERSION
Focus on correct altitude control fundamentals
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
    Flight Controller with CORRECTED altitude control
    """
    
    def __init__(self):
        # Core components
        self.dynamics = UAVDynamics()
        self.sensor_model = SensorModel()
        self.ekf = ExtendedKalmanFilter()
        
        # Conservative PID tuning
        self.altitude_pid = PIDController(
            kp=0.6,    # Moderate proportional
            ki=0.01,   # Small integral  
            kd=0.3,    # Good damping
            output_limits=(-0.1, 0.1)
        )
        
        # Attitude controllers
        self.roll_pid = PIDController(1.0, 0.02, 0.2, (-0.2, 0.2))
        self.pitch_pid = PIDController(1.0, 0.02, 0.2, (-0.2, 0.2))
        self.yaw_pid = PIDController(0.8, 0.01, 0.15, (-0.15, 0.15))
        
        # RL autopilot
        self.rl_autopilot = RLAutopilot()
        
        # Flight mode
        self.flight_mode = FlightMode.STABILIZE
        
        # Initialize state - FORCE correct initial conditions
        self.state = UAVState()
        self.state.position = np.array([0.0, 0.0, -10.0])  # 10m altitude in NED
        self.state.velocity = np.zeros(3)
        self.state.orientation = np.zeros(3)
        self.state.angular_velocity = np.zeros(3)
        self.state.motor_speeds = np.array([5000.0, 5000.0, 5000.0, 5000.0])
        self.state.acceleration = np.zeros(3)
        self.state.timestamp = 0.0
        
        # Estimated state
        self.estimated_state = UAVState()
        self.estimated_state.position = np.array([0.0, 0.0, -10.0])
        self.estimated_state.velocity = np.zeros(3)
        self.estimated_state.orientation = np.zeros(3)
        
        # Sensor data
        self.sensor_data = SensorData()
        
        # Setpoints
        self.setpoints = {
            'altitude': 10.0,  # meters above ground
            'position': np.array([0.0, 0.0, -10.0]),  # NED coordinates
            'yaw': 0.0
        }
        
        # Mission waypoints
        self.waypoints = []
        self.current_waypoint_index = 0
        self.mission_complete = False
        
        # Control output - START WITH MODERATE THROTTLE
        self.control_output = np.array([0.55, 0.0, 0.0, 0.0])
        
        # Timing
        self.dt = 0.01
        self.last_update = time.time()
        self.last_log_time = time.time()
        self.log_interval = 0.1
        
        # Data logging
        self.telemetry_history = deque(maxlen=1000)
        self.data_logger = DataLogger()
        self.data_logger.start_new_log()
        
        # Hover throttle - will be calibrated
        self.hover_throttle = 0.52
        
        logger.info("✓ FlightController initialized - MINIMAL FIXED VERSION")
    
    def update(self, manual_control: Optional[np.ndarray] = None) -> UAVState:
        """
        Main control loop - SIMPLIFIED AND CORRECTED
        """
        current_time = time.time()
        dt = current_time - self.last_update
        
        if dt >= self.dt:
            try:
                # Step 1: Get sensor measurements
                self.sensor_data = self.sensor_model.measure(self.state, dt)
                self.sensor_model.update_biases(dt)
                
                # Step 2: Estimate state
                self.ekf.predict(self.sensor_data, dt)
                if np.linalg.norm(self.sensor_data.gps_position) > 0.01:
                    self.ekf.update_gps(self.sensor_data)
                self.ekf.update_barometer(self.sensor_data.barometer_altitude)
                self.ekf.update_magnetometer(self.sensor_data.magnetometer)
                self.estimated_state = self.ekf.get_estimated_state()
                
                # Step 3: Compute control
                if manual_control is not None and self.flight_mode == FlightMode.MANUAL:
                    self.control_output = manual_control
                else:
                    self.control_output = self._compute_control()
                
                # Step 4: Apply control to dynamics
                self.state = self.dynamics.update(self.state, self.control_output, dt)
                
                # Step 5: Basic safety
                current_altitude = -self.state.position[2]
                if current_altitude < 0.5:
                    self.state.position[2] = -0.5
                    if self.state.velocity[2] > 0:
                        self.state.velocity[2] = 0.0
                
                # Step 6: Logging
                self._log_telemetry()
                if current_time - self.last_log_time >= self.log_interval:
                    self.data_logger.log_data(self)
                    self.last_log_time = current_time
                
                self.last_update = current_time
                
            except Exception as e:
                logger.error(f"Control loop error: {e}")
                self.control_output = np.array([0.55, 0.0, 0.0, 0.0])
        
        return self.state
    
    def _compute_control(self) -> np.ndarray:
        """
        Simple control routing
        """
        if self.flight_mode == FlightMode.STABILIZE:
            return self._stabilize_mode()
        elif self.flight_mode == FlightMode.ALTITUDE_HOLD:
            return self._altitude_hold_mode()
        elif self.flight_mode == FlightMode.POSITION_HOLD:
            return self._position_hold_mode()
        elif self.flight_mode == FlightMode.AUTO:
            return self._auto_mode()
        elif self.flight_mode == FlightMode.RTL:
            return self._rtl_mode()
        elif self.flight_mode == FlightMode.LAND:
            return self._land_mode()
        elif self.flight_mode == FlightMode.AI_PILOT:
            return self._ai_pilot_mode()
        else:
            return self._stabilize_mode()
    
    def _stabilize_mode(self) -> np.ndarray:
        current_altitude = -self.estimated_state.position[2]
        target_altitude = self.setpoints['altitude']

        # ALTITUDE CONTROL
        throttle_adjustment = self.altitude_pid.compute(
            setpoint=target_altitude,
            measurement=current_altitude,
            dt=self.dt
        )

        throttle = self.hover_throttle + throttle_adjustment
        throttle = np.clip(throttle, 0.5, 0.65)
        current_pos = self.estimated_state.position[:2] 
        pos_gain = 0.02
        desired_pitch = -current_pos[0] * pos_gain  
        desired_roll = current_pos[1] * pos_gain
        max_tilt = 0.1  
        desired_pitch = np.clip(desired_pitch, -max_tilt, max_tilt)
        desired_roll = np.clip(desired_roll, -max_tilt, max_tilt)
        roll = self.roll_pid.compute(desired_roll, self.estimated_state.orientation[0], self.dt)
        pitch = self.pitch_pid.compute(desired_pitch, self.estimated_state.orientation[1], self.dt)
        yaw = self.yaw_pid.compute(0.0, self.estimated_state.orientation[2], self.dt)

        return np.array([throttle, roll, pitch, yaw])
    def _altitude_hold_mode(self) -> np.ndarray:
        """
        ALTITUDE HOLD: Same as stabilize but holds current altitude
        """
        return self._stabilize_mode()
    
    def _position_hold_mode(self) -> np.ndarray:
        """
        POSITION HOLD: Hold position and altitude
        """
        current_pos = self.estimated_state.position[:2]
        target_pos = self.setpoints['position'][:2]
        
        pos_error = target_pos - current_pos
        
        if np.linalg.norm(pos_error) < 0.5:
            desired_roll = 0.0
            desired_pitch = 0.0
        else:
            max_tilt = 0.2
            gain = 0.3
            desired_pitch = np.clip(-pos_error[0] * gain, -max_tilt, max_tilt)
            desired_roll = np.clip(pos_error[1] * gain, -max_tilt, max_tilt)
        
        # Use altitude control from stabilize mode
        target_altitude = -self.setpoints['position'][2]
        throttle_adjustment = self.altitude_pid.compute(
            target_altitude,
            -self.estimated_state.position[2], 
            self.dt
        )
        throttle = self.hover_throttle + throttle_adjustment
        throttle = np.clip(throttle, 0.4, 0.7)
        
        roll = self.roll_pid.compute(desired_roll, self.estimated_state.orientation[0], self.dt)
        pitch = self.pitch_pid.compute(desired_pitch, self.estimated_state.orientation[1], self.dt)
        yaw = self.yaw_pid.compute(self.setpoints['yaw'], self.estimated_state.orientation[2], self.dt)
        
        return np.array([throttle, roll, pitch, yaw])
    
    def _auto_mode(self) -> np.ndarray:
        """
        AUTO: Follow waypoints
        """
        if not self.waypoints or self.mission_complete:
            return self._position_hold_mode()
        
        current_wp = self.waypoints[self.current_waypoint_index]
        self.setpoints['position'] = current_wp
        
        control = self._position_hold_mode()
        
        current_pos = self.estimated_state.position
        distance = np.linalg.norm(current_pos - current_wp)
        
        if distance < 2.0:
            if self.current_waypoint_index < len(self.waypoints) - 1:
                self.current_waypoint_index += 1
                logger.info(f"Waypoint reached")
            else:
                self.mission_complete = True
                logger.info("Mission complete!")
        
        return control
    
    def _rtl_mode(self) -> np.ndarray:
        """
        RETURN TO LAUNCH
        """
        home = np.array([0.0, 0.0, -10.0])
        self.setpoints['position'] = home
        
        control = self._position_hold_mode()
        
        distance = np.linalg.norm(self.estimated_state.position - home)
        if distance < 2.0:
            logger.info("Home reached, landing...")
            self.set_flight_mode(FlightMode.LAND)
        
        return control
    
    def _land_mode(self) -> np.ndarray:
        """
        LAND: Descend to ground
        """
        current_altitude = -self.estimated_state.position[2]
        
        if current_altitude < 0.3:
            logger.info("Touchdown!")
            return np.array([0.0, 0.0, 0.0, 0.0])
        
        # Slow descent
        descent_rate = 0.3
        target_altitude = current_altitude - descent_rate * self.dt
        target_altitude = max(0.0, target_altitude)
        
        throttle_adjustment = self.altitude_pid.compute(
            target_altitude,
            current_altitude,
            self.dt
        )
        throttle = self.hover_throttle + throttle_adjustment
        
        # Reduce throttle near ground
        if current_altitude < 2.0:
            throttle *= 0.8
        
        roll = self.roll_pid.compute(0.0, self.estimated_state.orientation[0], self.dt)
        pitch = self.pitch_pid.compute(0.0, self.estimated_state.orientation[1], self.dt)
        yaw = self.yaw_pid.compute(0.0, self.estimated_state.orientation[2], self.dt)
        
        return np.array([throttle, roll, pitch, yaw])
    
    def _ai_pilot_mode(self) -> np.ndarray:
        """
        AI PILOT
        """
        obs = np.concatenate([
            self.estimated_state.position,
            self.estimated_state.velocity,
            self.estimated_state.orientation,
            self.estimated_state.angular_velocity,
            self.sensor_data.imu_accel[:3]
        ])
        return self.rl_autopilot.compute_control(obs)
    
    def _log_telemetry(self):
        """Simple telemetry logging"""
        current_altitude = -self.state.position[2]
        vertical_velocity = -self.state.velocity[2]
        
        telemetry = {
            'timestamp': time.time(),
            'position': self.state.position.tolist(),
            'velocity': self.state.velocity.tolist(),
            'orientation': np.degrees(self.state.orientation).tolist(),
            'control_output': self.control_output.tolist(),
            'flight_mode': self.flight_mode.value,
            'altitude': current_altitude,
            'vertical_velocity': vertical_velocity
        }
        
        self.telemetry_history.append(telemetry)
        
        if len(self.telemetry_history) % 100 == 0:
            logger.info(f"Alt: {current_altitude:.1f}m | Vz: {vertical_velocity:.2f}m/s | Thr: {self.control_output[0]:.3f}")
    
    def set_flight_mode(self, mode: FlightMode):
        """Change flight mode"""
        if mode == self.flight_mode:
            return
        
        logger.info(f"Mode change: {self.flight_mode.value} → {mode.value}")
        
        self.altitude_pid.reset()
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.yaw_pid.reset()
        
        current_altitude = -self.estimated_state.position[2]
        
        if mode == FlightMode.ALTITUDE_HOLD:
            self.setpoints['altitude'] = current_altitude
            logger.info(f"Holding altitude at {current_altitude:.1f}m")
        
        elif mode == FlightMode.POSITION_HOLD:
            self.setpoints['position'] = self.estimated_state.position.copy()
            logger.info("Holding position")
        
        elif mode == FlightMode.AUTO:
            self.current_waypoint_index = 0
            self.mission_complete = False
            logger.info("Starting mission")
        
        elif mode == FlightMode.LAND:
            logger.info(f"Landing from {current_altitude:.1f}m")
        
        self.flight_mode = mode
    
    def set_waypoints(self, waypoints: List[np.ndarray]):
        """Set mission waypoints"""
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        self.mission_complete = False
        
        if waypoints:
            logger.info(f"Mission loaded: {len(waypoints)} waypoints")
    
    def get_telemetry(self) -> Dict:
        """Get telemetry for dashboard"""
        current_altitude = -self.estimated_state.position[2]
        vertical_velocity = -self.estimated_state.velocity[2]
        
        return {
            'position': [
                self.estimated_state.position[0],  # X (North)
                self.estimated_state.position[1],  # Y (East) 
                current_altitude  # Z (Altitude - positive!)
            ],
            'velocity': [
                self.estimated_state.velocity[0],
                self.estimated_state.velocity[1],
                vertical_velocity  # Positive for climb
            ],
            'attitude': self.estimated_state.orientation.tolist(),
            'flight_mode': self.flight_mode.value,
            'battery': 85.0,
            'gps_fix': True,
            'waypoint_index': self.current_waypoint_index,
            'mission_complete': self.mission_complete,
            'altitude': current_altitude,
            'vertical_velocity': vertical_velocity,
            'control_output': self.control_output.tolist()
        }
    
    def stop_logging(self):
        """Stop data logging"""
        self.data_logger.stop_logging()
    
    def get_flight_report(self):
        """Generate flight report"""
        return self.data_logger.generate_report()
    
    def get_recent_logs(self):
        """Get recent log files"""
        return self.data_logger.get_recent_logs()