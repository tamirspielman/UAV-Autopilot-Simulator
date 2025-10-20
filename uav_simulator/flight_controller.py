"""
Flight Controller - FIXED Hover Throttle Calibration
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
    Flight Controller with Proper Hover Throttle Calibration
    """
    
    def __init__(self):
        # Initialize components
        self.dynamics = UAVDynamics()
        self.sensor_model = SensorModel()
        self.ekf = ExtendedKalmanFilter()
        
        # FIXED: Better tuned PID controllers
        self.altitude_pid = PIDController(
            kp=1.2,    # Increased proportional for faster response
            ki=0.08,   # Increased integral for better steady-state
            kd=0.3,    # Increased derivative for damping
            output_limits=(-0.25, 0.25)  # Slightly wider limits
        )
        
        # Attitude controllers
        self.roll_pid = PIDController(1.5, 0.02, 0.15, (-0.4, 0.4))
        self.pitch_pid = PIDController(1.5, 0.02, 0.15, (-0.4, 0.4))
        self.yaw_pid = PIDController(1.0, 0.01, 0.1, (-0.3, 0.3))

        # RL autopilot
        self.rl_autopilot = RLAutopilot()
        
        # CRITICAL FIX: Initialize state with proper values
        self.state = UAVState()
        # Start at 10m altitude in NED: position[2] = -10.0
        self.state.position = np.array([0.0, 0.0, -10.0])
        self.state.velocity = np.zeros(3)
        self.state.orientation = np.zeros(3)
        self.state.angular_velocity = np.zeros(3)
        self.state.motor_speeds = np.array([5000, 5000, 5000, 5000])  # Near-hover RPM
        
        # Initialize estimated state
        self.estimated_state = UAVState()
        self.estimated_state.position = np.array([0.0, 0.0, -10.0])
        
        self.sensor_data = SensorData()
        
        # Flight mode and setpoints
        self.flight_mode = FlightMode.STABILIZE
        self.setpoints = {
            'altitude': 10.0,  # Meters above ground
            'position': np.array([0.0, 0.0, -10.0]),  # NED coordinates
            'velocity': np.zeros(3),
            'attitude': np.zeros(3)
        }
        
        # Mission waypoints
        self.waypoints = []
        self.current_waypoint_index = 0
        self.mission_complete = False
        
        # Control outputs - FIXED: Adjusted hover throttle
        self.control_output = np.array([0.55, 0.0, 0.0, 0.0])  # Reduced from 0.58
        
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
        
        logger.info("FlightController initialized - Starting at 10m altitude")
        
    def update(self, manual_control: Optional[np.ndarray] = None) -> UAVState:
        """Main update loop with comprehensive safety checks"""
        current_time = time.time()
        dt = current_time - self.last_update
        
        if dt >= self.dt:
            try:
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
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                # Emergency recovery - maintain hover
                self.control_output = np.array([0.55, 0.0, 0.0, 0.0])
        
        return self.state
    
    def _compute_autonomous_control(self) -> np.ndarray:
        """Compute control based on flight mode"""
        # Always check altitude safety first
        emergency_throttle = self._check_altitude_safety()
        if emergency_throttle is not None:
            return np.array([emergency_throttle, 0.0, 0.0, 0.0])
            
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
        Improved altitude control with adaptive hover throttle
        """
        # Get current altitude above ground
        current_down = self.estimated_state.position[2]
        current_altitude = -current_down

        # Calculate error
        altitude_error = target_altitude - current_altitude

        # Adaptive deadzone based on altitude
        deadzone = max(0.1, min(0.5, abs(current_altitude) * 0.01))
        if abs(altitude_error) < deadzone:
            altitude_error = 0

        # Compute PID adjustment
        throttle_adjustment = self.altitude_pid.compute(0, altitude_error, self.dt)

        # FIXED: Adjusted base hover throttle - this is the key change
        # If drone is ascending, reduce hover throttle. If descending, increase it.
        hover_throttle = 0.55  # Reduced from 0.58 based on observed behavior

        # Calculate final throttle
        throttle = hover_throttle + throttle_adjustment

        # Safe limits
        throttle = np.clip(throttle, 0.40, 0.70)
        
        # Debug logging - more informative
        if int(time.time()) % 5 == 0:
            logger.info(f"Altitude: {current_altitude:.1f}m -> {target_altitude:.1f}m, Error: {altitude_error:.2f}m, Throttle: {throttle:.3f}")

        return throttle
    
    def _check_altitude_safety(self):
        """
        Safety check - prevent extreme altitude values
        """
        current_down = self.estimated_state.position[2]
        current_altitude = -current_down

        # If altitude is extremely negative (way below ground)
        if current_altitude < -100:
            logger.warning(f"CRITICAL: Altitude emergency! {current_altitude:.1f}m below ground")
            return 0.65  # Reduced maximum climb
        
        # If altitude is extremely high
        if current_altitude > 500:
            logger.warning(f"CRITICAL: Altitude too high! {current_altitude:.1f}m")
            return 0.48  # Increased descent
    
        return None
    
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
        """Altitude hold with level attitude"""
        throttle = self._compute_altitude_control(self.setpoints['altitude'])
        
        # Keep level attitude
        roll = self.roll_pid.compute(0, self.estimated_state.orientation[0], self.dt)
        pitch = self.pitch_pid.compute(0, self.estimated_state.orientation[1], self.dt)
        yaw = self.yaw_pid.compute(0, self.estimated_state.orientation[2], self.dt)
        
        return np.array([throttle, roll, pitch, yaw])
    
    def _compute_position_hold_control(self) -> np.ndarray:
        """Position hold control"""
        # Horizontal control
        pos_error = self.setpoints['position'][:2] - self.estimated_state.position[:2]
        desired_velocity = np.clip(pos_error * 0.2, -1.0, 1.0)
        vel_error = desired_velocity - self.estimated_state.velocity[:2]
        
        # Velocity to attitude
        desired_pitch = np.clip(vel_error[0] * 0.1, -0.2, 0.2)
        desired_roll = np.clip(-vel_error[1] * 0.1, -0.2, 0.2)
        
        # Altitude control
        target_altitude = -self.setpoints['position'][2]
        throttle = self._compute_altitude_control(target_altitude)
        
        # Attitude control
        roll = self.roll_pid.compute(desired_roll, self.estimated_state.orientation[0], self.dt)
        pitch = self.pitch_pid.compute(desired_pitch, self.estimated_state.orientation[1], self.dt)
        yaw = self.yaw_pid.compute(0, self.estimated_state.orientation[2], self.dt)
        
        return np.array([throttle, roll, pitch, yaw])
    
    def _compute_auto_control(self) -> np.ndarray:
        """Auto mission control"""
        if not self.waypoints or self.mission_complete:
            return self._compute_position_hold_control()

        idx = max(0, min(self.current_waypoint_index, len(self.waypoints) - 1))
        current_wp = self.waypoints[idx]

        # Set target
        self.setpoints['position'] = current_wp
        target_altitude = -current_wp[2]
        self.setpoints['altitude'] = target_altitude

        # Use position hold to navigate to waypoint
        control = self._compute_position_hold_control()

        # Check if waypoint reached
        distance = np.linalg.norm(self.estimated_state.position - current_wp)
        if distance < self.waypoint_acceptance_radius:
            if self.current_waypoint_index < len(self.waypoints) - 1:
                logger.info(f"Waypoint {self.current_waypoint_index + 1} reached")
                self.current_waypoint_index += 1
            else:
                logger.info("Mission complete!")
                self.mission_complete = True

        return control
    
    def _compute_rtl_control(self) -> np.ndarray:
        """Return to Launch"""
        home_position = np.array([0.0, 0.0, -10.0])  # Home at 10m altitude
        
        # Set home as target
        self.setpoints['position'] = home_position
        self.setpoints['altitude'] = 10.0
        
        # Use position hold to navigate home
        control = self._compute_position_hold_control()
        
        # Check if close to home for landing
        distance_to_home = np.linalg.norm(self.estimated_state.position - home_position)
        if distance_to_home < 3.0:
            logger.info("Close to home, initiating landing")
            self.set_flight_mode(FlightMode.LAND)
        
        return control
    
    def _compute_land_control(self) -> np.ndarray:
        """Safe landing control"""
        current_altitude = -self.estimated_state.position[2]
        
        # If very close to ground, cut motors
        if current_altitude < 0.5:
            logger.info("✅ Landed successfully!")
            return np.array([0.0, 0.0, 0.0, 0.0])
        
        # Smooth descent
        target_altitude = max(0, current_altitude - 0.5 * self.dt)
        
        # Use altitude control for smooth descent
        throttle = self._compute_altitude_control(target_altitude)
        
        # Keep level attitude
        roll = self.roll_pid.compute(0, self.estimated_state.orientation[0], self.dt)
        pitch = self.pitch_pid.compute(0, self.estimated_state.orientation[1], self.dt)
        yaw = self.yaw_pid.compute(0, self.estimated_state.orientation[2], self.dt)
        
        # Gradually reduce throttle as we approach ground
        if current_altitude < 5.0:
            throttle_factor = current_altitude / 5.0
            throttle = throttle * throttle_factor
        
        return np.array([throttle, roll, pitch, yaw])
    
    def _compute_stabilize_control(self) -> np.ndarray:
        """Stabilize mode - maintain current altitude and level attitude"""
        # Maintain current altitude
        current_altitude = -self.estimated_state.position[2]
        throttle = self._compute_altitude_control(current_altitude)
        
        # Keep level attitude
        roll = self.roll_pid.compute(0, self.estimated_state.orientation[0], self.dt)
        pitch = self.pitch_pid.compute(0, self.estimated_state.orientation[1], self.dt)
        yaw = self.yaw_pid.compute(0, self.estimated_state.orientation[2], self.dt)
        
        return np.array([throttle, roll, pitch, yaw])
    
    def _update_mission(self):
        """Update mission state"""
        pass
    
    def _log_telemetry(self):
        """Telemetry logging"""
        telemetry = {
            'timestamp': time.time(),
            'position': self.state.position.tolist(),
            'velocity': self.state.velocity.tolist(),
            'orientation': np.degrees(self.state.orientation).tolist(),
            'control_output': self.control_output.tolist(),
            'flight_mode': self.flight_mode.value,
            'altitude': -self.state.position[2],
            'motor_speeds': self.state.motor_speeds.tolist()
        }

        # Log key parameters occasionally
        if int(time.time()) % 10 == 0:
            altitude = -self.state.position[2]
            logger.info(f"Mode: {self.flight_mode.value}, Alt: {altitude:.1f}m, Throttle: {self.control_output[0]:.3f}")

        self.telemetry_history.append(telemetry)
    
    def set_flight_mode(self, mode: FlightMode):
        """Change flight mode with proper initialization"""
        if mode != self.flight_mode:
            logger.info(f"Flight mode changing from {self.flight_mode.value} to {mode.value}")

            # Reset all controllers
            self.altitude_pid.reset()
            self.roll_pid.reset()
            self.pitch_pid.reset()
            self.yaw_pid.reset()

            # Mode-specific initialization
            current_altitude = -self.estimated_state.position[2]
            
            if mode == FlightMode.ALTITUDE_HOLD:
                self.setpoints['altitude'] = current_altitude
            elif mode == FlightMode.POSITION_HOLD:
                self.setpoints['position'] = self.estimated_state.position.copy()
                self.setpoints['altitude'] = current_altitude
            elif mode == FlightMode.LAND:
                logger.info(f"Starting landing from {current_altitude:.1f}m")

            self.flight_mode = mode
    
    def set_waypoints(self, waypoints: List[np.ndarray]):
        """Set waypoints (in NED coordinates)"""
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        self.mission_complete = False
        logger.info(f"Mission: {len(waypoints)} waypoints set")
    
    def get_telemetry(self) -> Dict:
        """Get telemetry for dashboard"""
        current_altitude = -self.estimated_state.position[2]
        
        return {
            'position': self.estimated_state.position.tolist(),
            'velocity': self.estimated_state.velocity.tolist(),
            'attitude': self.estimated_state.orientation.tolist(),
            'flight_mode': self.flight_mode.value,
            'battery': 85.0,
            'gps_fix': True,
            'waypoint_index': self.current_waypoint_index,
            'mission_complete': self.mission_complete,
            'altitude': current_altitude
        }
    
    def stop_logging(self):
        """Stop data logging"""
        self.data_logger.stop_logging()
    
    def get_flight_report(self):
        """Get flight report"""
        return self.data_logger.generate_report()
    
    def get_recent_logs(self):
        """Get recent log files"""
        return self.data_logger.get_recent_logs()