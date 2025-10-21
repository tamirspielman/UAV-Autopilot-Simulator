"""
Flight Controller - FIXED VERSION
Simplified flight modes and proper stabilization
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
    Flight Controller with FIXED stabilization and simplified modes
    """
    
    def __init__(self):
        # Core components
        self.dynamics = UAVDynamics()
        self.sensor_model = SensorModel()
        self.ekf = ExtendedKalmanFilter()
        
        # FIXED: Better tuned PID controllers
        self.altitude_pid = PIDController(
            kp=0.8,    # Stronger proportional
            ki=0.05,   # Small integral  
            kd=0.4,    # Good damping
            output_limits=(-0.15, 0.15)
        )
        
        # Position controllers (for holding position)
        self.pos_x_pid = PIDController(0.3, 0.01, 0.2, (-0.15, 0.15))
        self.pos_y_pid = PIDController(0.3, 0.01, 0.2, (-0.15, 0.15))
        
        # Attitude controllers
        self.roll_pid = PIDController(1.2, 0.02, 0.25, (-0.25, 0.25))
        self.pitch_pid = PIDController(1.2, 0.02, 0.25, (-0.25, 0.25))
        self.yaw_pid = PIDController(0.8, 0.01, 0.15, (-0.15, 0.15))
        
        # Flight mode - start in MANUAL (on ground)
        self.flight_mode = FlightMode.MANUAL
        
        # FIXED: Initialize state at GROUND LEVEL [0, 0, 0]
        self.state = UAVState()
        self.state.position = np.array([0.0, 0.0, 0.0])  # Ground level in NED
        self.state.velocity = np.zeros(3)
        self.state.orientation = np.zeros(3)
        self.state.angular_velocity = np.zeros(3)
        self.state.motor_speeds = np.array([2000.0, 2000.0, 2000.0, 2000.0])  # Low RPM
        self.state.acceleration = np.zeros(3)
        self.state.timestamp = 0.0
        
        # Estimated state
        self.estimated_state = UAVState()
        self.estimated_state.position = np.array([0.0, 0.0, 0.0])
        self.estimated_state.velocity = np.zeros(3)
        self.estimated_state.orientation = np.zeros(3)
        
        # Sensor data
        self.sensor_data = SensorData()
        
        # Setpoints - start at ground
        self.setpoints = {
            'altitude': 0.0,  # Ground level
            'position': np.array([0.0, 0.0, 0.0]),  # Ground position
            'yaw': 0.0
        }
        
        # Mission waypoints
        self.waypoints = []
        self.current_waypoint_index = 0
        self.mission_complete = False
        self.launch_position = np.array([0.0, 0.0, 0.0])  # Store launch position
        
        # Control output - START WITH ZERO (on ground)
        self.control_output = np.array([0.0, 0.0, 0.0, 0.0])
        
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
        
        # Launched flag
        self.is_launched = False
        
        logger.info("✓ FlightController initialized - FIXED VERSION")
    
    def update(self, manual_control: Optional[np.ndarray] = None) -> UAVState:
        """
        Main control loop - FIXED
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
                
                # Step 5: Ground collision protection
                if self.state.position[2] > 0.0:
                    self.state.position[2] = 0.0
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
                self.control_output = np.array([0.0, 0.0, 0.0, 0.0])
        
        return self.state
    
    def _compute_control(self) -> np.ndarray:
        """
        Simplified control routing - only 3 modes
        """
        if self.flight_mode == FlightMode.MANUAL:
            return np.array([0.0, 0.0, 0.0, 0.0])  # Motors off on ground
        
        elif self.flight_mode == FlightMode.STABILIZE:
            return self._stabilize_mode()
        
        elif self.flight_mode == FlightMode.AUTO:
            return self._auto_mode()
        
        elif self.flight_mode == FlightMode.RTL:
            return self._rtl_mode()
        
        elif self.flight_mode == FlightMode.LAND:
            return self._land_mode()
        
        else:
            return np.array([0.0, 0.0, 0.0, 0.0])
    
    def _stabilize_mode(self) -> np.ndarray:
        """
        FIXED: Stabilize at current altitude and hold position
        """
        current_altitude = -self.estimated_state.position[2]
        target_altitude = self.setpoints['altitude']
        
        # Altitude control
        throttle_adjustment = self.altitude_pid.compute(
            setpoint=target_altitude,
            measurement=current_altitude,
            dt=self.dt
        )
        
        throttle = self.hover_throttle + throttle_adjustment
        throttle = np.clip(throttle, 0.3, 0.7)
        
        # FIXED: Position hold - use target position instead of zero
        target_pos = self.setpoints['position'][:2]
        current_pos = self.estimated_state.position[:2]
        
        pos_error = target_pos - current_pos
        
        # Convert position error to desired tilt angles
        # Pitch controls North/South (X), Roll controls East/West (Y)
        desired_pitch = -self.pos_x_pid.compute(0, pos_error[0], self.dt)
        desired_roll = self.pos_y_pid.compute(0, pos_error[1], self.dt)
        
        # Attitude control
        roll = self.roll_pid.compute(desired_roll, self.estimated_state.orientation[0], self.dt)
        pitch = self.pitch_pid.compute(desired_pitch, self.estimated_state.orientation[1], self.dt)
        yaw = self.yaw_pid.compute(self.setpoints['yaw'], self.estimated_state.orientation[2], self.dt)
        
        return np.array([throttle, roll, pitch, yaw])
    
    def _auto_mode(self) -> np.ndarray:
        """
        AUTO: Follow waypoints in sequence
        """
        if not self.waypoints or self.mission_complete:
            # No waypoints, just hold current position
            return self._stabilize_mode()
        
        # Get current waypoint
        current_wp = self.waypoints[self.current_waypoint_index]
        
        # Update setpoint to current waypoint
        self.setpoints['position'] = current_wp
        self.setpoints['altitude'] = -current_wp[2]  # Convert NED to altitude
        
        # Use stabilize control to reach waypoint
        control = self._stabilize_mode()
        
        # Check if waypoint reached
        current_pos = self.estimated_state.position
        distance = np.linalg.norm(current_pos - current_wp)
        
        if distance < 1.5:  # Within 1.5m of waypoint
            if self.current_waypoint_index < len(self.waypoints) - 1:
                self.current_waypoint_index += 1
                next_wp = self.waypoints[self.current_waypoint_index]
                logger.info(f"✓ Waypoint {self.current_waypoint_index} reached, going to next: {next_wp}")
            else:
                self.mission_complete = True
                logger.info("✓ All waypoints reached!")
        
        return control
    
    def _rtl_mode(self) -> np.ndarray:
        """
        RTL: Return to Launch - Go to last waypoint, then return to [0,0,0] and land
        """
        current_pos = self.estimated_state.position
        
        # Phase 1: If we have waypoints, go to last waypoint first
        if self.waypoints and not self.mission_complete:
            last_wp = self.waypoints[-1]
            distance_to_last = np.linalg.norm(current_pos - last_wp)
            
            if distance_to_last > 1.5:
                # Still going to last waypoint
                self.setpoints['position'] = last_wp
                self.setpoints['altitude'] = -last_wp[2]
                return self._stabilize_mode()
            else:
                # Reached last waypoint
                self.mission_complete = True
                logger.info("✓ Reached last waypoint, returning home...")
        
        # Phase 2: Return to launch position [0, 0, 0]
        home = self.launch_position.copy()
        home[2] = -2.0  # Go to 2m altitude first
        
        distance_to_home = np.linalg.norm(current_pos[:2] - home[:2])
        
        if distance_to_home > 1.0:
            # Flying home at 2m
            self.setpoints['position'] = home
            self.setpoints['altitude'] = 2.0
            return self._stabilize_mode()
        else:
            # At home position, now land
            logger.info("✓ Home position reached, landing...")
            self.setpoints['position'] = self.launch_position
            self.setpoints['altitude'] = 0.0
            
            current_altitude = -self.estimated_state.position[2]
            if current_altitude < 0.3:
                logger.info("✓ RTL complete - Landed at launch position")
                self.flight_mode = FlightMode.MANUAL
                return np.array([0.0, 0.0, 0.0, 0.0])
            
            return self._controlled_descent()
    
    def _land_mode(self) -> np.ndarray:
        """
        LAND: Go to last waypoint, then land at current location
        """
        current_pos = self.estimated_state.position
        
        # Phase 1: If we have waypoints and haven't completed mission, go to last waypoint
        if self.waypoints and not self.mission_complete:
            last_wp = self.waypoints[-1]
            distance_to_last = np.linalg.norm(current_pos - last_wp)
            
            if distance_to_last > 1.5:
                # Still going to last waypoint
                self.setpoints['position'] = last_wp
                self.setpoints['altitude'] = -last_wp[2]
                return self._stabilize_mode()
            else:
                # Reached last waypoint
                self.mission_complete = True
                logger.info("✓ Reached last waypoint, landing here...")
        
        # Phase 2: Land at current XY position
        current_altitude = -self.estimated_state.position[2]
        
        if current_altitude < 0.3:
            logger.info("✓ Landed at current position")
            self.flight_mode = FlightMode.MANUAL
            return np.array([0.0, 0.0, 0.0, 0.0])
        
        # Set landing target to current XY, altitude 0
        land_target = current_pos.copy()
        land_target[2] = 0.0
        self.setpoints['position'] = land_target
        self.setpoints['altitude'] = 0.0
        
        return self._controlled_descent()
    
    def _controlled_descent(self) -> np.ndarray:
        """
        Controlled descent for landing
        """
        current_altitude = -self.estimated_state.position[2]
        
        # Slow descent rate
        descent_rate = 0.5  # m/s
        target_altitude = max(0.0, current_altitude - descent_rate * self.dt)
        
        throttle_adjustment = self.altitude_pid.compute(
            target_altitude,
            current_altitude,
            self.dt
        )
        
        throttle = self.hover_throttle + throttle_adjustment
        
        # Reduce throttle gradually as we approach ground
        if current_altitude < 1.0:
            throttle *= 0.7
        
        throttle = np.clip(throttle, 0.2, 0.6)
        
        # Keep level attitude
        roll = self.roll_pid.compute(0.0, self.estimated_state.orientation[0], self.dt)
        pitch = self.pitch_pid.compute(0.0, self.estimated_state.orientation[1], self.dt)
        yaw = self.yaw_pid.compute(0.0, self.estimated_state.orientation[2], self.dt)
        
        return np.array([throttle, roll, pitch, yaw])
    
    def launch(self, target_altitude: float = 2.0):
        """
        Launch sequence - take off to specified altitude
        """
        if self.is_launched:
            logger.warning("Already launched!")
            return
        
        # Store launch position
        self.launch_position = self.estimated_state.position.copy()
        
        # Set target altitude and position
        self.setpoints['altitude'] = target_altitude
        self.setpoints['position'] = np.array([0.0, 0.0, -target_altitude])
        self.setpoints['yaw'] = 0.0
        
        # Switch to stabilize mode
        self.set_flight_mode(FlightMode.STABILIZE)
        self.is_launched = True
        
        logger.info(f"🚀 Launching to {target_altitude}m altitude!")
    
    def emergency_land(self):
        """
        EMERGENCY LAND: Immediate landing at current position
        """
        logger.warning("⚠️ EMERGENCY LANDING!")
        
        # Set landing target to current position
        current_pos = self.estimated_state.position.copy()
        current_pos[2] = 0.0  # Ground level
        
        self.setpoints['position'] = current_pos
        self.setpoints['altitude'] = 0.0
        self.mission_complete = True  # Skip any waypoint navigation
        
        # Force land mode
        self.flight_mode = FlightMode.LAND
    
    def add_waypoint(self, north: float, east: float, altitude: float):
        """
        Add waypoint in NED coordinates
        north: meters north
        east: meters east  
        altitude: meters above ground (will be converted to NED -Z)
        """
        waypoint = np.array([north, east, -abs(altitude)])
        self.waypoints.append(waypoint)
        logger.info(f"Added waypoint: N{north} E{east} Alt{altitude}m")
    
    def clear_waypoints(self):
        """Clear all waypoints"""
        self.waypoints = []
        self.current_waypoint_index = 0
        self.mission_complete = False
        logger.info("Waypoints cleared")
    
    def start_mission(self):
        """Start autonomous mission"""
        if not self.waypoints:
            logger.warning("No waypoints set!")
            return
        
        self.current_waypoint_index = 0
        self.mission_complete = False
        self.set_flight_mode(FlightMode.AUTO)
        logger.info(f"🎯 Mission started with {len(self.waypoints)} waypoints")
    
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
    
    def set_flight_mode(self, mode: FlightMode):
        """Change flight mode"""
        if mode == self.flight_mode:
            return
        
        logger.info(f"Mode change: {self.flight_mode.value} → {mode.value}")
        
        # Reset PIDs
        self.altitude_pid.reset()
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.yaw_pid.reset()
        self.pos_x_pid.reset()
        self.pos_y_pid.reset()
        
        # Mode-specific initialization
        if mode == FlightMode.STABILIZE:
            # Hold current altitude and position
            self.setpoints['altitude'] = -self.estimated_state.position[2]
            self.setpoints['position'] = self.estimated_state.position.copy()
        
        self.flight_mode = mode
    
    def get_telemetry(self) -> Dict:
        """Get telemetry for dashboard"""
        current_altitude = -self.estimated_state.position[2]
        vertical_velocity = -self.estimated_state.velocity[2]
        
        return {
            'position': [
                self.estimated_state.position[0],
                self.estimated_state.position[1],
                current_altitude
            ],
            'velocity': [
                self.estimated_state.velocity[0],
                self.estimated_state.velocity[1],
                vertical_velocity
            ],
            'attitude': self.estimated_state.orientation.tolist(),
            'flight_mode': self.flight_mode.value,
            'battery': 85.0,
            'gps_fix': True,
            'waypoint_index': self.current_waypoint_index,
            'mission_complete': self.mission_complete,
            'altitude': current_altitude,
            'vertical_velocity': vertical_velocity,
            'control_output': self.control_output.tolist(),
            'is_launched': self.is_launched
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