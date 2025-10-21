# flight_controller.py - FIXED VERSION
"""
Flight Controller - STABLE WAYPOINT FOLLOWING
Proper state initialization and control logic
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
    Flight Controller with STABLE waypoint following
    """
    
    def __init__(self):
        # Core components
        self.dynamics = UAVDynamics()
        self.sensor_model = SensorModel()
        self.ekf = ExtendedKalmanFilter()
        
        # PROPERLY TUNED PID controllers for smooth flight
        self.altitude_pid = PIDController(
            kp=2.0,    # Stronger altitude control
            ki=0.2,    # Small integral  
            kd=1.2,    # Good damping
            output_limits=(-0.3, 0.3)
        )
        self.state = UAVState()
        self.state.position = np.array([0.0, 0.0, 0.0])  # Ground level in NED
        self.state.velocity = np.zeros(3)
        self.state.orientation = np.zeros(3)
        self.state.angular_velocity = np.zeros(3)
        self.state.motor_speeds = np.array([2000.0, 2000.0, 2000.0, 2000.0])
        self.state.acceleration = np.zeros(3)
        self.state.timestamp = time.time()
        self.estimated_state = UAVState()
        self.estimated_state.position = np.array([0.0, 0.0, 0.0])
        self.estimated_state.velocity = np.zeros(3)
        self.estimated_state.orientation = np.zeros(3)
        self.setpoints = {
            'altitude': 0.0,  # Ground level
            'position': np.array([0.0, 0.0, 0.0]),  # Ground position
            'yaw': 0.0
        }
        # Position controllers - CRITICAL FIX: Tune for smooth position control
        self.pos_x_pid = PIDController(1.2, 0.1, 0.8, (-0.4, 0.4))
        self.pos_y_pid = PIDController(1.2, 0.1, 0.8, (-0.4, 0.4))
        
        # Attitude controllers
        self.roll_pid = PIDController(2.5, 0.2, 0.6, (-0.5, 0.5))
        self.pitch_pid = PIDController(2.5, 0.2, 0.6, (-0.5, 0.5))
        self.yaw_pid = PIDController(1.5, 0.1, 0.3, (-0.3, 0.3))
        
        # Flight mode - start in MANUAL (on ground)
        self.flight_mode = FlightMode.MANUAL
        
        # FIXED: Proper state initialization at GROUND LEVEL [0, 0, 0]
        self.state = UAVState()
        self.state.position = np.array([0.0, 0.0, 0.0])  # Ground level in NED
        self.state.velocity = np.zeros(3)
        self.state.orientation = np.zeros(3)
        self.state.angular_velocity = np.zeros(3)
        self.state.motor_speeds = np.array([2000.0, 2000.0, 2000.0, 2000.0])  # Low RPM
        self.state.acceleration = np.zeros(3)
        self.state.timestamp = time.time()
        
        # Estimated state - also start at ground
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
        
        # Hover throttle - calibrated for stable hover
        self.hover_throttle = 0.62
        
        # Launched flag
        self.is_launched = False
        
        # Waypoint acceptance radius (meters)
        self.waypoint_radius = 2.0
        
        logger.info("✓ FlightController initialized - STABLE WAYPOINT VERSION")
    
    def update(self, manual_control: Optional[np.ndarray] = None) -> UAVState:
        current_time = time.time()
        dt = current_time - self.last_update
        if dt >= self.dt:
            try:
                # Step 1: Get sensor measurements from TRUE state
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

                # Step 4: Apply control to dynamics - THIS MOVES THE TRUE STATE
                self.state = self.dynamics.update(self.state, self.control_output, dt)

                # Step 5: Logging with CORRECT states
                self._log_telemetry()
                if current_time - self.last_log_time >= self.log_interval:
                    self.data_logger.log_data(self)
                    self.last_log_time = current_time

                self.last_update = current_time
        
            except Exception as e:
                logger.error(f"Control loop error: {e}")
                # Safe fallback
                self.control_output = np.array([0.4, 0.0, 0.0, 0.0])
        return self.state
    def _compute_control(self) -> np.ndarray:
        """
        Stable control routing
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
        current_altitude = -self.estimated_state.position[2]  # Convert NED to altitude
        target_altitude = self.setpoints['altitude']
        if target_altitude > 0.1 or current_altitude > 0.1:
            throttle_adjustment = self.altitude_pid.compute(
                setpoint=target_altitude,
                measurement=current_altitude,
                dt=self.dt
            )
            throttle = self.hover_throttle + throttle_adjustment
            throttle = np.clip(throttle, 0.4, 0.8)
        else:
            throttle = 0.0  
        target_pos = self.setpoints['position']
        current_pos = self.estimated_state.position
        pos_error = target_pos - current_pos
        # Convert position error to desired tilt angles (gentle)
        max_tilt_angle = 0.4  # radians (~23 degrees)
        desired_pitch = np.clip(-self.pos_x_pid.compute(0, pos_error[0], self.dt), 
                          -max_tilt_angle, max_tilt_angle)
        desired_roll = np.clip(self.pos_y_pid.compute(0, pos_error[1], self.dt), 
                          -max_tilt_angle, max_tilt_angle)
        # Attitude control
        roll = self.roll_pid.compute(desired_roll, self.estimated_state.orientation[0], self.dt)
        pitch = self.pitch_pid.compute(desired_pitch, self.estimated_state.orientation[1], self.dt)
        yaw = self.yaw_pid.compute(self.setpoints['yaw'], self.estimated_state.orientation[2], self.dt)
        return np.array([throttle, roll, pitch, yaw])
    
    def _auto_mode(self) -> np.ndarray:
        """
        AUTO: Smooth waypoint following
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
        if distance < self.waypoint_radius:
            if self.current_waypoint_index < len(self.waypoints) - 1:
                self.current_waypoint_index += 1
                next_wp = self.waypoints[self.current_waypoint_index]
                logger.info(f"✓ Waypoint {self.current_waypoint_index} reached! Next: N{next_wp[0]:.1f} E{next_wp[1]:.1f} Alt{-next_wp[2]:.1f}m")
            else:
                self.mission_complete = True
                logger.info("✓ All waypoints reached! Holding position.")
                # Hold at last waypoint
                self.setpoints['position'] = current_wp
                self.setpoints['altitude'] = -current_wp[2]
        return control
    
    def _rtl_mode(self) -> np.ndarray:
        """
        FIXED: Return to Launch - Go to [0,0] at safe altitude, then land at [0,0,0]
        """
        current_pos = self.estimated_state.position
        current_altitude = -current_pos[2]
    
        # Define positions
        home_position_xy = np.array([0.0, 0.0])
        safe_altitude = 5.0  # meters
        ground_position = np.array([0.0, 0.0, 0.0])  # Ground level in NED
    
        # Calculate horizontal distance to home
        distance_to_home_xy = np.linalg.norm(current_pos[:2] - home_position_xy)

        if distance_to_home_xy > 3.0 or current_altitude < safe_altitude - 1.0:
            # Phase 1: Go to home XY position at safe altitude
            target_position = np.array([0.0, 0.0, -safe_altitude])  # NED coordinates
            self.setpoints['position'] = target_position
            self.setpoints['altitude'] = safe_altitude
            logger.info("🔄 Returning to launch position...")
            return self._stabilize_mode()
        else:
            # Phase 2: At home position, now descend to land
            logger.info("✓ Home position reached, landing...")
            self.setpoints['position'] = ground_position
            self.setpoints['altitude'] = 0.0

            if current_altitude < 0.3:
                logger.info("✓ RTL complete - Landed at launch position")
                self.flight_mode = FlightMode.MANUAL
                self.is_launched = False
                return np.array([0.0, 0.0, 0.0, 0.0])  # Motors off

            return self._controlled_descent()
    def _land_mode(self) -> np.ndarray:
        """
        LAND: Land at current XY position
        """
        current_pos = self.estimated_state.position
        current_altitude = -current_pos[2]
        
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
        Smooth controlled descent for landing
        """
        current_altitude = -self.estimated_state.position[2]
        
        if current_altitude < 0.3:
            # Landed
            return np.array([0.0, 0.0, 0.0, 0.0])
        
        # Progressive descent rate based on altitude
        if current_altitude > 5.0:
            descent_rate = 1.0  # m/s
        elif current_altitude > 2.0:
            descent_rate = 0.5  # m/s
        else:
            descent_rate = 0.2  # m/s (very slow near ground)
        
        target_altitude = max(0.0, current_altitude - descent_rate * self.dt)
        
        throttle_adjustment = self.altitude_pid.compute(
            target_altitude,
            current_altitude,
            self.dt
        )
        
        throttle = self.hover_throttle + throttle_adjustment
        
        # Reduce throttle gradually as we approach ground
        if current_altitude < 2.0:
            throttle *= 0.8
        if current_altitude < 1.0:
            throttle *= 0.6
        
        throttle = np.clip(throttle, 0.2, 0.6)
        
        # Keep level attitude during descent
        roll = self.roll_pid.compute(0.0, self.estimated_state.orientation[0], self.dt)
        pitch = self.pitch_pid.compute(0.0, self.estimated_state.orientation[1], self.dt)
        yaw = self.yaw_pid.compute(0.0, self.estimated_state.orientation[2], self.dt)
        
        return np.array([throttle, roll, pitch, yaw])
    
    def launch(self, target_altitude: float = 5.0):
        if self.is_launched:
            logger.warning("Already launched!")
            return

        # FIXED: Don't require manual mode to launch
        if self.flight_mode not in [FlightMode.MANUAL, FlightMode.STABILIZE]:
            logger.warning(f"Cannot launch from {self.flight_mode.value} mode!")
            return

        # Store launch position
        self.launch_position = self.estimated_state.position.copy()

        # Set target altitude and position
        self.setpoints['altitude'] = target_altitude
        self.setpoints['position'] = np.array([0.0, 0.0, -target_altitude])  # NED: negative Z = positive altitude
        self.setpoints['yaw'] = 0.0

        # Reset PIDs for clean start
        self.altitude_pid.reset()
        self.pos_x_pid.reset()
        self.pos_y_pid.reset()
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.yaw_pid.reset()
    
        # FIXED: Use higher hover throttle for takeoff
        self.hover_throttle = 0.65  # Increase for takeoff
    
        # Switch to stabilize mode for takeoff
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
        waypoint = np.array([north, east, -abs(altitude)])
        self.waypoints.append(waypoint)
        logger.info(f"📍 Added waypoint: N{north:.1f} E{east:.1f} Alt{altitude:.1f}m")
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
        
        if not self.is_launched:
            logger.warning("Not launched yet! Auto-launching to 5m...")
            self.launch(5.0)
            time.sleep(3)  # Let it stabilize
        
        self.current_waypoint_index = 0
        self.mission_complete = False
        self.set_flight_mode(FlightMode.AUTO)
        logger.info(f"🎯 Mission started with {len(self.waypoints)} waypoints")

    def _log_telemetry(self):
        """Simple telemetry logging (include both true and estimated states)"""
        current_altitude = -self.state.position[2]
        vertical_velocity = -self.state.velocity[2]

        telemetry = {
            'timestamp': time.time(),
            # true (dynamics) NED position & velocity
            'true_position': self.state.position.tolist(),
            'true_velocity': self.state.velocity.tolist(),
            # estimated NED position & velocity (EKF)
            'estimated_position': self.estimated_state.position.tolist(),
            'estimated_velocity': self.estimated_state.velocity.tolist(),
            'orientation_deg': np.degrees(self.state.orientation).tolist(),
            'control_output': self.control_output.tolist(),
            'flight_mode': self.flight_mode.value,
            'altitude_true': current_altitude,
            'vertical_velocity_true': vertical_velocity
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
        """Get telemetry for dashboard - returns NED position (z negative = below origin)"""
        # NED position (keep sign convention consistent with dynamics / EKF)
        ned_pos = self.estimated_state.position.copy()  # shape (3,) -> N,E,Z (Z negative for altitude)
        ned_vel = self.estimated_state.velocity.copy()

        current_altitude = -ned_pos[2]  # positive altitude (for displays)
        vertical_velocity = -ned_vel[2]

        return {
            # Keep position as NED vector (do NOT put positive altitude in position[2])
            'position': [float(ned_pos[0]), float(ned_pos[1]), float(ned_pos[2])],
            # velocity in NED (vx, vy, vz)
            'velocity': [float(ned_vel[0]), float(ned_vel[1]), float(ned_vel[2])],
            # also expose altitude explicitly so consumers don't guess sign
            'altitude': float(current_altitude),
            'vertical_velocity': float(vertical_velocity),
            'attitude': self.estimated_state.orientation.tolist(),  # roll, pitch, yaw (radians)
            'flight_mode': self.flight_mode.value,
            'battery': 85.0,
            'gps_fix': True,
            'waypoint_index': self.current_waypoint_index,
            'mission_complete': self.mission_complete,
            'control_output': self.control_output.tolist(),
            'is_launched': self.is_launched,
            # add estimated_state vector for debugging/tracing if needed
            'estimated_position': self.estimated_state.position.tolist(),
            'true_position': self.state.position.tolist()  # true (dynamics) state for cross-check
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