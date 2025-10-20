"""
Flight Controller - FULLY FIXED Altitude and Position Control
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
    Flight Controller with CORRECTED Coordinate System and Control Logic
    """
    
    def __init__(self):
        # Initialize components
        self.dynamics = UAVDynamics()
        self.sensor_model = SensorModel()
        self.ekf = ExtendedKalmanFilter()
        
        # FIXED: More aggressive PID controllers for better response
        self.altitude_pid = PIDController(
            kp=2.0,    # Increased for better response
            ki=0.1,    # Increased for steady-state
            kd=0.5,    # Increased for damping
            output_limits=(-0.3, 0.3)
        )
        
        # Attitude controllers - more aggressive
        self.roll_pid = PIDController(3.0, 0.2, 0.5, (-0.5, 0.5))
        self.pitch_pid = PIDController(3.0, 0.2, 0.5, (-0.5, 0.5))
        self.yaw_pid = PIDController(2.0, 0.1, 0.3, (-0.3, 0.3))

        # RL autopilot
        self.rl_autopilot = RLAutopilot()
        
        # CRITICAL FIX: Initialize state consistently
        self.state = UAVState()
        self.state.position = np.array([0.0, 0.0, -10.0])  # NED: -10 = 10m altitude
        self.state.velocity = np.zeros(3)
        self.state.orientation = np.zeros(3)
        self.state.angular_velocity = np.zeros(3)
        self.state.motor_speeds = np.array([5200, 5200, 5200, 5200])
        
        # FIXED: Initialize estimated state to match
        self.estimated_state = UAVState()
        self.estimated_state.position = np.array([0.0, 0.0, -10.0])  # Consistent with state
        self.estimated_state.velocity = np.zeros(3)
        self.estimated_state.orientation = np.zeros(3)
        
        self.sensor_data = SensorData()
        
        # Flight mode and setpoints
        self.flight_mode = FlightMode.STABILIZE
        self.setpoints = {
            'altitude': 10.0,  # Meters above ground (up-positive for user interface)
            'position': np.array([0.0, 0.0, -10.0]),  # NED coordinates
            'velocity': np.zeros(3),
            'attitude': np.zeros(3)
        }
        
        # Mission waypoints
        self.waypoints = []
        self.current_waypoint_index = 0
        self.mission_complete = False
        
        # Control outputs
        self.control_output = np.array([0.55, 0.0, 0.0, 0.0]) 
        
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
                self.control_output = np.array([0.52, 0.0, 0.0, 0.0])
        
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
        FIXED: Correct altitude PID controller aligned with NED frame.
        target_altitude: desired altitude in meters above ground (up-positive)
        """
        # Convert NED Down coordinate to Up-positive altitude
        current_altitude_up = -self.estimated_state.position[2]
        
        # FIXED: Correct error sign - positive error means we need to go UP
        altitude_error = target_altitude - current_altitude_up
        
        # Small deadband to prevent oscillation
        if abs(altitude_error) < 0.1:
            altitude_error = 0
        
        # PID correction
        throttle_adjustment = self.altitude_pid.compute(target_altitude, current_altitude_up, self.dt)
        
        # Hover throttle tuned for level flight
        hover_throttle = 0.55
        throttle = hover_throttle + throttle_adjustment
        
        # Safety clamp
        throttle = np.clip(throttle, 0.35, 0.75)
        
        # Periodic logging
        if int(time.time() * 2) % 10 == 0:  # Log every 5 seconds
            logger.info(
                f"AltCtrl | Current={current_altitude_up:.2f}m Target={target_altitude:.2f}m "
                f"Error={altitude_error:.2f}m Throttle={throttle:.3f}"
            )
        
        return throttle
    
    def _check_altitude_safety(self):
        """Check altitude bounds and return emergency throttle if needed"""
        current_altitude_up = -self.estimated_state.position[2]

        if current_altitude_up < 0.5:
            logger.warning(f"Too low! Alt={current_altitude_up:.2f}m - Climbing")
            return 0.65  # Strong climb
        elif current_altitude_up > 100.0:
            logger.warning(f"Too high! Alt={current_altitude_up:.2f}m - Descending")
            return 0.40  # Strong descend
        
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
        """
        FIXED: Position hold control with correct NED frame handling
        """
        # Get current position (NED)
        current_pos = self.estimated_state.position[:2]  # [North, East]
        target_pos = self.setpoints['position'][:2]      # [North, East]

        # Position error in NED frame
        pos_error = target_pos - current_pos
        
        # Convert position error to desired velocity
        max_velocity = 3.0  # m/s - increased for better response
        kp_pos = 0.8  # Position gain
        desired_velocity = np.clip(pos_error * kp_pos, -max_velocity, max_velocity)

        # Current velocity (NED)
        current_vel = self.estimated_state.velocity[:2]

        # Velocity error
        vel_error = desired_velocity - current_vel
        
        # FIXED: Convert velocity to attitude commands with CORRECT signs
        # In NED frame with X-configuration:
        # - Positive pitch (nose up) moves BACKWARD (negative North)
        # - Positive roll (right wing down) moves RIGHT (positive East)
        max_angle = 0.35  # radians (~20 degrees)
        kv = 0.3  # Velocity to angle gain
        
        # North velocity error -> NEGATIVE pitch (pitch forward to go north)
        desired_pitch = np.clip(-vel_error[0] * kv, -max_angle, max_angle)
        
        # East velocity error -> POSITIVE roll (roll right to go east)
        desired_roll = np.clip(vel_error[1] * kv, -max_angle, max_angle)
        
        # Altitude control - convert NED to altitude
        target_altitude = -self.setpoints['position'][2]
        throttle = self._compute_altitude_control(target_altitude)
        
        # Attitude control to achieve desired angles
        roll_output = self.roll_pid.compute(desired_roll, self.estimated_state.orientation[0], self.dt)
        pitch_output = self.pitch_pid.compute(desired_pitch, self.estimated_state.orientation[1], self.dt)
        yaw_output = self.yaw_pid.compute(self.setpoints['attitude'][2], 
                                         self.estimated_state.orientation[2], self.dt)
        
        # Periodic logging for debugging
        if int(time.time() * 2) % 10 == 0:
            logger.info(
                f"PosHold | PosErr: N{pos_error[0]:.1f} E{pos_error[1]:.1f} | "
                f"DesAtt: R{np.degrees(desired_roll):.1f}° P{np.degrees(desired_pitch):.1f}°"
            )
        
        return np.array([throttle, roll_output, pitch_output, yaw_output])
    
    def _compute_auto_control(self) -> np.ndarray:
        """FIXED: Auto mission control with proper waypoint navigation"""
        if not self.waypoints or self.mission_complete:
            # Hold current position if no mission
            return self._compute_position_hold_control()

        # Get current waypoint
        current_wp = self.waypoints[self.current_waypoint_index]

        # Set target position (NED)
        self.setpoints['position'] = current_wp.copy()
        target_altitude = -current_wp[2]  # Convert NED to up-positive altitude
        self.setpoints['altitude'] = target_altitude

        # Use position hold to navigate to waypoint
        control = self._compute_position_hold_control()

        # Check if waypoint reached
        current_pos = self.estimated_state.position
        horizontal_distance = np.linalg.norm(current_pos[:2] - current_wp[:2])
        vertical_distance = abs(current_pos[2] - current_wp[2])

        # Waypoint acceptance criteria
        horizontal_acceptance = 2.5  # meters
        vertical_acceptance = 1.5    # meters
        
        if horizontal_distance < horizontal_acceptance and vertical_distance < vertical_acceptance:
            if self.current_waypoint_index < len(self.waypoints) - 1:
                logger.info(f"✓ Waypoint {self.current_waypoint_index + 1} reached! Moving to next...")
                self.current_waypoint_index += 1
            else:
                logger.info("🎉 Mission complete! All waypoints reached!")
                self.mission_complete = True
        
        # Regular progress logging
        if int(time.time() * 2) % 10 == 0:
            logger.info(
                f"Mission {self.current_waypoint_index + 1}/{len(self.waypoints)} | "
                f"Dist: H{horizontal_distance:.1f}m V{vertical_distance:.1f}m"
            )

        return control
    
    def _compute_rtl_control(self) -> np.ndarray:
        """Return to Launch"""
        # Home position in NED
        home_position = np.array([0.0, 0.0, -10.0])
        
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
        """FIXED: Safe landing control"""
        # Get current altitude (up-positive)
        current_altitude = -self.estimated_state.position[2]
        
        # If very close to ground, cut motors
        if current_altitude < 0.3:
            logger.info("✅ Landed successfully!")
            return np.array([0.0, 0.0, 0.0, 0.0])
        
        # Gradual descent rate: 0.5 m/s
        descent_rate = 0.5  # m/s
        target_altitude = max(0, current_altitude - descent_rate * self.dt)
        
        # Use altitude control for smooth descent
        throttle = self._compute_altitude_control(target_altitude)
        
        # Keep level attitude during landing
        roll = self.roll_pid.compute(0, self.estimated_state.orientation[0], self.dt)
        pitch = self.pitch_pid.compute(0, self.estimated_state.orientation[1], self.dt)
        yaw = self.yaw_pid.compute(0, self.estimated_state.orientation[2], self.dt)
        
        # Reduce throttle near ground for soft touchdown
        if current_altitude < 2.0:
            throttle *= (0.3 + 0.7 * (current_altitude / 2.0))
        
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
        """Telemetry logging with position tracking"""
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

        # Log position and waypoint info
        if self.waypoints and self.current_waypoint_index < len(self.waypoints):
            current_wp = self.waypoints[self.current_waypoint_index]
            distance_to_wp = np.linalg.norm(self.state.position - current_wp)
            
            if int(time.time() * 2) % 10 == 0:
                logger.info(
                    f"Pos: N{self.state.position[0]:.1f} E{self.state.position[1]:.1f} "
                    f"Alt{-self.state.position[2]:.1f}m | Mode: {self.flight_mode.value} | "
                    f"WP {self.current_waypoint_index + 1}/{len(self.waypoints)} "
                    f"Dist: {distance_to_wp:.1f}m"
                )
        
        self.telemetry_history.append(telemetry)
    
    def set_flight_mode(self, mode: FlightMode):
        """Change flight mode with proper initialization"""
        if mode != self.flight_mode:
            logger.info(f"Flight mode changing: {self.flight_mode.value} → {mode.value}")

            # Reset all controllers
            self.altitude_pid.reset()
            self.roll_pid.reset()
            self.pitch_pid.reset()
            self.yaw_pid.reset()

            # Mode-specific initialization
            current_altitude = -self.estimated_state.position[2]  # Up-positive
            
            if mode == FlightMode.ALTITUDE_HOLD:
                self.setpoints['altitude'] = current_altitude
                logger.info(f"Altitude hold at {current_altitude:.1f}m")
                
            elif mode == FlightMode.POSITION_HOLD:
                self.setpoints['position'] = self.estimated_state.position.copy()
                self.setpoints['altitude'] = current_altitude
                logger.info(f"Position hold at N{self.setpoints['position'][0]:.1f} "
                          f"E{self.setpoints['position'][1]:.1f} Alt{current_altitude:.1f}m")
                
            elif mode == FlightMode.LAND:
                logger.info(f"Starting landing from {current_altitude:.1f}m")
                
            elif mode == FlightMode.AUTO:
                logger.info(f"Starting auto mission with {len(self.waypoints)} waypoints")
                self.current_waypoint_index = 0
                self.mission_complete = False

            self.flight_mode = mode
    
    def set_waypoints(self, waypoints: List[np.ndarray]):
        """Set waypoints (in NED coordinates)"""
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        self.mission_complete = False
        if waypoints:
            logger.info(f"Mission set: {len(waypoints)} waypoints")
            for i, wp in enumerate(waypoints):
                logger.info(f"  WP{i+1}: N{wp[0]:.1f} E{wp[1]:.1f} Alt{-wp[2]:.1f}m")
    
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