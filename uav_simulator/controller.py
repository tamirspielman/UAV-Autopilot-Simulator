#controller.py
import numpy as np
from typing import List, Dict, Any, Tuple
from .utils import FlightMode, logger
from .drone import Drone

class PIDController:
    """PID controller with anti-windup and derivative filtering."""
    def __init__(self, kp: float, ki: float, kd: float,
                 output_limits: Tuple[float, float] = (-0.5, 0.5),
                 is_angle: bool = False):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min, self.output_max = output_limits
        self.is_angle = is_angle

        self.integral_limit = 1.0
        self.derivative_filter_alpha = 0.15
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0

    def compute(self, setpoint: float, measurement: float, dt: float) -> float:
        if dt <= 0:
            return 0.0

        error = setpoint - measurement
        if self.is_angle and abs(error) > np.pi:
            error -= 2 * np.pi * np.sign(error)

        p_term = self.kp * error
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral

        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        derivative = (self.derivative_filter_alpha * derivative +
                      (1 - self.derivative_filter_alpha) * self.prev_derivative)
        d_term = self.kd * derivative

        output = p_term + i_term + d_term
        output_limited = np.clip(output, self.output_min, self.output_max)

        if output != output_limited:
            self.integral -= error * dt

        self.prev_error = error
        self.prev_derivative = derivative

        return output_limited

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0

class Controller:
    def __init__(self):
        self.flight_mode = FlightMode.MANUAL
        self.setpoints = {
            'altitude': 0.0,
            'position': np.array([0.0, 0.0, 0.0]),  # NED coordinates
            'yaw': 0.0
        }
        self.waypoints: List[np.ndarray] = []
        self.current_waypoint_index = 0
        self.mission_complete = False
        self.launch_position = np.array([0.0, 0.0, 0.0])
        self.control_output = np.zeros(4)
        self.is_launched = False

        # SAFE LIMITS - tighter waypoint accuracy
        self.waypoint_radius = 0.5  # 0.5m horizontal accuracy
        self.waypoint_altitude_tolerance = 0.5  # 0.5m vertical accuracy

        self.max_xy_velocity = 5.0
        self.max_tilt_angle = 0.785  # 45 degrees

        self.max_climb_rate = self.max_xy_velocity
        self.max_descent_rate = 3.0

        # Altitude PID
        self.altitude_pid = PIDController(2.5, 0.3, 1.2, (-0.8, 0.8))
        
        # Position control - Balanced gains for accuracy
        self.pos_x_pid = PIDController(0.4, 0.002, 0.6, (-5.0, 5.0))
        self.pos_y_pid = PIDController(0.4, 0.002, 0.6, (-5.0, 5.0))
        
        # Attitude control
        self.roll_pid = PIDController(4.0, 0.1, 0.3, (-0.5, 0.5), is_angle=True)
        self.pitch_pid = PIDController(4.0, 0.1, 0.3, (-0.5, 0.5), is_angle=True)
        self.yaw_pid = PIDController(2.0, 0.08, 0.25, (-0.12, 0.12), is_angle=True)
        
        # Velocity control - Reduced for precision
        self.vel_x_pid = PIDController(0.4, 0.005, 0.12, (-0.6, 0.6))
        self.vel_y_pid = PIDController(0.4, 0.005, 0.12, (-0.6, 0.6))
        
        self.vel_x_pid.integral_limit = 0.1
        self.vel_y_pid.integral_limit = 0.1
        
        # RTL and LAND state tracking
        self._rtl_started = False
        self._land_started = False
        
        logger.info("✓ Controller initialized with precision tuning")

    def compute_control(self, drone: Drone, dt: float) -> np.ndarray:
        if self.flight_mode == FlightMode.MANUAL:
            return np.zeros(4)
        elif self.flight_mode == FlightMode.STABILIZE:
            return self._stabilize_mode(drone, dt)
        elif self.flight_mode == FlightMode.AUTO:
            return self._auto_mode(drone, dt)
        elif self.flight_mode == FlightMode.RTL:
            return self._rtl_mode(drone, dt)
        elif self.flight_mode == FlightMode.LAND:
            return self._land_mode(drone, dt)
        else:
            return np.zeros(4)

    def _stabilize_mode(self, drone: Drone, dt: float) -> np.ndarray:
        current_altitude_ned = drone.estimated_state.position[2]
        target_altitude_ned = -self.setpoints['altitude']
        altitude_error = target_altitude_ned - current_altitude_ned
        current_vertical_velocity_ned = drone.estimated_state.velocity[2]
        current_altitude_m = -current_altitude_ned

        # Altitude control
        altitude_velocity_gain = 1.2
        desired_vertical_velocity = np.clip(
            altitude_error * altitude_velocity_gain,
            -self.max_climb_rate,
            self.max_descent_rate
        )
        velocity_error = desired_vertical_velocity - current_vertical_velocity_ned
        throttle_adjustment = self.altitude_pid.compute(-velocity_error, 0, dt)
        hover_throttle = drone.get_hover_throttle()
        throttle = hover_throttle + throttle_adjustment

        desired_roll = 0.0
        desired_pitch = 0.0

        # EMERGENCY LEVEL MODE
        if (current_altitude_m < 1.2) or (current_vertical_velocity_ned > 0.3):
            desired_pitch = 0.0
            desired_roll = 0.0
        else:
            # Position control with reduced limits for precision
            max_pos_error = 15.0
            max_desired_vel = 3.0  # Reduced from 5.0 for better accuracy

            limited_target_x = np.clip(
                self.setpoints['position'][0],
                drone.estimated_state.position[0] - max_pos_error,
                drone.estimated_state.position[0] + max_pos_error)
            limited_target_y = np.clip(
                self.setpoints['position'][1],
                drone.estimated_state.position[1] - max_pos_error,
                drone.estimated_state.position[1] + max_pos_error)

            pos_x_output = self.pos_x_pid.compute(limited_target_x, drone.estimated_state.position[0], dt)
            pos_y_output = self.pos_y_pid.compute(limited_target_y, drone.estimated_state.position[1], dt)

            desired_vel_x = np.clip(pos_x_output, -max_desired_vel, max_desired_vel)
            desired_vel_y = np.clip(pos_y_output, -max_desired_vel, max_desired_vel)

            current_vel = drone.estimated_state.velocity

            # Velocity control with reduced tilt for precision
            pitch_command = self.vel_x_pid.compute(desired_vel_x, current_vel[0], dt)
            roll_command = self.vel_y_pid.compute(desired_vel_y, current_vel[1], dt)

            # Reduced max tilt for better precision (30 degrees instead of 45)
            max_tilt = 0.52  # 30 degrees for precision
            pitch_command = np.clip(pitch_command, -max_tilt, max_tilt)
            roll_command = np.clip(roll_command, -max_tilt, max_tilt)

            # NED convention: negative pitch = forward
            desired_pitch = -pitch_command
            desired_roll = roll_command

        # Attitude control
        current_attitude = drone.estimated_state.orientation
        roll_output  = self.roll_pid.compute(desired_roll, current_attitude[0], dt)
        pitch_output = self.pitch_pid.compute(desired_pitch, current_attitude[1], dt)
        yaw_output   = self.yaw_pid.compute(self.setpoints['yaw'], current_attitude[2], dt)

        roll_output  = np.clip(roll_output, -0.3, 0.3)
        pitch_output = np.clip(pitch_output, -0.3, 0.3)
        yaw_output   = np.clip(yaw_output, -0.25, 0.25)

        # Tilt compensation
        current_tilt = np.sqrt(current_attitude[0]**2 + current_attitude[1]**2)
        compensation_factor = 1.0 / max(np.cos(current_tilt), 0.5)
        throttle *= compensation_factor

        # Fall protection
        if current_vertical_velocity_ned > 1.0:
            throttle += 0.2
            roll_output *= 0.1
            pitch_output *= 0.1

        throttle = np.clip(throttle, 0.35, 1.0)

        self.control_output = np.array([throttle, roll_output, pitch_output, yaw_output])
        return self.control_output

    def _auto_mode(self, drone: Drone, dt: float) -> np.ndarray:
        """AUTO mode - synchronized horizontal and vertical climb"""
        if not self.waypoints or self.mission_complete:
            return self._stabilize_mode(drone, dt)

        # Current and target states
        current_wp = self.waypoints[self.current_waypoint_index]
        current_altitude_m = -drone.estimated_state.position[2]
        target_altitude_m = abs(current_wp[2])

        # Set target altitude directly
        self.setpoints['altitude'] = target_altitude_m

        # Set XY waypoint
        self.setpoints['position'] = np.array([current_wp[0], current_wp[1], 0.0])

        # Run stabilize control loop
        control = self._stabilize_mode(drone, dt)

        # Waypoint reached check - BOTH horizontal AND vertical
        current_pos = drone.estimated_state.position
        horizontal_distance = np.linalg.norm(current_pos[:2] - current_wp[:2])
        vertical_distance = abs(current_altitude_m - target_altitude_m)

        waypoint_reached = (
            horizontal_distance < self.waypoint_radius and 
            vertical_distance < self.waypoint_altitude_tolerance
        )
        
        if waypoint_reached:
            if self.current_waypoint_index < len(self.waypoints) - 1:
                self.current_waypoint_index += 1
                next_wp = self.waypoints[self.current_waypoint_index]
                logger.info(f"✓ Waypoint {self.current_waypoint_index} reached! "
                            f"Next: N{next_wp[0]:.1f} E{next_wp[1]:.1f} Alt{abs(next_wp[2]):.1f}m")
                # CRITICAL: Reset ALL PIDs when switching waypoints
                self.pos_x_pid.reset()
                self.pos_y_pid.reset()
                self.altitude_pid.reset()
                self.vel_x_pid.reset()
                self.vel_y_pid.reset()
            else:
                self.mission_complete = True
                logger.info("✓ All waypoints reached! Holding position.")

        return control

    def _rtl_mode(self, drone: Drone, dt: float) -> np.ndarray:
        """Return to launch mode - SIMPLE DIRECT APPROACH"""
        current_pos = drone.estimated_state.position
        current_altitude = -current_pos[2]
        distance_to_home_xy = np.linalg.norm(current_pos[:2] - self.launch_position[:2])
        
        # If not at home position yet, go there first
        if distance_to_home_xy > self.waypoint_radius:
            if not self._rtl_started:
                logger.info("RTL: Moving to home position")
                self._rtl_started = True
            
            # Set target to home position at current altitude
            self.setpoints['position'] = np.array([self.launch_position[0], self.launch_position[1], -current_altitude])
            self.setpoints['altitude'] = current_altitude
            return self._stabilize_mode(drone, dt)
        
        # At home position, now descend to land
        if current_altitude > 0.5:
            # Force descent by setting target altitude to 0
            self.setpoints['position'] = np.array([self.launch_position[0], self.launch_position[1], 0.0])
            self.setpoints['altitude'] = 0.0
            
            # Get normal stabilize control
            control = self._stabilize_mode(drone, dt)
            
            # FORCE DESCENT by reducing throttle
            # Calculate how much to reduce throttle based on altitude
            throttle_reduction = 0.0
            if current_altitude > 3.0:
                throttle_reduction = 0.2
            elif current_altitude > 1.5:
                throttle_reduction = 0.3
            else:
                throttle_reduction = 0.4
                
            control[0] = control[0] * (1.0 - throttle_reduction)
            control[0] = max(control[0], 0.3)  # Minimum throttle to maintain control
            
            logger.debug(f"RTL: Descending from {current_altitude:.1f}m, throttle: {control[0]:.2f}")
            return control
        else:
            # Landed
            logger.info("✓ RTL complete - Landed at launch position")
            self.flight_mode = FlightMode.MANUAL
            self.is_launched = False
            self._rtl_started = False
            return np.zeros(4)

    def _land_mode(self, drone: Drone, dt: float) -> np.ndarray:
        """Land at current position - SIMPLE DIRECT APPROACH"""
        current_pos = drone.estimated_state.position
        current_altitude = -current_pos[2]
        
        if current_altitude > 0.5:
            if not self._land_started:
                logger.info("LAND: Starting descent at current position")
                self._land_started = True
            
            # Set target to current position but at ground level
            self.setpoints['position'] = np.array([current_pos[0], current_pos[1], 0.0])
            self.setpoints['altitude'] = 0.0
            
            # Get normal stabilize control
            control = self._stabilize_mode(drone, dt)
            
            # FORCE DESCENT by reducing throttle
            throttle_reduction = 0.0
            if current_altitude > 3.0:
                throttle_reduction = 0.2
            elif current_altitude > 1.5:
                throttle_reduction = 0.3
            else:
                throttle_reduction = 0.4
                
            control[0] = control[0] * (1.0 - throttle_reduction)
            control[0] = max(control[0], 0.3)  # Minimum throttle to maintain control
            
            logger.debug(f"LAND: Descending from {current_altitude:.1f}m, throttle: {control[0]:.2f}")
            return control
        else:
            # Landed
            logger.info("✓ Land complete - Landed at current position")
            self.flight_mode = FlightMode.MANUAL
            self.is_launched = False
            self._land_started = False
            return np.zeros(4)

    def set_launch_position(self, north: float, east: float, altitude: float = 0.0):
        """Set custom launch position (for future use)"""
        self.launch_position = np.array([north, east, -altitude])
        logger.info(f"Launch position set to: N{north:.1f} E{east:.1f} Alt{altitude:.1f}m")

    def launch(self, target_altitude: float = 2.0):
        """Launch the drone to target altitude"""
        if self.is_launched:
            logger.warning("Already launched!")
            return

        self.launch_position = self.setpoints['position'].copy()
        launch_target = self.launch_position.copy()
        launch_target[2] = -target_altitude

        self.setpoints['altitude'] = target_altitude
        self.setpoints['position'] = launch_target
        self.setpoints['yaw'] = 0.0

        # Reset all PIDs
        self.altitude_pid.reset()
        self.pos_x_pid.reset()
        self.pos_y_pid.reset()
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.yaw_pid.reset()
        self.vel_x_pid.reset()
        self.vel_y_pid.reset()
        
        # Reset RTL/LAND states
        self._rtl_started = False
        self._land_started = False
        
        self.set_flight_mode(FlightMode.STABILIZE)
        self.is_launched = True
        
        logger.info(f"🚀 Launching to {target_altitude}m altitude!")

    def add_waypoint(self, north: float, east: float, altitude: float):
        """Add a waypoint to the mission"""
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
            logger.warning("Not launched yet! Auto-launching to 2m...")
            self.launch(2.0)
        self.current_waypoint_index = 0
        self.mission_complete = False
        self.set_flight_mode(FlightMode.AUTO)
        logger.info(f"🎯 Mission started with {len(self.waypoints)} waypoints")

    def emergency_land(self):
        """Emergency landing"""
        logger.warning("⚠️ EMERGENCY LANDING!")
        self.mission_complete = True
        self.flight_mode = FlightMode.LAND

    def set_flight_mode(self, mode: FlightMode):
        """Change flight mode"""
        if mode != self.flight_mode:
            logger.info(f"Mode change: {self.flight_mode.value} → {mode.value}")
            
            # Reset all PIDs on mode change
            self.altitude_pid.reset()
            self.roll_pid.reset()
            self.pitch_pid.reset()
            self.yaw_pid.reset()
            self.pos_x_pid.reset()
            self.pos_y_pid.reset()
            self.vel_x_pid.reset()
            self.vel_y_pid.reset()
            
            self.flight_mode = mode

    def get_status(self) -> Dict[str, Any]:
        """Get controller status"""
        return {
            'flight_mode': self.flight_mode.value,
            'waypoint_index': self.current_waypoint_index,
            'mission_complete': self.mission_complete,
            'is_launched': self.is_launched,
            'control_output': self.control_output.tolist(),
            'setpoints': {
                'altitude': self.setpoints['altitude'],
                'position': self.setpoints['position'].tolist(),
                'yaw': self.setpoints['yaw']
            },
            'launch_position': self.launch_position.tolist()
        }