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

        # SAFE LIMITS
        self.waypoint_radius = 2.5

        # --- Move and increase max XY velocity first so climb can reference it ---
        self.max_xy_velocity = 5.0  # Increased horizontal capability
        self.max_tilt_angle = 0.785  # 45 degrees

        # Make vertical capability comparable to horizontal capability
        self.max_climb_rate = self.max_xy_velocity     # m/s (now matches horizontal speed)
        self.max_descent_rate = 3.0                    # m/s (allow faster descent if needed)

        # Altitude PID - allow larger throttle adjustments for faster climbs
        self.altitude_pid = PIDController(2.5, 0.3, 1.2, (-0.8, 0.8))
        
        # Position control - INCREASED gains
        self.pos_x_pid = PIDController(0.5, 0.005, 0.8, (-5.0, 5.0))  # Much higher gains
        self.pos_y_pid = PIDController(0.5, 0.005, 0.8, (-5.0, 5.0))
        
        # Attitude control - INCREASED gains
        self.roll_pid = PIDController(4.0, 0.1, 0.3, (-0.5, 0.5), is_angle=True)
        self.pitch_pid = PIDController(4.0, 0.1, 0.3, (-0.5, 0.5), is_angle=True)
        self.yaw_pid = PIDController(2.0, 0.08, 0.25, (-0.12, 0.12), is_angle=True)
        
        # Velocity control - INCREASED gains
        self.vel_x_pid = PIDController(0.6, 0.01, 0.15, (-0.785, 0.785))  # Output in radians (45 deg)
        self.vel_y_pid = PIDController(0.6, 0.01, 0.15, (-0.785, 0.785))
        
        self.vel_x_pid.integral_limit = 0.1
        self.vel_y_pid.integral_limit = 0.1
        logger.info("✓ Controller initialized with hard safety clamps")

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

        # EMERGENCY LEVEL MODE: If too low or falling
        if (current_altitude_m < 1.2) or (current_vertical_velocity_ned > 0.3):
            desired_pitch = 0.0
            desired_roll = 0.0
        else:
            # INCREASED position control gains and limits
            max_pos_error = 15.0  # Allow tracking further targets
            max_desired_vel = 5.0  # Increase max desired velocity

            limited_target_x = np.clip(
                self.setpoints['position'][0],
                drone.estimated_state.position[0] - max_pos_error,
                drone.estimated_state.position[0] + max_pos_error)
            limited_target_y = np.clip(
                self.setpoints['position'][1],
                drone.estimated_state.position[1] - max_pos_error,
                drone.estimated_state.position[1] + max_pos_error)

            # Increased position gains for faster response
            pos_x_output = self.pos_x_pid.compute(limited_target_x, drone.estimated_state.position[0], dt)
            pos_y_output = self.pos_y_pid.compute(limited_target_y, drone.estimated_state.position[1], dt)

            desired_vel_x = np.clip(pos_x_output, -max_desired_vel, max_desired_vel)
            desired_vel_y = np.clip(pos_y_output, -max_desired_vel, max_desired_vel)

            current_vel = drone.estimated_state.velocity

            # INCREASED velocity control limits for aggressive movement
            pitch_command = self.vel_x_pid.compute(desired_vel_x, current_vel[0], dt)
            roll_command = self.vel_y_pid.compute(desired_vel_y, current_vel[1], dt)

            # Allow full tilt range (up to 45 degrees = 0.785 rad)
            max_tilt = 0.785  # 45 degrees
            pitch_command = np.clip(pitch_command, -max_tilt, max_tilt)
            roll_command = np.clip(roll_command, -max_tilt, max_tilt)

            # NED convention: negative pitch = forward
            desired_pitch = -pitch_command
            desired_roll = roll_command

        # Attitude control with increased output limits
        current_attitude = drone.estimated_state.orientation
        roll_output  = self.roll_pid.compute(desired_roll, current_attitude[0], dt)
        pitch_output = self.pitch_pid.compute(desired_pitch, current_attitude[1], dt)
        yaw_output   = self.yaw_pid.compute(self.setpoints['yaw'], current_attitude[2], dt)

        # Increased control authority
        roll_output  = np.clip(roll_output, -0.3, 0.3)
        pitch_output = np.clip(pitch_output, -0.3, 0.3)
        yaw_output   = np.clip(yaw_output, -0.25, 0.25)

        # Tilt compensation
        current_tilt = np.sqrt(current_attitude[0]**2 + current_attitude[1]**2)
        compensation_factor = 1.0 / max(np.cos(current_tilt), 0.5)  # Less aggressive compensation
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
        """AUTO mode — synchronized horizontal and vertical climb, correct NED handling."""
        if not self.waypoints or self.mission_complete:
            return self._stabilize_mode(drone, dt)

        # --- Current and target states ---
        current_wp = self.waypoints[self.current_waypoint_index]
        current_altitude_m = -drone.estimated_state.position[2]  # convert NED down → positive altitude
        target_altitude_m = abs(current_wp[2])                   # stored as negative NED, convert back to positive

        # --- Aggressive climb rate (sync with XY) ---
        altitude_error = target_altitude_m - current_altitude_m
        # Push altitude toward target aggressively
        climb_speed = np.clip(abs(altitude_error) * 1.5, 2.0, self.max_climb_rate * 2.0)
        new_altitude = current_altitude_m + np.sign(altitude_error) * climb_speed * dt

        # Don’t overshoot
        if (altitude_error > 0 and new_altitude > target_altitude_m) or \
           (altitude_error < 0 and new_altitude < target_altitude_m):
            new_altitude = target_altitude_m

        # ✅ Set positive altitude (up)
        self.setpoints['altitude'] = new_altitude

        # ✅ Set XY waypoint (Z handled separately)
        self.setpoints['position'] = np.array([current_wp[0], current_wp[1], 0.0])

        # --- Run stabilize control loop ---
        control = self._stabilize_mode(drone, dt)

        # --- Waypoint reached check ---
        current_pos = drone.estimated_state.position
        horizontal_distance = np.linalg.norm(current_pos[:2] - current_wp[:2])
        vertical_distance = abs(current_altitude_m - target_altitude_m)

        waypoint_reached = (horizontal_distance < self.waypoint_radius and vertical_distance < 1.0)
        if waypoint_reached:
            if self.current_waypoint_index < len(self.waypoints) - 1:
                self.current_waypoint_index += 1
                next_wp = self.waypoints[self.current_waypoint_index]
                logger.info(f"✓ Waypoint {self.current_waypoint_index} reached! "
                            f"Next: N{next_wp[0]:.1f} E{next_wp[1]:.1f} Alt{abs(next_wp[2]):.1f}m")
                self.pos_x_pid.reset()
                self.pos_y_pid.reset()
            else:
                self.mission_complete = True
                logger.info("✓ All waypoints reached! Holding position.")

        return control
    def _rtl_mode(self, drone: Drone, dt: float) -> np.ndarray:
        """Return to launch mode"""
        current_pos = drone.estimated_state.position
        current_altitude = -current_pos[2]  # Convert to positive altitude
        safe_altitude = max(5.0, current_altitude + 2.0)
        distance_to_home_xy = np.linalg.norm(current_pos[:2] - self.launch_position[:2])
        
        if distance_to_home_xy > 2.0 or current_altitude < safe_altitude - 1.0:
            # Phase 1: Climb to safe altitude and move to home
            target_position = np.array([0.0, 0.0, -safe_altitude])  # NED coordinates
            self.setpoints['position'] = target_position
            self.setpoints['altitude'] = safe_altitude
            return self._stabilize_mode(drone, dt)
        else:
            # Phase 2: At home position, descend to land
            self.setpoints['position'] = np.array([0.0, 0.0, 0.0])
            self.setpoints['altitude'] = 0.0
            
            if current_altitude < 0.3:
                logger.info("✓ RTL complete - Landed at launch position")
                self.flight_mode = FlightMode.MANUAL
                self.is_launched = False
                return np.zeros(4)
            
            return self._controlled_descent(drone, dt)

    def _land_mode(self, drone: Drone, dt: float) -> np.ndarray:
        """Land at current position"""
        current_pos = drone.estimated_state.position
        current_altitude = -current_pos[2]  # Convert to positive altitude
        
        if current_altitude < 0.3:
            logger.info("✓ Landed at current position")
            self.flight_mode = FlightMode.MANUAL
            return np.zeros(4)
        
        # Set landing target to current XY, altitude 0
        land_target = current_pos.copy()
        land_target[2] = 0.0  # NED: ground level
        self.setpoints['position'] = land_target
        self.setpoints['altitude'] = 0.0
        
        return self._controlled_descent(drone, dt)

    def _controlled_descent(self, drone: Drone, dt: float) -> np.ndarray:
        """Controlled descent for landing"""
        current_altitude = -drone.estimated_state.position[2]  # Convert to positive altitude
        
        if current_altitude < 0.3:
            return np.zeros(4)
        
        # Progressive descent rate based on altitude
        if current_altitude > 5.0:
            descent_rate = 0.8  # m/s
        elif current_altitude > 2.0:
            descent_rate = 0.4  # m/s
        else:
            descent_rate = 0.15  # m/s - gentle final approach
        
        target_altitude = max(0.0, current_altitude - descent_rate * dt)
        
        # Velocity-based descent control
        current_vertical_velocity = -drone.estimated_state.velocity[2]  # Convert to positive up
        velocity_error = -descent_rate - current_vertical_velocity
        
        throttle_adjustment = self.altitude_pid.compute(0, -velocity_error, dt)
        hover_throttle = drone.get_hover_throttle()
        throttle = hover_throttle + throttle_adjustment
        
        # Reduce throttle progressively as we descend
        if current_altitude < 2.0:
            throttle *= 0.9
        if current_altitude < 1.0:
            throttle *= 0.8
        
        throttle = np.clip(throttle, 0.2, 0.7)
        
        # Keep level attitude during descent
        roll = self.roll_pid.compute(0.0, drone.estimated_state.orientation[0], dt)
        pitch = self.pitch_pid.compute(0.0, drone.estimated_state.orientation[1], dt)
        yaw = self.yaw_pid.compute(0.0, drone.estimated_state.orientation[2], dt)
        
        return np.array([throttle, roll, pitch, yaw])

    def launch(self, target_altitude: float = 2.0):
        """Launch the drone to target altitude"""
        if self.is_launched:
            logger.warning("Already launched!")
            return

        # Store launch position
        self.launch_position = self.setpoints['position'].copy()
        
        # Set target - keep current XY, only change altitude
        launch_target = self.launch_position.copy()
        launch_target[2] = -target_altitude  # NED: negative Z for altitude

        self.setpoints['altitude'] = target_altitude
        self.setpoints['position'] = launch_target
        self.setpoints['yaw'] = 0.0

        # Reset all PIDs for clean start
        self.altitude_pid.reset()
        self.pos_x_pid.reset()
        self.pos_y_pid.reset()
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.yaw_pid.reset()
        
        # Switch to stabilize mode
        self.set_flight_mode(FlightMode.STABILIZE)
        self.is_launched = True
        
        logger.info(f"🚀 Launching to {target_altitude}m altitude!")

    def add_waypoint(self, north: float, east: float, altitude: float):
        """Add a waypoint to the mission"""
        # Store waypoints in NED coordinates (positive altitude becomes negative Z)
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
            # Reset PIDs on mode change for smooth transition
            self.altitude_pid.reset()
            self.roll_pid.reset()
            self.pitch_pid.reset()
            self.yaw_pid.reset()
            self.pos_x_pid.reset()
            self.pos_y_pid.reset()
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
            }
        }