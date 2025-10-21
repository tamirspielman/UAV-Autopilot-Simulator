# controller.py
"""
Controller - handles flight modes and control algorithms
"""
import time
import numpy as np
from typing import List, Optional, Dict, Tuple, Any
from collections import deque

from .utils import FlightMode, logger
from .drone import Drone

class PIDController:
    """PID controller with anti-windup and filtering"""
    
    def __init__(self, kp: float, ki: float, kd: float, 
                 output_limits: Tuple[float, float] = (-0.5, 0.5)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min, self.output_max = output_limits
        
        # Integral limits
        self.integral_limit = 1.0
        
        # Derivative filtering
        self.derivative_filter_alpha = 0.1
        
        # Initialize state variables
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.is_launched = False
        self.hover_throttle = 0.75  
        self.waypoint_radius = 2.0
    def compute(self, setpoint: float, measurement: float, dt: float) -> float:
        if dt <= 0:
            return 0.0
            
        error = setpoint - measurement
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with clamping
        self.integral += error * dt
        # Simple clamping
        if self.integral > self.integral_limit:
            self.integral = self.integral_limit
        elif self.integral < -self.integral_limit:
            self.integral = -self.integral_limit
        i_term = self.ki * self.integral
        
        # Derivative term with filtering
        if dt > 0:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0
            
        # Apply derivative filtering
        derivative = (self.derivative_filter_alpha * derivative + 
                     (1 - self.derivative_filter_alpha) * self.prev_derivative)
        
        d_term = self.kd * derivative
        
        # Compute output
        output = p_term + i_term + d_term
        
        # Apply output limits
        if output > self.output_max:
            output_limited = self.output_max
        elif output < self.output_min:
            output_limited = self.output_min
        else:
            output_limited = output
        
        # Anti-windup: stop integrating when saturated
        if output_limited == self.output_min or output_limited == self.output_max:
            self.integral -= error * dt  # Undo the integration
        
        # Update state
        self.prev_error = error
        self.prev_derivative = derivative
        
        return output_limited
        
    def reset(self):
        """Reset controller state"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0

class Controller:
    """
    Flight controller with multiple flight modes
    """
    
    def __init__(self):
        # Flight mode
        self.flight_mode = FlightMode.MANUAL
        
        # Setpoints
        self.setpoints = {
            'altitude': 0.0,
            'position': np.array([0.0, 0.0, 0.0]),
            'yaw': 0.0
        }
        
        # Mission waypoints
        self.waypoints: List[np.ndarray] = []
        self.current_waypoint_index = 0
        self.mission_complete = False
        self.launch_position = np.array([0.0, 0.0, 0.0])
        
        # Control outputs
        self.control_output: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0])
        
        # IMPROVED PID TUNING - more aggressive for better response
        self.altitude_pid = PIDController(4.0, 0.8, 2.5, (-0.6, 0.6))  # More aggressive
        self.pos_x_pid = PIDController(2.5, 0.3, 1.5, (-0.8, 0.8))     # More responsive position control
        self.pos_y_pid = PIDController(2.5, 0.3, 1.5, (-0.8, 0.8))     # More responsive position control
        self.roll_pid = PIDController(4.0, 0.3, 1.0, (-0.8, 0.8))      
        self.pitch_pid = PIDController(4.0, 0.3, 1.0, (-0.8, 0.8))     
        self.yaw_pid = PIDController(2.0, 0.2, 0.5, (-0.4, 0.4))   
        
        # System state
        self.is_launched = False
        self.hover_throttle = 0.75  # Increased hover throttle for new physics
        self.waypoint_radius = 2.0
        
        logger.info("✓ Flight Controller initialized")
    
    def compute_control(self, drone: Drone, dt: float) -> np.ndarray:
        """
        Compute control outputs based on current flight mode
        """
        if self.flight_mode == FlightMode.MANUAL:
            return np.array([0.0, 0.0, 0.0, 0.0])
        
        elif self.flight_mode == FlightMode.STABILIZE:
            return self._stabilize_mode(drone, dt)
        
        elif self.flight_mode == FlightMode.AUTO:
            return self._auto_mode(drone, dt)
        
        elif self.flight_mode == FlightMode.RTL:
            return self._rtl_mode(drone, dt)
        
        elif self.flight_mode == FlightMode.LAND:
            return self._land_mode(drone, dt)
        
        else:
            return np.array([0.0, 0.0, 0.0, 0.0])
    def _stabilize_mode(self, drone: Drone, dt: float) -> np.ndarray:
        """Stabilize mode - hold altitude and position"""
        current_altitude = -drone.estimated_state.position[2]
        target_altitude = self.setpoints['altitude']

        # Altitude control - MORE AGGRESSIVE FOR TAKEOFF
        if target_altitude > 0.1 or current_altitude > 0.1:
            throttle_adjustment = self.altitude_pid.compute(
                setpoint=target_altitude,
                measurement=current_altitude,
                dt=dt
            )
            throttle = self.hover_throttle + throttle_adjustment
            # Adjusted clipping for takeoff - allow more throttle
            if throttle > 0.95: throttle = 0.95  # Increased max
            if throttle < 0.6: throttle = 0.6    # Increased min for takeoff
        else:
            # On ground but ready for takeoff - apply minimal throttle
            throttle = 0.0

        # POSITION CONTROL - FIXED: Always control position, even during takeoff
        target_pos = self.setpoints['position']
        current_pos = drone.estimated_state.position
        pos_error = target_pos - current_pos

        # Convert position error to desired tilt angles
        max_tilt_angle = 0.3  # radians (~17 degrees) - reduced for stability

        # Only apply position control when we have significant altitude or are close to target
        if current_altitude > 0.2 or abs(pos_error[0]) > 0.5 or abs(pos_error[1]) > 0.5:
            desired_pitch = -self.pos_x_pid.compute(0, pos_error[0], dt)
            desired_roll = self.pos_y_pid.compute(0, pos_error[1], dt)

            # Clip desired angles
            if desired_pitch > max_tilt_angle: desired_pitch = max_tilt_angle
            if desired_pitch < -max_tilt_angle: desired_pitch = -max_tilt_angle
            if desired_roll > max_tilt_angle: desired_roll = max_tilt_angle
            if desired_roll < -max_tilt_angle: desired_roll = -max_tilt_angle
        else:
            # During initial takeoff or very close to target - keep level
            desired_roll = 0.0
            desired_pitch = 0.0
    
        # Attitude control
        roll = self.roll_pid.compute(desired_roll, drone.estimated_state.orientation[0], dt)
        pitch = self.pitch_pid.compute(desired_pitch, drone.estimated_state.orientation[1], dt)
        yaw = self.yaw_pid.compute(self.setpoints['yaw'], drone.estimated_state.orientation[2], dt)
    
        return np.array([throttle, roll, pitch, yaw])
    
    def _auto_mode(self, drone: Drone, dt: float) -> np.ndarray:
        """Auto mode - follow waypoints"""
        if not self.waypoints or self.mission_complete:
            return self._stabilize_mode(drone, dt)

        # Get current waypoint
        current_wp = self.waypoints[self.current_waypoint_index]

        # Update setpoint to current waypoint - FIXED: Set both position and altitude
        self.setpoints['position'] = current_wp
        self.setpoints['altitude'] = -current_wp[2]  # Convert NED to altitude
    
        # Use stabilize control to reach waypoint
        control = self._stabilize_mode(drone, dt)
    
        # Check if waypoint reached
        current_pos = drone.estimated_state.position
        distance = np.linalg.norm(current_pos - current_wp)
    
        # More lenient waypoint acceptance for altitude
        horizontal_distance = np.linalg.norm(current_pos[:2] - current_wp[:2])
        vertical_distance = abs(current_pos[2] - current_wp[2])
    
        if horizontal_distance < self.waypoint_radius and vertical_distance < 1.0:
            if self.current_waypoint_index < len(self.waypoints) - 1:
                self.current_waypoint_index += 1
                next_wp = self.waypoints[self.current_waypoint_index]
                logger.info(f"✓ Waypoint {self.current_waypoint_index} reached! Next: N{next_wp[0]:.1f} E{next_wp[1]:.1f} Alt{-next_wp[2]:.1f}m")
            else:
                self.mission_complete = True
                logger.info("✓ All waypoints reached! Holding position.")
    
        return control
    
    def _rtl_mode(self, drone: Drone, dt: float) -> np.ndarray:
        """Return to launch mode"""
        current_pos = drone.estimated_state.position
        current_altitude = -current_pos[2]
        
        safe_altitude = 5.0  # meters
        ground_position = np.array([0.0, 0.0, 0.0])
        
        # Calculate horizontal distance to home
        distance_to_home_xy = np.linalg.norm(current_pos[:2])
        
        if distance_to_home_xy > 3.0 or current_altitude < safe_altitude - 1.0:
            # Phase 1: Go to home XY position at safe altitude
            target_position = np.array([0.0, 0.0, -safe_altitude])
            self.setpoints['position'] = target_position
            self.setpoints['altitude'] = safe_altitude
            return self._stabilize_mode(drone, dt)
        else:
            # Phase 2: At home position, now descend to land
            self.setpoints['position'] = ground_position
            self.setpoints['altitude'] = 0.0
            
            if current_altitude < 0.3:
                logger.info("✓ RTL complete - Landed at launch position")
                self.flight_mode = FlightMode.MANUAL
                self.is_launched = False
                return np.array([0.0, 0.0, 0.0, 0.0])
            
            return self._controlled_descent(drone, dt)
    
    def _land_mode(self, drone: Drone, dt: float) -> np.ndarray:
        """Land at current position"""
        current_pos = drone.estimated_state.position
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
        
        return self._controlled_descent(drone, dt)
    
    def _controlled_descent(self, drone: Drone, dt: float) -> np.ndarray:
        """Controlled descent for landing"""
        current_altitude = -drone.estimated_state.position[2]
        
        if current_altitude < 0.3:
            return np.array([0.0, 0.0, 0.0, 0.0])
        
        # Progressive descent rate based on altitude
        if current_altitude > 5.0:
            descent_rate = 1.0  # m/s
        elif current_altitude > 2.0:
            descent_rate = 0.5  # m/s
        else:
            descent_rate = 0.2  # m/s
        
        target_altitude = max(0.0, current_altitude - descent_rate * dt)
        
        throttle_adjustment = self.altitude_pid.compute(
            target_altitude,
            current_altitude,
            dt
        )
        
        throttle = self.hover_throttle + throttle_adjustment
        
        # Reduce throttle gradually as we approach ground
        if current_altitude < 2.0:
            throttle *= 0.8
        if current_altitude < 1.0:
            throttle *= 0.6
        
        # Clip throttle
        if throttle > 0.6: throttle = 0.6
        if throttle < 0.2: throttle = 0.2
        
        # Keep level attitude during descent
        roll = self.roll_pid.compute(0.0, drone.estimated_state.orientation[0], dt)
        pitch = self.pitch_pid.compute(0.0, drone.estimated_state.orientation[1], dt)
        yaw = self.yaw_pid.compute(0.0, drone.estimated_state.orientation[2], dt)
        
        return np.array([throttle, roll, pitch, yaw])
    
    def launch(self, target_altitude: float = 2.0):
        """Launch the drone"""
        if self.is_launched:
            logger.warning("Already launched!")
            return

        # Set target altitude and position - FIXED: Keep current XY position, only change altitude
        current_pos = self.setpoints['position'].copy()
        current_pos[2] = -target_altitude  # Only change Z coordinate (NED: negative for altitude)

        self.setpoints['altitude'] = target_altitude
        self.setpoints['position'] = current_pos  # Keep the same XY position
        self.setpoints['yaw'] = 0.0

        # Reset PIDs
        self.altitude_pid.reset()
        self.pos_x_pid.reset()
        self.pos_y_pid.reset()
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.yaw_pid.reset()
    
        # Increase hover throttle for takeoff
        self.hover_throttle = 0.85  # Increased for better takeoff
    
        # Switch to stabilize mode
        self.set_flight_mode(FlightMode.STABILIZE)
        self.is_launched = True
    
        logger.info(f"🚀 Launching to {target_altitude}m altitude at current position!")
    
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
            logger.warning("Not launched yet! Auto-launching to 5m...")
            self.launch(5.0)
            # Don't sleep here - let the simulation handle timing
        
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