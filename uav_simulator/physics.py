"""
Physics engine - handles gravity, forces, and dynamics
"""
import numpy as np
from typing import Dict

from .drone import Drone, DroneState
from .utils import rotation_matrix, normalize_angles, logger

class Physics:
    """
    Physics engine for drone simulation
    Handles gravity, forces, and dynamics calculations
    """
    
    def __init__(self):
        # Physical constants
        self.gravity = 9.80665  # m/s^2
        
        # Moment of inertia (kg*m^2) - adjusted for better stability
        self.inertia = np.diag([0.02, 0.02, 0.03])
        self.inertia_inv = np.linalg.inv(self.inertia)
        # Environment properties
        self.ground_level = 0.0  # Ground at Z=0 in NED
        
        logger.info("✓ Physics engine initialized")
    
    def update(self, drone: Drone, control_input: np.ndarray, dt: float) -> DroneState:
        """
        Update drone physics for one time step using RK4 integration
        """
        # Convert control inputs to motor speeds
        motor_speeds = self._control_to_motors(control_input, drone)
        drone.set_motor_speeds(motor_speeds)
        
        try:
            # RK4 integration for smooth, accurate dynamics
            k1 = self._state_derivative(drone.true_state, drone)
            k2 = self._state_derivative(self._add_state(drone.true_state, k1, dt/2), drone)
            k3 = self._state_derivative(self._add_state(drone.true_state, k2, dt/2), drone)
            k4 = self._state_derivative(self._add_state(drone.true_state, k3, dt), drone)
            
            # Combine derivatives: (k1 + 2*k2 + 2*k3 + k4) / 6
            derivative = {}
            for key in k1.keys():
                derivative[key] = (k1[key] + 2*k2[key] + 2*k3[key] + k4[key]) / 6
            
        except Exception as e:
            logger.error(f"Physics integration error: {e}")
            return drone.true_state
        
        
        # Create new state
        new_state = DroneState()
        current_thrust = drone.calculate_thrust()
        weight_force = drone.mass * self.gravity

        # Update position
        new_state.position = drone.true_state.position + derivative['position'] * dt
        
        # Update velocity
        new_state.velocity = drone.true_state.velocity + derivative['velocity'] * dt
        
        # Update orientation
        new_state.orientation = drone.true_state.orientation + derivative['orientation'] * dt
        new_state.orientation = normalize_angles(new_state.orientation)
        
        # Update angular velocity
        new_state.angular_velocity = drone.true_state.angular_velocity + derivative['angular_velocity'] * dt
        
        # Update acceleration
        new_state.acceleration = derivative['velocity']
        
        # Update timestamp
        new_state.timestamp = drone.true_state.timestamp + dt
        
        # Ground protection - prevent going below ground
        if new_state.position[2] >= self.ground_level:  # At or below ground
            if current_thrust < weight_force * 0.8:  # Not enough thrust to lift off
                new_state.position[2] = self.ground_level
                # Stop downward velocity and reduce upward velocity if on ground
                if new_state.velocity[2] > 0:  # Moving downward
                    new_state.velocity[2] = 0.0
                elif new_state.velocity[2] < 0 and current_thrust < weight_force:  # Moving upward but not enough thrust
                    new_state.velocity[2] *= 0.5  # Dampen upward velocity
            # If we have enough thrust, allow the drone to lift off
        return new_state
    
    def _control_to_motors(self, control: np.ndarray, drone: Drone) -> np.ndarray:
        """Convert control inputs to motor speeds"""
        throttle = np.clip(control[0], 0.0, 1.0)
        roll = np.clip(control[1], -0.5, 0.5)
        pitch = np.clip(control[2], -0.5, 0.5)
        yaw = np.clip(control[3], -0.3, 0.3)
        
        # Mix control inputs to motor commands
        m0 = throttle - pitch + roll - yaw  # Front Right
        m1 = throttle - pitch - roll + yaw  # Front Left  
        m2 = throttle + pitch - roll - yaw  # Rear Left
        m3 = throttle + pitch + roll + yaw  # Rear Right
        
        # Normalize to avoid exceeding limits
        motor_commands = np.array([m0, m1, m2, m3])
        motor_commands = np.clip(motor_commands, 0.0, 1.0)
        
        # Convert to RPM - INCREASED RPM RANGE FOR MORE THRUST
        rpm_range = drone.max_rpm - drone.min_rpm
        motor_speeds = drone.min_rpm + motor_commands * rpm_range
        
        return np.clip(motor_speeds, drone.min_rpm, drone.max_rpm)
    
    def _state_derivative(self, state: DroneState, drone: Drone) -> Dict[str, np.ndarray]:
        """
        Calculate state derivatives (rates of change)
        """
        # Calculate forces and torques from motors
        thrust = drone.calculate_thrust()
        torques = drone.calculate_torques()
        
        # Aerodynamic drag (opposes velocity)
        drag = -drone.drag_coeff * state.velocity * np.linalg.norm(state.velocity)
        
        # Rotation matrix: body frame -> world frame
        R = rotation_matrix(state.orientation)
        
        # Forces in body frame (thrust acts upward in body Z)
        # In NED: positive Z is down, so thrust is negative in world Z
        forces_body = np.array([0.0, 0.0, -thrust])
        
        # Transform forces to world frame and add gravity + drag
        # Gravity acts downward in world frame (positive Z in NED)
        forces_world = R @ forces_body + np.array([0.0, 0.0, drone.mass * self.gravity]) + drag
        
        # Linear acceleration (world frame)
        acceleration = forces_world / drone.mass
        
        # Angular acceleration (body frame)
        # Euler's rotation equation: I * dw/dt + w × (I * w) = torque
        inertia_times_omega = self.inertia @ state.angular_velocity
        angular_acceleration = self.inertia_inv @ (
            torques - np.cross(state.angular_velocity, inertia_times_omega)
        )
        
        # Return derivatives
        return {
            'position': state.velocity,           # dp/dt = v
            'velocity': acceleration,             # dv/dt = a
            'orientation': state.angular_velocity, # dΘ/dt = ω (simplified for small angles)
            'angular_velocity': angular_acceleration  # dω/dt = α
        }
    
    def _add_state(self, state: DroneState, derivative: Dict[str, np.ndarray], dt: float) -> DroneState:
        """
        Create intermediate state for RK4 integration
        """
        new_state = DroneState()
        
        # Advance each state variable
        new_state.position = state.position + derivative['position'] * dt
        new_state.velocity = state.velocity + derivative['velocity'] * dt
        new_state.orientation = state.orientation + derivative['orientation'] * dt
        new_state.angular_velocity = state.angular_velocity + derivative['angular_velocity'] * dt
        
        # Copy other fields
        new_state.acceleration = derivative['velocity']  # Current acceleration
        new_state.timestamp = state.timestamp
        
        return new_state