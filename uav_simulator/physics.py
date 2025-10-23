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
        self.gravity = 9.80665  # m/s^2
        self.inertia = np.diag([0.008, 0.008, 0.015])
        self.inertia_inv = np.linalg.inv(self.inertia)
        self.ground_level = 0.0

        # SAFE PHYSICAL LIMITS
        self.max_pitch = np.radians(45)  # 45° pitch up/down (non-acro, safe)
        self.max_roll  = np.radians(45)  # 45° roll left/right
        self.max_yaw_rate = np.radians(270)  # 270°/s yaw rate
        self.max_ang_vel = np.radians(120)   # 120°/s for pitch/roll angular velocity
        self.max_torque = 1.5  # Nm, conservative (adjust for your drone inertia!)

        logger.info("✓ Physics engine initialized with SAFE attitude clamps")

    def update(self, drone: Drone, control_input: np.ndarray, dt: float) -> DroneState:
        motor_speeds = self._control_to_motors(control_input, drone)
        drone.set_motor_speeds(motor_speeds)

        try:
            k1 = self._state_derivative(drone.true_state, drone)
            k2 = self._state_derivative(self._add_state(drone.true_state, k1, dt/2), drone)
            k3 = self._state_derivative(self._add_state(drone.true_state, k2, dt/2), drone)
            k4 = self._state_derivative(self._add_state(drone.true_state, k3, dt), drone)
            derivative = {}
            for key in k1.keys():
                derivative[key] = (k1[key] + 2*k2[key] + 2*k3[key] + k4[key]) / 6
        except Exception as e:
            logger.error(f"Physics integration error: {e}")
            return drone.true_state

        new_state = DroneState()
        current_thrust = drone.calculate_thrust()
        weight_force = drone.mass * self.gravity

        # Position/velocity updates
        new_state.position = drone.true_state.position + derivative['position'] * dt
        new_state.velocity = drone.true_state.velocity + derivative['velocity'] * dt

        # Orientation/attitude updates
        new_state.orientation = drone.true_state.orientation + derivative['orientation'] * dt
        new_state.orientation = normalize_angles(new_state.orientation)
        new_state.angular_velocity = drone.true_state.angular_velocity + derivative['angular_velocity'] * dt

        # CLAMP orientation and angular velocity for safety
        # Roll (x), Pitch (y), Yaw (z)
        new_state.orientation[0] = np.clip(new_state.orientation[0], -self.max_roll, self.max_roll)
        new_state.orientation[1] = np.clip(new_state.orientation[1], -self.max_pitch, self.max_pitch)
        # Yaw can wrap, but clamp rate below

        # Clamp angular velocities
        new_state.angular_velocity[0] = np.clip(new_state.angular_velocity[0], -self.max_ang_vel, self.max_ang_vel)
        new_state.angular_velocity[1] = np.clip(new_state.angular_velocity[1], -self.max_ang_vel, self.max_ang_vel)
        new_state.angular_velocity[2] = np.clip(new_state.angular_velocity[2], -self.max_yaw_rate, self.max_yaw_rate)

        new_state.acceleration = derivative['velocity']
        new_state.timestamp = drone.true_state.timestamp + dt

        # Ground protection
        if new_state.position[2] >= self.ground_level:
            if current_thrust < weight_force * 0.8:
                new_state.position[2] = self.ground_level
                if new_state.velocity[2] > 0:
                    new_state.velocity[2] = 0.0
                elif new_state.velocity[2] < 0 and current_thrust < weight_force:
                    new_state.velocity[2] *= 0.5

        drone.true_state = new_state
        drone.estimated_state = DroneState()
        drone.estimated_state.position = new_state.position.copy()
        drone.estimated_state.velocity = new_state.velocity.copy()
        drone.estimated_state.orientation = new_state.orientation.copy()
        drone.estimated_state.angular_velocity = new_state.angular_velocity.copy()
        drone.estimated_state.acceleration = new_state.acceleration.copy()
        drone.estimated_state.timestamp = new_state.timestamp
        return new_state

    def _control_to_motors(self, control: np.ndarray, drone: Drone) -> np.ndarray:
        throttle = np.clip(control[0], 0.0, 1.0)
        roll = np.clip(control[1], -0.5, 0.5)    
        pitch = np.clip(control[2], -0.5, 0.5)  
        yaw = np.clip(control[3], -0.3, 0.3)

        # Standard X-configuration motor mixing
        m0 = throttle + pitch + roll - yaw  
        m1 = throttle + pitch - roll + yaw 
        m2 = throttle - pitch - roll - yaw  
        m3 = throttle - pitch + roll + yaw 

        motor_commands = np.array([m0, m1, m2, m3])
    
        # REMOVED: The normalization that was causing issues
        # This was fighting the controller and reducing responsiveness
    
        # Simple clipping to valid range
        motor_commands = np.clip(motor_commands, 0.0, 1.0)
    
        # Scale to RPM range
        rpm_range = drone.max_rpm - drone.min_rpm
        motor_speeds = drone.min_rpm + motor_commands * rpm_range
        return np.clip(motor_speeds, drone.min_rpm, drone.max_rpm)

    def _state_derivative(self, state: DroneState, drone: Drone) -> Dict[str, np.ndarray]:
        thrust = drone.calculate_thrust()
        torques = drone.calculate_torques()
        # Clamp torques to reasonable values to prevent flips
        torques = np.clip(torques, -self.max_torque, self.max_torque)

        drag = -drone.drag_coeff * state.velocity * np.linalg.norm(state.velocity)
        R = rotation_matrix(state.orientation)
        forces_body = np.array([0.0, 0.0, -thrust])
        forces_world = R @ forces_body + np.array([0.0, 0.0, drone.mass * self.gravity]) + drag
        acceleration = forces_world / drone.mass
        inertia_times_omega = self.inertia @ state.angular_velocity
        k_damp = 0.2
        damping_torque = -k_damp * state.angular_velocity
        linear_damping = np.array([
            -0.5 * state.velocity[0],
            -0.5 * state.velocity[1],
            -0.25 * state.velocity[2]
        ])
        angular_acceleration = self.inertia_inv @ (
            torques + damping_torque - np.cross(state.angular_velocity, inertia_times_omega)
        )
        forces_world += linear_damping * drone.mass
        return {
            'position': state.velocity,
            'velocity': acceleration,
            'orientation': state.angular_velocity,
            'angular_velocity': angular_acceleration
        }

    def _add_state(self, state: DroneState, derivative: Dict[str, np.ndarray], dt: float) -> DroneState:
        new_state = DroneState()
        new_state.position = state.position + derivative['position'] * dt
        new_state.velocity = state.velocity + derivative['velocity'] * dt
        new_state.orientation = state.orientation + derivative['orientation'] * dt
        new_state.angular_velocity = state.angular_velocity + derivative['angular_velocity'] * dt
        new_state.acceleration = derivative['velocity']
        new_state.timestamp = state.timestamp
        return new_state