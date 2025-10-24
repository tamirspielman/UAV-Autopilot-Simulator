import numpy as np
import heapq
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from .utils import FlightMode, logger
from .drone import Drone

#TODO:
# fix landing & RTL.
# Implement wind disturbance model

@dataclass(order=True)
class PathNode:
    """A* search node for path planning"""
    f_cost: float
    position: Tuple[float, float, float] = field(compare=False)
    g_cost: float = field(compare=False)
    h_cost: float = field(compare=False)
    parent: Optional['PathNode'] = field(default=None, compare=False)


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float,
                 output_limits: Tuple[float, float] = (-0.5, 0.5),
                 is_angle: bool = False):
        """Initialize PID controller with gains and limits"""
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
        """Compute PID output given setpoint, measurement and time delta"""
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
        """Reset controller state (integral and previous errors)"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0

class Controller:
    def __init__(self):
        """Main flight controller managing all flight modes and PID loops"""
        self.flight_mode = FlightMode.MANUAL
        self.setpoints = {
            'altitude': 0.0,
            'position': np.array([0.0, 0.0, 0.0]),  
            'yaw': 0.0
        }
        self.waypoints: List[np.ndarray] = []
        self.optimized_waypoints: List[np.ndarray] = []  
        self.current_waypoint_index = 0
        self.mission_complete = False
        self.launch_position = np.array([0.0, 0.0, 0.0])
        self.control_output = np.zeros(4)
        self.is_launched = False

        self.waypoint_radius = 0.5  # Looser for smoother flight
        self.waypoint_altitude_tolerance = 0.3  # Reasonable altitude tolerance
        self.max_velocity = 3.0  # Reduced for stability
        self.max_tilt_angle = 0.55  # Reduced max tilt (30 degrees) 
        
        # Path planning parameters
        self.grid_resolution = 1.0
        self.distance_weight = 1.0
        self.altitude_weight = 0.4
        self.energy_weight = 0.3
        self.use_path_optimization = True
        
        # Initialize PID controllers with CONSERVATIVE gains
        self.altitude_pid = PIDController(1.5, 0.1, 0.8, (-0.5, 0.5))  # Much gentler altitude control
        self.pos_x_pid = PIDController(0.3, 0.001, 0.4, (-3.0, 3.0))   # Reduced position gains
        self.pos_y_pid = PIDController(0.3, 0.001, 0.4, (-3.0, 3.0))
        self.roll_pid = PIDController(3.0, 0.05, 0.2, (-0.3, 0.3), is_angle=True)  # Reduced attitude gains
        self.pitch_pid = PIDController(3.0, 0.05, 0.2, (-0.3, 0.3), is_angle=True)
        self.yaw_pid = PIDController(1.5, 0.05, 0.15, (-0.1, 0.1), is_angle=True)
        self.vel_x_pid = PIDController(0.3, 0.003, 0.08, (-0.4, 0.4))  # Reduced velocity gains
        self.vel_y_pid = PIDController(0.3, 0.003, 0.08, (-0.4, 0.4))
        self.vel_x_pid.integral_limit = 0.05
        self.vel_y_pid.integral_limit = 0.05
        self._land_started = False
        self._land_initial_altitude = None
        self._land_burst_done = False
        self._rtl_started = False
        self._rtl_initial_altitude = None
        self._rtl_burst_done = False
        self._mission_transition_timer = 0.0
        
        logger.info("✓ Controller initialized with conservative PID gains")

    def _heuristic(self, pos: np.ndarray, goal: np.ndarray) -> float:
        """
        A* heuristic function h(n)
        Formula: h(n) = sqrt((x_goal - x_n)^2 + (y_goal - y_n)^2) + alpha*|z_goal - z_n|
        """
        dx = goal[0] - pos[0]
        dy = goal[1] - pos[1]
        dz = goal[2] - pos[2]
        
        horizontal_dist = np.sqrt(dx**2 + dy**2)
        vertical_dist = abs(dz) * self.altitude_weight
        
        return horizontal_dist + vertical_dist

    def _path_cost(self, from_pos: np.ndarray, to_pos: np.ndarray) -> float:
        """
        Cost function g(n) for A* 
        Formula: Cost = distance_weight * ||P_i+1 - P_i|| + altitude_weight * |z_i+1 - z_i|
        """
        distance = np.linalg.norm(to_pos - from_pos)
        altitude_change = abs(to_pos[2] - from_pos[2])
        
        # Climbing costs more energy than descending
        if to_pos[2] < from_pos[2]:  # Climbing (negative z is up)
            energy_cost = altitude_change * self.energy_weight * 1.5
        else:  # Descending
            energy_cost = altitude_change * self.energy_weight * 0.5
        
        return self.distance_weight * distance + altitude_change * self.altitude_weight + energy_cost

    def _get_path_neighbors(self, pos: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        """Get neighboring positions for A* search"""
        neighbors = []
        
        # Adaptive step size based on distance to goal
        dist_to_goal = np.linalg.norm(goal - pos)
        step_size = min(self.grid_resolution * 2, max(0.5, dist_to_goal / 10))
        
        # 3D directions prioritizing horizontal movement
        directions = [
            # Horizontal (same altitude) - higher priority
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
            (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
            # Vertical - lower priority
            (0, 0, 1), (0, 0, -1),
            # Diagonal 3D
            (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
            (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
        ]
        
        for dx, dy, dz in directions:
            # Scale altitude changes to prefer horizontal movement
            altitude_scale = 0.5 if dz != 0 else 1.0
            neighbor = pos + np.array([
                dx * step_size,
                dy * step_size,
                dz * step_size * altitude_scale
            ])
            
            # Altitude bounds check
            if neighbor[2] > 0 or neighbor[2] < -100:
                continue
            
            neighbors.append(neighbor)
        
        return neighbors

    def _reconstruct_path(self, node: PathNode, goal: np.ndarray) -> List[np.ndarray]:
        """Reconstruct path from A* node chain"""
        path = [goal]  # Add goal first
        current = node
        
        while current.parent is not None:
            path.append(np.array(current.position))
            current = current.parent
        
        path.reverse()
        return path

    def _plan_optimal_path(self, start: np.ndarray, goal: np.ndarray, 
                           max_iterations: int = 3000) -> Optional[List[np.ndarray]]:
        """
        A* path planning: finds shortest path minimizing J = sum(Cost(P_i, P_i+1))
        
        Formula: f(n) = g(n) + h(n)
        where g(n) = actual cost from start, h(n) = heuristic to goal
        """
        logger.info(f"🔍 Planning optimal path from {start[:2]} to {goal[:2]}")
        
        open_list = []
        closed_set = set()
        
        start_tuple = tuple(start)
        g_start = 0.0
        h_start = self._heuristic(start, goal)
        f_start = g_start + h_start
        start_node = PathNode(f_start, start_tuple, g_start, h_start, None)
        
        heapq.heappush(open_list, start_node)
        g_costs = {start_tuple: 0.0}
        
        iterations = 0
        goal_threshold = self.grid_resolution * 1.5
        
        while open_list and iterations < max_iterations:
            iterations += 1
            
            current = heapq.heappop(open_list)
            current_pos = np.array(current.position)
            
            # Goal check
            if np.linalg.norm(current_pos - goal) < goal_threshold:
                path = self._reconstruct_path(current, goal)
                total_distance = sum(np.linalg.norm(path[i+1] - path[i]) 
                                    for i in range(len(path)-1))
                logger.info(f"✓ Optimal path found: {len(path)} waypoints, "
                           f"{total_distance:.1f}m total distance ({iterations} iterations)")
                return path
            
            closed_set.add(current.position)
            
            # Expand neighbors
            for neighbor_pos in self._get_path_neighbors(current_pos, goal):
                neighbor_tuple = tuple(neighbor_pos)
                
                if neighbor_tuple in closed_set:
                    continue
                
                tentative_g = current.g_cost + self._path_cost(current_pos, neighbor_pos)
                
                if neighbor_tuple in g_costs and tentative_g >= g_costs[neighbor_tuple]:
                    continue
                
                # Better path found
                g_costs[neighbor_tuple] = tentative_g
                h_cost = self._heuristic(neighbor_pos, goal)
                f_cost = tentative_g + h_cost
                
                neighbor_node = PathNode(f_cost, neighbor_tuple, tentative_g, h_cost, current)
                heapq.heappush(open_list, neighbor_node)
        
        logger.warning(f"⚠ No path found after {iterations} iterations, using direct path")
        return None

    def _optimize_waypoints(self) -> List[np.ndarray]:
        """Optimize waypoint path using A* between each consecutive pair"""
        if not self.waypoints:
            return []
        if len(self.waypoints) < 2:
            return self.waypoints.copy()  # Return copy of single waypoint
        optimized_path = []
        current_position = self.launch_position
        for i in range(len(self.waypoints)):
            goal = self.waypoints[i]
        
            # Run A* from current position to waypoint
            segment_path = self._plan_optimal_path(current_position, goal)
        
            if segment_path:
                # Add path segment (skip first point as it's the current position)
                optimized_path.extend(segment_path[1:])
                current_position = segment_path[-1]  # Update current position
            else:
                # Fallback: direct path
                optimized_path.append(goal)
                current_position = goal
    
        return optimized_path

    def compute_control(self, drone: Drone, dt: float) -> np.ndarray:
        """Main control computation - dispatches to current flight mode"""
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
        """Stabilize mode: maintain altitude and position with attitude control"""
        # Altitude control using vertical velocity PID
        current_altitude_ned = drone.estimated_state.position[2]
        target_altitude_ned = -self.setpoints['altitude']
        altitude_error = target_altitude_ned - current_altitude_ned
        current_vertical_velocity_ned = drone.estimated_state.velocity[2]
        current_altitude_m = -current_altitude_ned
        altitude_velocity_gain = 0.8 
        desired_vertical_velocity = np.clip(
            altitude_error * altitude_velocity_gain,
            -self.max_velocity,
            self.max_velocity
        )
        
        velocity_error = desired_vertical_velocity - current_vertical_velocity_ned
        throttle_adjustment = self.altitude_pid.compute(-velocity_error, 0, dt)
        hover_throttle = drone.get_hover_throttle()
        throttle = hover_throttle + throttle_adjustment

        # Position control cascaded through velocity PIDs
        desired_roll = 0.0
        desired_pitch = 0.0
        
        # Always allow some position control, but be more conservative at low altitude
        position_gain_scale = 1.0
        if current_altitude_m < 1.0:
            position_gain_scale = 0.5  # Reduce position control at low altitude
        
        max_pos_error = 10.0
        max_desired_vel = 2.0 * position_gain_scale
        
        # Limit position commands for safety
        limited_target_x = np.clip(
            self.setpoints['position'][0],
            drone.estimated_state.position[0] - max_pos_error,
            drone.estimated_state.position[0] + max_pos_error)
        limited_target_y = np.clip(
            self.setpoints['position'][1],
            drone.estimated_state.position[1] - max_pos_error,
            drone.estimated_state.position[1] + max_pos_error)
        
        # Position to velocity cascade
        pos_x_output = self.pos_x_pid.compute(limited_target_x, drone.estimated_state.position[0], dt)
        pos_y_output = self.pos_y_pid.compute(limited_target_y, drone.estimated_state.position[1], dt)
        desired_vel_x = np.clip(pos_x_output, -max_desired_vel, max_desired_vel)
        desired_vel_y = np.clip(pos_y_output, -max_desired_vel, max_desired_vel)
        
        # Velocity to attitude cascade
        current_vel = drone.estimated_state.velocity
        pitch_command = self.vel_x_pid.compute(desired_vel_x, current_vel[0], dt)
        roll_command = self.vel_y_pid.compute(desired_vel_y, current_vel[1], dt)
        
        pitch_command = np.clip(pitch_command, -self.max_tilt_angle, self.max_tilt_angle)
        roll_command = np.clip(roll_command, -self.max_tilt_angle, self.max_tilt_angle)
        
        desired_pitch = -pitch_command
        desired_roll = roll_command

        # Attitude stabilization with roll/pitch/yaw PIDs
        current_attitude = drone.estimated_state.orientation
        roll_output  = self.roll_pid.compute(desired_roll, current_attitude[0], dt)
        pitch_output = self.pitch_pid.compute(desired_pitch, current_attitude[1], dt)
        yaw_output   = self.yaw_pid.compute(self.setpoints['yaw'], current_attitude[2], dt)
        
        # Conservative control output limits
        roll_output  = np.clip(roll_output, -0.2, 0.2)
        pitch_output = np.clip(pitch_output, -0.2, 0.2)
        yaw_output   = np.clip(yaw_output, -0.15, 0.15)
        
        # Gentle tilt compensation for throttle
        current_tilt = np.sqrt(current_attitude[0]**2 + current_attitude[1]**2)
        compensation_factor = 1.0 / max(np.cos(current_tilt), 0.7)  # Less aggressive compensation
        throttle *= compensation_factor
        
        # Remove aggressive ascent boost - this was causing the shooting up
        # Only very gentle compensation for fast descent
        if current_vertical_velocity_ned > 0.5:  # Only if descending fast
            throttle = min(throttle + 0.05, 1.0)  # Very small boost
            
        throttle = np.clip(throttle, 0.4, 0.9)  # Conservative throttle limits
        
        self.control_output = np.array([throttle, roll_output, pitch_output, yaw_output])
        return self.control_output

    def _auto_mode(self, drone: Drone, dt: float) -> np.ndarray:
        """Auto mode: follow optimized waypoint mission"""
        # Check if optimized_waypoints is empty or mission is complete
        if not self.optimized_waypoints or self.mission_complete:
            return self._stabilize_mode(drone, dt)
        
        # Check if current_waypoint_index is valid
        if self.current_waypoint_index >= len(self.optimized_waypoints):
            self.mission_complete = True
            return self._stabilize_mode(drone, dt)

        current_wp = self.optimized_waypoints[self.current_waypoint_index]
        current_altitude_m = -drone.estimated_state.position[2]
        target_altitude_m = abs(current_wp[2])

        # Set current waypoint as target
        self.setpoints['altitude'] = target_altitude_m
        self.setpoints['position'] = np.array([current_wp[0], current_wp[1], 0.0])

        # Gentle transition handling - no aggressive boosts
        if self.current_waypoint_index == 0 and self._mission_transition_timer < 2.0:
            self._mission_transition_timer += dt
            # Let the drone stabilize naturally

        control = self._stabilize_mode(drone, dt)

        # Check waypoint reaching condition
        current_pos = drone.estimated_state.position
        horizontal_distance = np.linalg.norm(current_pos[:2] - current_wp[:2])
        vertical_distance = abs(current_altitude_m - target_altitude_m)

        waypoint_reached = (
            horizontal_distance < self.waypoint_radius and 
            vertical_distance < self.waypoint_altitude_tolerance
        )

        if waypoint_reached:
            if self.current_waypoint_index < len(self.optimized_waypoints) - 1:
                # Advance to next waypoint
                self.current_waypoint_index += 1
                self._mission_transition_timer = 0.0
                next_wp = self.optimized_waypoints[self.current_waypoint_index]
                logger.info(f"✓ Waypoint {self.current_waypoint_index}/{len(self.optimized_waypoints)} reached! "
                            f"Next: N{next_wp[0]:.1f} E{next_wp[1]:.1f} Alt{abs(next_wp[2]):.1f}m")

                # Reset PIDs when switching waypoints for clean transition
                self.pos_x_pid.reset()
                self.pos_y_pid.reset()
                self.altitude_pid.reset()
                self.vel_x_pid.reset()
                self.vel_y_pid.reset()
            else:
                self.mission_complete = True
                logger.info("✓ All waypoints reached! Holding position.")

        return control
    
    def _compute_adaptive_landing_throttle(self, drone: Drone, current_altitude: float, mode: str = "LAND") -> float:
        hover = float(drone.get_hover_throttle())
        if mode == "RTL":
            # RTL wants faster descent overall, but still soft near ground
            if current_altitude > 5.0:
                throttle = hover - 0.75
            elif current_altitude > 3.0:
                t = (current_altitude - 3.0) / 2.0
                throttle = hover - (0.60 + 0.15 * t)
            elif current_altitude > 1.5:
                t = (current_altitude - 1.5) / 1.5
                throttle = hover - (0.35 + 0.25 * t)
            else:
                # soft curve below 1.5m
                soft_factor = (current_altitude / 1.5) ** 1.7
                throttle = hover - (0.05 + 0.25 * soft_factor)

        else:  # LAND
            # Start fast but slow sharply below 2m
            if current_altitude > 3.0:
                throttle = hover - 0.65
            elif current_altitude > 2.0:
                t = (current_altitude - 2.0)
                throttle = hover - (0.45 + 0.20 * t)
            elif current_altitude > 1.0:
                t = (current_altitude - 1.0)
                throttle = hover - (0.20 + 0.25 * t)
            else:
                # smooth nonlinear softening near ground
                soft_factor = (current_altitude / 1.0) ** 2.0
                throttle = hover - (0.05 + 0.15 * soft_factor)

        return float(np.clip(throttle, 0.05, 0.95))


    def _rtl_mode(self, drone: Drone, dt: float) -> np.ndarray:
        current_pos = drone.estimated_state.position
        current_altitude = -current_pos[2]
        distance_to_home_xy = np.linalg.norm(current_pos[:2] - self.launch_position[:2])
        approach_radius = max(self.waypoint_radius, 1.0)
        if not self._rtl_started:
            self._rtl_started = True
            self._rtl_burst_done = False
            self._rtl_initial_altitude = float(current_altitude)
            logger.info(f"RTL: start at {self._rtl_initial_altitude:.2f}m — initial burst enabled")

        # initial 1m burst so it doesn't stall in the descent when starting RTL
        if not self._rtl_burst_done:
            if self._rtl_initial_altitude - current_altitude >= 1.0 or current_altitude <= 1.0:
                self._rtl_burst_done = True
                logger.debug("RTL: initial burst complete")
            else:
                hover = float(drone.get_hover_throttle())
                burst_throttle = float(np.clip(hover - 0.60, 0.02, 0.95))
                # command simultaneous XY->home and allow throttle burst to descend ~1m quickly
                self.setpoints['position'] = np.array([self.launch_position[0], self.launch_position[1], -1.0])
                control = self._stabilize_mode(drone, dt)
                control[0] = burst_throttle
                logger.debug(f"RTL burst: alt {current_altitude:.2f} -> throttle {control[0]:.3f}")
                return control

        # Phase: move toward home XY while descending to 2m (simultaneous XY + altitude)
        if distance_to_home_xy > approach_radius:
            self.setpoints['position'] = np.array([self.launch_position[0], self.launch_position[1], -2.0])
            self.setpoints['altitude'] = 2.0
            control = self._stabilize_mode(drone, dt)
            # above 2m we want faster descent (allow control to go lower throttle if needed)
            if current_altitude > 2.0:
                control[0] = float(np.clip(control[0], 0.03, 0.95))
            logger.debug(f"RTL: enroute to home XY, alt {current_altitude:.2f}, thr {control[0]:.3f}")
            return control

        # Close to home XY — if above 2m, descend to 2m then soft land
        if current_altitude > 2.0:
            self.setpoints['position'] = np.array([self.launch_position[0], self.launch_position[1], -2.0])
            self.setpoints['altitude'] = 2.0
            control = self._stabilize_mode(drone, dt)
            control[0] = float(np.clip(control[0], 0.03, 0.95))
            logger.debug(f"RTL: co-located XY, descending to 2m, alt {current_altitude:.2f}, thr {control[0]:.3f}")
            return control

        # Soft-landing at home when <= 2m
        if current_altitude > 0.05:
            self.setpoints['position'] = np.array([self.launch_position[0], self.launch_position[1], 0.0])
            self.setpoints['altitude'] = 0.0
            control = self._stabilize_mode(drone, dt)
            control[0] = self._compute_adaptive_landing_throttle(drone, current_altitude, mode="RTL")
            logger.debug(f"RTL: soft-landing from {current_altitude:.2f}m, thr -> {control[0]:.3f}")
            return control

        # Landed
        logger.info("✓ RTL complete - Landed at launch position")
        self.flight_mode = FlightMode.MANUAL
        self.is_launched = False
        self._rtl_started = False
        self._rtl_initial_altitude = None
        self._rtl_burst_done = False
        return np.zeros(4)
    def _land_mode(self, drone: Drone, dt: float) -> np.ndarray:
        current_pos = drone.estimated_state.position
        current_altitude = -current_pos[2]  
        if not self._land_started:
            self._land_started = True
            self._land_burst_done = False
            self._land_initial_altitude = float(current_altitude)
            logger.info(f"LAND: start at {self._land_initial_altitude:.2f}m — initial burst enabled")

        # If we haven't done the 1m burst yet, do it until we've dropped ~1.0m
        if not self._land_burst_done:
            # If starting altitude < 1m, skip burst
            if self._land_initial_altitude - current_altitude >= 1.0 or current_altitude <= 1.0:
                self._land_burst_done = True
                logger.debug("LAND: initial burst complete")
            else:
                # aggressive throttle (fast first-meter descent)
                hover = float(drone.get_hover_throttle())
                burst_throttle = float(np.clip(hover - 0.60, 0.02, 0.95))
                # Keep XY on current position while bursting down
                self.setpoints['position'] = np.array([current_pos[0], current_pos[1], 0.0])
                control = self._stabilize_mode(drone, dt)
                control[0] = burst_throttle
                logger.debug(f"LAND burst: alt {current_altitude:.2f} -> throttle {control[0]:.3f}")
                return control
        if current_altitude > 0.1:
            self.setpoints['position'] = np.array([self.launch_position[0], self.launch_position[1], 0.0])
            self.setpoints['altitude'] = 0.0
            control = self._stabilize_mode(drone, dt)
            control[0] = self._compute_adaptive_landing_throttle(drone, current_altitude, mode="LAND")
            logger.debug(f"RTL: adaptive descent from {current_altitude:.2f}m, thr -> {control[0]:.3f}")
            return control

        # Soft-landing phase when <= 2.0m: use soft throttle curve
        if current_altitude > 0.05:
            # lock XY, then apply soft throttle mapping
            self.setpoints['position'] = np.array([current_pos[0], current_pos[1], 0.0])
            self.setpoints['altitude'] = 0.0
            control = self._stabilize_mode(drone, dt)
            control[0] = self._compute_adaptive_landing_throttle(drone, current_altitude, mode="LAND")
            logger.debug(f"LAND: soft-landing from {current_altitude:.2f}m, thr -> {control[0]:.3f}")
            return control

        # Landed
        logger.info("✓ Land complete - landed at current position")
        self.flight_mode = FlightMode.MANUAL
        self.is_launched = False
        self._land_started = False
        self._land_initial_altitude = None
        self._land_burst_done = False
        return np.zeros(4)

    def set_launch_position(self, north: float, east: float, altitude: float = 0.0):
        """Set the launch position for RTL reference"""
        self.launch_position = np.array([north, east, -altitude])
        logger.info(f"Launch position set to: N{north:.1f} E{east:.1f} Alt{altitude:.1f}m")

    def launch(self, target_altitude: float = 2.0):
        """Launch sequence: take off to specified altitude"""
        if self.is_launched:
            logger.warning("Already launched!")
            return
            
        self.launch_position = self.setpoints['position'].copy()
        launch_target = self.launch_position.copy()
        launch_target[2] = -target_altitude
        
        self.setpoints['altitude'] = target_altitude
        self.setpoints['position'] = launch_target
        self.setpoints['yaw'] = 0.0
        
        # Reset all controllers for clean start
        self.altitude_pid.reset()
        self.pos_x_pid.reset()
        self.pos_y_pid.reset()
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.yaw_pid.reset()
        self.vel_x_pid.reset()
        self.vel_y_pid.reset()
        
        self._rtl_started = False
        self._land_started = False
        self._mission_transition_timer = 0.0
        self.set_flight_mode(FlightMode.STABILIZE)
        self.is_launched = True
        logger.info(f"🚀 Launching to {target_altitude}m altitude!")

    def add_waypoint(self, north: float, east: float, altitude: float):
        """Add a waypoint to the mission"""
        waypoint = np.array([north, east, -abs(altitude)])
        self.waypoints.append(waypoint)
        logger.info(f"📍 Added waypoint: N{north:.1f} E{east:.1f} Alt{altitude:.1f}m")

    def clear_waypoints(self):
        """Clear all waypoints from mission"""
        self.waypoints = []
        self.optimized_waypoints = []
        self.current_waypoint_index = 0
        self.mission_complete = False
        self._mission_transition_timer = 0.0
        logger.info("Waypoints cleared")
        
    def start_mission(self):
        """Start the waypoint mission with A* path optimization"""
        if not self.waypoints:
            logger.warning("No waypoints set!")
            return

        if not self.is_launched:
            logger.warning("Not launched yet! Auto-launching to 2m...")
            self.launch(2.0)
            return

        # Optimize path using A* if enabled
        if self.use_path_optimization:
            logger.info("🎯 Optimizing mission path with A* algorithm...")
            self.optimized_waypoints = self._optimize_waypoints()
            # If optimization failed, fall back to original waypoints
            if not self.optimized_waypoints:
                logger.warning("Path optimization failed, using original waypoints")
                self.optimized_waypoints = self.waypoints.copy()
            logger.info(f"📊 Optimization complete: {len(self.waypoints)} user waypoints → "
                       f"{len(self.optimized_waypoints)} optimized waypoints")
        else:
            self.optimized_waypoints = self.waypoints.copy()

        # Reset PIDs for clean mission start
        self.altitude_pid.reset()
        self.pos_x_pid.reset()
        self.pos_y_pid.reset()
        self.vel_x_pid.reset()
        self.vel_y_pid.reset()
        self._mission_transition_timer = 0.0
        
        self.current_waypoint_index = 0
        self.mission_complete = False
        self.set_flight_mode(FlightMode.AUTO)
        logger.info(f"🎯 Mission started with {len(self.optimized_waypoints)} waypoints")
        
    def set_flight_mode(self, mode: FlightMode):
        """Change flight mode and reset controllers"""
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
            self._mission_transition_timer = 0.0
            self.flight_mode = mode

    def get_status(self) -> Dict[str, Any]:
        """Get current controller status for telemetry"""
        return {
            'flight_mode': self.flight_mode.value,
            'waypoint_index': self.current_waypoint_index,
            'total_waypoints': len(self.optimized_waypoints),
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