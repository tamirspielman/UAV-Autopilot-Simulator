"""
Autopilot Logic, Flight Modes, and Reinforcement Learning Components
"""
import numpy as np
from typing import List, Optional, Dict, Tuple
import os
import time

from .utils import FlightMode, logger, check_imports
from .dynamics import UAVState
from .sensor_model import SensorData

# Check for dependencies
IMPORTS = check_imports()
HAS_TORCH = IMPORTS.get('torch', False)

if HAS_TORCH:
    import torch
    import torch.nn as nn

class PIDController:
    """
    PID controller with anti-windup and derivative filtering
    """
    
    def __init__(self, kp: float, ki: float, kd: float, 
                 output_limits: Tuple[float, float] = (-1, 1)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min, self.output_max = output_limits
        
        # Internal state
        self.integral = 0
        self.prev_error = 0
        self.prev_derivative = 0
        
        # Anti-windup
        self.integral_limit = 10
        
        # Derivative filter
        self.derivative_filter_alpha = 0.1
        
    def compute(self, setpoint: float, measurement: float, dt: float) -> float:
        """Compute PID output"""
        error = setpoint - measurement
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral
        
        # Derivative term with filtering
        if dt > 0:
            derivative = (error - self.prev_error) / dt
            # Low-pass filter on derivative
            derivative = (self.derivative_filter_alpha * derivative + 
                         (1 - self.derivative_filter_alpha) * self.prev_derivative)
            self.prev_derivative = derivative
            d_term = self.kd * derivative
        else:
            d_term = 0
        
        # Compute output
        output = p_term + i_term + d_term
        
        # Apply limits
        output_limited = np.clip(output, self.output_min, self.output_max)
        
        # Back-calculation anti-windup
        if output != output_limited and np.sign(error) == np.sign(output - output_limited):
            self.integral -= error * dt
        
        self.prev_error = error
        
        return output_limited
    
    def reset(self):
        """Reset controller state"""
        self.integral = 0
        self.prev_error = 0
        self.prev_derivative = 0

class UAVEnvironment:
    """
    Custom UAV environment for control
    Simplified version without Gym dependency
    """
    
    def __init__(self):
        # Import here to avoid circular imports
        from .dynamics import UAVDynamics
        from .sensor_model import SensorModel
        
        # Initialize dynamics and sensors
        self.dynamics = UAVDynamics()
        self.sensor_model = SensorModel()
        self.dt = 0.01
        
        # State and action spaces - CORRECTED: 15 features, not 18
        self.observation_shape = (15,)  # Fixed: 15 features
        self.action_shape = (4,)
        self.action_low = np.array([0, -1, -1, -1])
        self.action_high = np.array([1, 1, 1, 1])
        
        # Episode parameters
        self.max_steps = 1000
        self.current_step = 0
        
        # Target position for training
        self.target_position = np.array([0, 0, -10])  # 10m altitude
        
        # Track previous action for smoothness penalty
        self.prev_action = None
        
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.state = UAVState()
        self.state.position = np.array([0, 0, -1])  # Start 1m above ground
        self.current_step = 0
        self.prev_action = None
        return self._get_observation()
    
    def step(self, action: np.ndarray):
        """Execute one environment step"""
        # Store action for reward calculation
        current_action = action.copy()
        
        # Apply action
        self.state = self.dynamics.update(self.state, action, self.dt)
        
        # Get observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(current_action)
        
        # Check termination
        done = self._is_done()
        
        self.current_step += 1
        
        info = {'state': self.state.to_dict()}
        
        return obs, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get observation vector from state - CORRECTED: 15 features"""
        sensor_data = self.sensor_model.measure(self.state, self.dt)
        
        # CORRECTED: Use only 15 features that match the neural network input
        obs = np.concatenate([
            self.state.position,           # 3 features
            self.state.velocity,           # 3 features  
            self.state.orientation,        # 3 features
            self.state.angular_velocity,   # 3 features
            sensor_data.imu_accel[:3]      # 3 features (only first 3 elements)
        ])
        # Total: 3+3+3+3+3 = 15 features
        
        return obs.astype(np.float32)
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward for RL training"""
        # Position error
        pos_error = np.linalg.norm(self.state.position - self.target_position)
        pos_reward = np.exp(-0.1 * pos_error)
        
        # Velocity penalty (encourage stable hover)
        vel_penalty = -0.01 * np.linalg.norm(self.state.velocity)
        
        # Orientation penalty (encourage level flight)
        orient_penalty = -0.1 * np.linalg.norm(self.state.orientation[:2])  # roll and pitch
        
        # Control smoothness penalty
        if self.prev_action is not None:
            action_diff = np.linalg.norm(self.prev_action - action)
            smooth_penalty = -0.01 * action_diff
        else:
            smooth_penalty = 0
            
        self.prev_action = action.copy()
        
        # Total reward
        total_reward = pos_reward + vel_penalty + orient_penalty + smooth_penalty
        
        # Bonus for reaching target
        if pos_error < 0.5:
            total_reward += 10
            
        return total_reward
    
    def _is_done(self) -> bool:
        """Check if episode should terminate"""
        # Check if crashed (too low altitude)
        if self.state.position[2] > -0.1:  # Too close to ground
            return True
            
        # Check if too far from target
        if np.linalg.norm(self.state.position) > 50:
            return True
            
        # Check max steps
        if self.current_step >= self.max_steps:
            return True
            
        return False

if HAS_TORCH:
    class PPONetwork(nn.Module):
        """Neural network for PPO policy and value functions - CORRECTED INPUT SIZE"""
        
        def __init__(self, obs_dim: int = 15, action_dim: int = 4):  # Fixed: 15 input features
            super().__init__()
            
            # Shared feature extractor
            self.shared_net = nn.Sequential(
                nn.Linear(obs_dim, 256),  # Now expects 15 inputs
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
            )
            
            # Policy head
            self.policy_net = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
                nn.Tanh()  # Actions in [-1, 1]
            )
            
            # Value head
            self.value_net = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            
            # Log standard deviation for continuous actions
            self.log_std = nn.Parameter(torch.zeros(1, action_dim))
            
        def forward(self, x):
            features = self.shared_net(x)
            action_mean = self.policy_net(features)
            value = self.value_net(features)
            return action_mean, value

class RLAutopilot:
    """
    Reinforcement Learning based autopilot using PPO
    Can be trained to perform complex maneuvers
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.env = UAVEnvironment()
        self.policy = None
        
        if HAS_TORCH and model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize untrained policy
            if HAS_TORCH:
                self.policy = PPONetwork(
                    obs_dim=15,  # Fixed: 15 input features
                    action_dim=4
                )
            
    def compute_control(self, observation: np.ndarray) -> np.ndarray:
        """Compute control action from observation"""
        if self.policy is None or not HAS_TORCH:
            # Return hover command if no policy loaded or no PyTorch
            return np.array([0.5, 0, 0, 0])
            
        with torch.no_grad():
            # Ensure observation has the correct size (15 features)
            if len(observation) != 15:
                # If we get more features, take only the first 15
                observation = observation[:15]
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            action_mean, _ = self.policy(obs_tensor)
            action = action_mean.squeeze(0).numpy()
            
        # Scale to appropriate control range
        action = np.clip(action, self.env.action_low, self.env.action_high)
        return action
    
    def train(self, total_timesteps: int = 10000):
        """Train the RL agent (simplified training loop)"""
        if not HAS_TORCH:
            print("PyTorch not available. Cannot train RL agent.")
            return
            
        logger.info("Starting RL training...")
        
        # Simplified training loop
        for episode in range(total_timesteps // 1000):
            obs = self.env.reset()
            episode_reward = 0
            
            for step in range(1000):
                action = self.compute_control(obs)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                
                if done:
                    break
                    
            if episode % 10 == 0:
                logger.info(f"Episode {episode}, Reward: {episode_reward:.2f}")
                
    def save_model(self, path: str):
        """Save trained model"""
        if self.policy is not None and HAS_TORCH:
            torch.save(self.policy.state_dict(), path)
        
    def load_model(self, path: str):
        """Load trained model"""
        if HAS_TORCH:
            self.policy = PPONetwork(obs_dim=15, action_dim=4)  # Fixed dimensions
            self.policy.load_state_dict(torch.load(path))
            self.policy.eval()