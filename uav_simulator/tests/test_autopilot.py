"""
Unit tests for autopilot components
"""
import unittest
import numpy as np
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uav_simulator.autopilot import PIDController, UAVEnvironment
from uav_simulator.utils import FlightMode

class TestPIDController(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.pid = PIDController(kp=1.0, ki=0.1, kd=0.01, output_limits=(-1, 1))
    
    def test_initialization(self):
        """Test PID controller initialization"""
        self.assertEqual(self.pid.kp, 1.0)
        self.assertEqual(self.pid.ki, 0.1)
        self.assertEqual(self.pid.kd, 0.01)
        self.assertEqual(self.pid.output_min, -1)
        self.assertEqual(self.pid.output_max, 1)
        self.assertEqual(self.pid.integral, 0)
    
    def test_compute_basic(self):
        """Test basic PID computation"""
        output = self.pid.compute(10.0, 5.0, 0.1)  # setpoint=10, measurement=5, dt=0.1
        self.assertIsInstance(output, float)
        self.assertTrue(-1 <= output <= 1)  # Should respect output limits
    
    def test_integral_windup_protection(self):
        """Test anti-windup protection"""
        # Create a large error that would cause windup
        for _ in range(100):
            output = self.pid.compute(100.0, 0.0, 0.1)
        
        # Integral should be limited
        self.assertTrue(abs(self.pid.integral) <= self.pid.integral_limit)
    
    def test_reset(self):
        """Test controller reset"""
        # Run the controller to accumulate state
        self.pid.compute(10.0, 5.0, 0.1)
        self.pid.compute(10.0, 6.0, 0.1)
        
        # Reset and verify state is cleared
        self.pid.reset()
        self.assertEqual(self.pid.integral, 0)
        self.assertEqual(self.pid.prev_error, 0)
        self.assertEqual(self.pid.prev_derivative, 0)
    
    def test_output_limiting(self):
        """Test output limiting functionality"""
        # Create a PID with very aggressive gains that would exceed limits
        aggressive_pid = PIDController(kp=100.0, ki=10.0, kd=1.0, output_limits=(-5, 5))
        
        output = aggressive_pid.compute(100.0, 0.0, 0.1)
        self.assertTrue(-5 <= output <= 5)

class TestUAVEnvironment(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.env = UAVEnvironment()
    
    def test_initialization(self):
        """Test environment initialization"""
        self.assertEqual(self.env.observation_shape, (15,))
        self.assertEqual(self.env.action_shape, (4,))
        self.assertEqual(self.env.max_steps, 1000)
    
    def test_reset(self):
        """Test environment reset"""
        obs = self.env.reset()
        
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (15,))
        self.assertEqual(self.env.current_step, 0)
        self.assertIsNone(self.env.prev_action)
    
    def test_step(self):
        """Test environment step"""
        obs = self.env.reset()
        action = np.array([0.5, 0.0, 0.0, 0.0])  # Hover command
        
        next_obs, reward, done, info = self.env.step(action)
        
        self.assertIsInstance(next_obs, np.ndarray)
        self.assertEqual(next_obs.shape, (15,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        self.assertIn('state', info)
    
    def test_observation_structure(self):
        """Test observation vector structure"""
        obs = self.env.reset()
        
        # Check that observation has exactly 15 features
        self.assertEqual(len(obs), 15)
        
        # Check data type
        self.assertEqual(obs.dtype, np.float32)
    
    def test_termination_conditions(self):
        """Test episode termination conditions"""
        self.env.reset()
        
        # Test crash condition (too low altitude)
        self.env.state.position[2] = -0.05  # Very close to ground
        self.assertTrue(self.env._is_done())
        
        # Test out-of-bounds condition
        self.env.state.position = np.array([100, 100, -10])  # Far from origin
        self.assertTrue(self.env._is_done())
    
    def test_reward_calculation(self):
        """Test reward calculation"""
        self.env.reset()
        action = np.array([0.5, 0.0, 0.0, 0.0])
        
        reward = self.env._calculate_reward(action)
        
        self.assertIsInstance(reward, float)
        
        # Test that being close to target gives higher reward
        self.env.state.position = self.env.target_position.copy()
        high_reward = self.env._calculate_reward(action)
        
        self.env.state.position = np.array([100, 100, -100])  # Far from target
        low_reward = self.env._calculate_reward(action)
        
        self.assertGreater(high_reward, low_reward)

class TestFlightModes(unittest.TestCase):
    
    def test_flight_mode_enum(self):
        """Test FlightMode enum values"""
        self.assertEqual(FlightMode.MANUAL.value, "manual")
        self.assertEqual(FlightMode.STABILIZE.value, "stabilize")
        self.assertEqual(FlightMode.ALTITUDE_HOLD.value, "altitude_hold")
        self.assertEqual(FlightMode.POSITION_HOLD.value, "position_hold")
        self.assertEqual(FlightMode.AUTO.value, "auto")
        self.assertEqual(FlightMode.RTL.value, "return_to_launch")
        self.assertEqual(FlightMode.LAND.value, "land")
        self.assertEqual(FlightMode.AI_PILOT.value, "ai_pilot")
    
    def test_flight_mode_from_string(self):
        """Test creating FlightMode from string"""
        mode = FlightMode("stabilize")
        self.assertEqual(mode, FlightMode.STABILIZE)

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)