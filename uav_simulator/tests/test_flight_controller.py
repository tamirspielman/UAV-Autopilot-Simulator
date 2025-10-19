"""
Unit tests for flight controller components
"""
import unittest
import numpy as np
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uav_simulator.flight_controller import FlightController
from uav_simulator.utils import FlightMode
from uav_simulator.dynamics import UAVState

class TestFlightController(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.fc = FlightController()
    
    def test_initialization(self):
        """Test flight controller initialization"""
        # Test basic attributes
        self.assertEqual(self.fc.flight_mode, FlightMode.STABILIZE)
        self.assertIsInstance(self.fc.state, UAVState)
        self.assertIsInstance(self.fc.estimated_state, UAVState)
        self.assertIsInstance(self.fc.control_output, np.ndarray)
        self.assertEqual(len(self.fc.control_output), 4)
        
        # Test setpoints
        self.assertIn('altitude', self.fc.setpoints)
        self.assertIn('position', self.fc.setpoints)
        self.assertIn('velocity', self.fc.setpoints)
        self.assertIn('attitude', self.fc.setpoints)
        
        # Test timing
        self.assertEqual(self.fc.dt, 0.01)
        self.assertIsInstance(self.fc.last_update, float)
    
    def test_flight_mode_transitions(self):
        """Test flight mode transitions"""
        initial_mode = self.fc.flight_mode
        
        # Test transition to ALTITUDE_HOLD
        self.fc.set_flight_mode(FlightMode.ALTITUDE_HOLD)
        self.assertEqual(self.fc.flight_mode, FlightMode.ALTITUDE_HOLD)
        
        # Test transition to POSITION_HOLD
        self.fc.set_flight_mode(FlightMode.POSITION_HOLD)
        self.assertEqual(self.fc.flight_mode, FlightMode.POSITION_HOLD)
        
        # Test that controllers are reset on mode change
        # (This would require accessing internal PID states)
    
    def test_waypoint_management(self):
        """Test waypoint setting and management"""
        waypoints = [
            np.array([10, 0, -10]),
            np.array([10, 10, -15]),
            np.array([0, 10, -10])
        ]
        
        self.fc.set_waypoints(waypoints)
        
        self.assertEqual(len(self.fc.waypoints), 3)
        self.assertEqual(self.fc.current_waypoint_index, 0)
        self.assertFalse(self.fc.mission_complete)
        
        # Verify waypoint values
        np.testing.assert_array_equal(self.fc.waypoints[0], waypoints[0])
        np.testing.assert_array_equal(self.fc.waypoints[1], waypoints[1])
        np.testing.assert_array_equal(self.fc.waypoints[2], waypoints[2])
    
    def test_control_computation(self):
        """Test control computation for different flight modes"""
        # Test stabilize mode
        self.fc.set_flight_mode(FlightMode.STABILIZE)
        control = self.fc._compute_stabilize_control()
        self.assertIsInstance(control, np.ndarray)
        self.assertEqual(len(control), 4)
        self.assertTrue(all(-1 <= x <= 1 for x in control[1:]))  # Roll, pitch, yaw limits
        self.assertTrue(0 <= control[0] <= 1)  # Throttle limits
        
        # Test altitude hold mode
        self.fc.set_flight_mode(FlightMode.ALTITUDE_HOLD)
        control = self.fc._compute_altitude_hold_control()
        self.assertIsInstance(control, np.ndarray)
        self.assertEqual(len(control), 4)
    
    def test_telemetry_data(self):
        """Test telemetry data generation"""
        telemetry = self.fc.get_telemetry()
        
        # Check required fields
        required_fields = [
            'position', 'velocity', 'attitude', 'flight_mode',
            'battery', 'gps_fix', 'waypoint_index', 'mission_complete'
        ]
        
        for field in required_fields:
            self.assertIn(field, telemetry)
        
        # Check data types
        self.assertIsInstance(telemetry['position'], list)
        self.assertIsInstance(telemetry['velocity'], list)
        self.assertIsInstance(telemetry['attitude'], list)
        self.assertIsInstance(telemetry['flight_mode'], str)
        self.assertIsInstance(telemetry['battery'], float)
        self.assertIsInstance(telemetry['gps_fix'], bool)
        self.assertIsInstance(telemetry['waypoint_index'], int)
        self.assertIsInstance(telemetry['mission_complete'], bool)
    
    def test_manual_control(self):
        """Test manual control input"""
        manual_control = np.array([0.7, 0.1, -0.1, 0.05])  # Throttle, roll, pitch, yaw
        
        # Set to manual mode
        self.fc.set_flight_mode(FlightMode.MANUAL)
        
        # Update with manual control
        state = self.fc.update(manual_control=manual_control)
        
        self.assertIsInstance(state, UAVState)
        # In manual mode, control output should match manual input
        np.testing.assert_array_equal(self.fc.control_output, manual_control)
    
    def test_auto_control_without_waypoints(self):
        """Test autonomous control when no waypoints are set"""
        self.fc.set_flight_mode(FlightMode.AUTO)
        self.fc.waypoints = []  # Ensure no waypoints
        
        control = self.fc._compute_auto_control()
        
        # Should fall back to position hold
        self.assertIsInstance(control, np.ndarray)
        self.assertEqual(len(control), 4)
    
    def test_rtl_control(self):
        """Test return-to-launch control"""
        self.fc.set_flight_mode(FlightMode.RTL)
        
        # Set UAV away from home position
        self.fc.estimated_state.position = np.array([5, 5, -10])
        
        control = self.fc._compute_rtl_control()
        
        self.assertIsInstance(control, np.ndarray)
        self.assertEqual(len(control), 4)
        
        # Verify setpoint is set to origin
        np.testing.assert_array_equal(
            self.fc.setpoints['position'][:2],  # x, y should be 0,0
            np.array([0, 0])
        )
    
    def test_land_control(self):
        """Test landing control logic"""
        self.fc.set_flight_mode(FlightMode.LAND)
        
        # Start at reasonable altitude
        self.fc.estimated_state.position[2] = -20  # 20m altitude
        
        control = self.fc._compute_land_control()
        
        self.assertIsInstance(control, np.ndarray)
        self.assertEqual(len(control), 4)
        
        # Test ground detection
        self.fc.estimated_state.position[2] = -0.1  # Very close to ground
        control = self.fc._compute_land_control()
        
        # Should cut motors when on ground
        np.testing.assert_array_equal(control, np.array([0, 0, 0, 0]))

class TestControlAlgorithms(unittest.TestCase):
    
    def test_waypoint_progression(self):
        """Test waypoint progression logic"""
        fc = FlightController()
        
        waypoints = [
            np.array([5, 0, -10]),
            np.array([5, 5, -15]),
            np.array([0, 5, -10])
        ]
        fc.set_waypoints(waypoints)
        fc.set_flight_mode(FlightMode.AUTO)
        
        # Simulate reaching first waypoint
        fc.estimated_state.position = waypoints[0] + np.array([0.1, 0.1, 0.1])  # Very close
        fc.waypoint_acceptance_radius = 1.0
        
        control = fc._compute_auto_control()
        
        # Should advance to next waypoint
        self.assertEqual(fc.current_waypoint_index, 1)
        
        # Simulate reaching final waypoint
        fc.estimated_state.position = waypoints[2] + np.array([0.1, 0.1, 0.1])
        fc.current_waypoint_index = 2
        
        control = fc._compute_auto_control()
        
        # Mission should be complete
        self.assertTrue(fc.mission_complete)

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)