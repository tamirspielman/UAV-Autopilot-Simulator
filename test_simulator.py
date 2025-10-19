#!/usr/bin/env python3
"""
Test UAV Simulator Components
"""
import sys
import os

def test_components():
    """Test individual components"""
    print("Testing UAV Simulator Components...")
    
    try:
        # Test basic imports
        from uav_simulator.dynamics import UAVState, UAVDynamics
        from uav_simulator.sensor_model import SensorModel
        from uav_simulator.PIDController import PIDController
        from uav_simulator.utils import FlightMode
        
        print("✓ Basic imports successful")
        
        # Test component initialization
        state = UAVState()
        dynamics = UAVDynamics()
        sensor_model = SensorModel()
        pid = PIDController(1.0, 0.1, 0.01)
        
        print("✓ Component initialization successful")
        
        # Test flight controller
        from uav_simulator.flight_controller import FlightController
        fc = FlightController()
        print("✓ Flight controller initialized")
        
        # Test simulation manager
        from uav_simulator.simulation_manager import SimulationManager
        sim = SimulationManager()
        print("✓ Simulation manager initialized")
        
        print("\n🎉 All tests passed! The simulator is ready to run.")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_components():
        print("\nYou can now run the simulator using: python run_simulator.py")
    else:
        print("\nThere are issues with the simulator setup.")