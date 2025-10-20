"""
Main Demo Script for UAV Simulator with New Flight Modes
Run this file to start the simulator
"""

from uav_simulator.simulation_manager import SimulationManager
from uav_simulator.utils import logger

def main():
    """
    Main entry point for the UAV simulator
    """
    print("=" * 60)
    print("🚁 UAV AUTOPILOT SIMULATOR - IMPROVED VERSION")
    print("=" * 60)
    print("\nNew Flight Modes:")
    print("  🎯 STABILIZE - Hover in place (stable hovering)")
    print("  ⬆️  ALTITUDE - Control up/down movement")
    print("  ↔️  HORIZONTAL - Control left/right movement")
    print("  ↕️  LONGITUDINAL - Control forward/backward movement")
    print("  🛬 LAND - Automatic landing")
    print("\nFeatures:")
    print("  ✅ Fixed hovering stability")
    print("  ✅ Intuitive manual controls")
    print("  ✅ 3D flight path visualization")
    print("  ✅ Real-time telemetry display")
    print("  ✅ Motor speed monitoring")
    print("=" * 60)
    
    # Create simulation manager
    sim = SimulationManager()
    
    # Run with dashboard
    print("\n🚀 Starting dashboard on http://localhost:8050")
    print("📝 Press Ctrl+C to stop the simulation\n")
    
    try:
        sim.run_with_dashboard(port=8050)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
        sim.stop_simulation()
    except Exception as e:
        logger.error(f"Error: {e}")
        sim.stop_simulation()

if __name__ == "__main__":
    main()