#!/usr/bin/env python3
"""
UAV Autopilot Simulator - Main Runner
"""
import sys
import os

def main():
    print("🚀 Starting UAV Autopilot Simulator")
    print("=" * 50)
    
    try:
        # Import and run the simulation manager
        from uav_simulator.simulation_manager import SimulationManager
        
        sim_manager = SimulationManager()
        
        # Check if dashboard is available
        try:
            from uav_simulator.simulation_manager import HAS_DASH
            if HAS_DASH:
                print("Starting with web dashboard...")
                print("Dashboard will be available at: http://localhost:8050")
                sim_manager.run_with_dashboard(port=8050)  # FIXED: added port parameter
            else:
                print("Dash not available. Running in headless mode...")
                print("To enable dashboard: pip install dash plotly dash-bootstrap-components")
                sim_manager.run_headless()
        except ImportError:
            print("Running in headless mode...")
            sim_manager.run_headless()
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()