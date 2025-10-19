"""
Simulation Manager and Dashboard Interface
"""
import time
import threading
import numpy as np
from typing import Dict, List, Optional
from collections import deque

from .utils import logger, check_imports, FlightMode
from .flight_controller import FlightController
from .datalogger import DataLogger

# Check for visualization dependencies
IMPORTS = check_imports()
HAS_DASH = IMPORTS.get('dash', False)
HAS_PLOTLY = IMPORTS.get('plotly', False)
HAS_DBC = IMPORTS.get('dbc', False)

if HAS_DASH:
    import dash
    from dash import dcc, html, Input, Output
    if HAS_DBC:
        import dash_bootstrap_components as dbc
    if HAS_PLOTLY:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

class SimulationManager:
    """
    Manages the complete simulation with data logging capabilities
    """
    
    def __init__(self):
        self.flight_controller = FlightController()
        if HAS_DASH:
            self.dashboard = UAVDashboard(self.flight_controller)
        else:
            self.dashboard = None
        
        # Simulation state
        self.running = False
        self.simulation_thread = None
        self.real_time_factor = 1.0
        
        # Performance monitoring
        self.update_times = deque(maxlen=100)
        
        # Initialize mission
        self._initialize_mission()
    
    def _initialize_mission(self):
        """Initialize with a realistic 3D mission"""
        waypoints = []
        radius = 15
        height_step = 2
        num_points = 20
        
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = radius * np.cos(angle)
            y = radius * np.sin(angle) 
            z = -10 - (i * height_step)
            waypoints.append(np.array([x, y, z]))
        
        # Return to start
        waypoints.append(np.array([0, 0, -10]))
        
        self.flight_controller.set_waypoints(waypoints)
        self.flight_controller.set_flight_mode(FlightMode.AUTO)
    
    def start_simulation(self):
        """Start the simulation"""
        if self.running:
            return
            
        self.running = True
        self.simulation_thread = threading.Thread(target=self._simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        logger.info("Simulation started")
    
    def _simulation_loop(self):
        """Main simulation loop"""
        last_time = time.time()
        
        while self.running:
            current_time = time.time()
            dt = current_time - last_time
            
            # Update flight controller
            start_time = time.time()
            self.flight_controller.update()
            update_time = time.time() - start_time
            
            self.update_times.append(update_time)
            
            # Maintain real-time factor
            expected_dt = self.flight_controller.dt / self.real_time_factor
            sleep_time = max(0, expected_dt - update_time)
            time.sleep(sleep_time)
            
            last_time = current_time
    
    def stop_simulation(self):
        """Stop the simulation and data logging"""
        self.running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=2.0)
        
        # Stop data logging
        self.flight_controller.stop_logging()
        
        # Generate final report
        report = self.flight_controller.get_flight_report()
        print("\n" + "="*50)
        print("SIMULATION COMPLETE - FLIGHT REPORT")
        print("="*50)
        print(report)
        
        logger.info("Simulation stopped")
    
    def get_performance_stats(self) -> Dict:
        """Get simulation performance statistics"""
        if not self.update_times:
            return {}
            
        times = list(self.update_times)
        return {
            'update_rate': 1.0 / np.mean(times) if np.mean(times) > 0 else 0,
            'update_time_mean': np.mean(times),
            'update_time_std': np.std(times),
            'real_time_factor': self.real_time_factor
        }
    
    def set_real_time_factor(self, factor: float):
        """Set real-time simulation factor"""
        self.real_time_factor = max(0.1, min(10.0, factor))
    
    def run_with_dashboard(self, dashboard_port: int = 8050):
        """Run simulation with web dashboard"""
        if not HAS_DASH or self.dashboard is None:
            print("Dash not available. Cannot run dashboard.")
            print("To enable the dashboard, install: pip install dash plotly dash-bootstrap-components")
            return
            
        logger.info("Starting simulation with dashboard...")
        
        # Start simulation
        self.start_simulation()
        
        try:
            # Start dashboard
            self.dashboard.run(port=dashboard_port)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
        finally:
            self.stop_simulation()

    def run_headless(self):
        """Run simulation without dashboard"""
        logger.info("Starting headless simulation...")
        
        # Start simulation
        self.start_simulation()
        
        try:
            # Keep running until interrupted
            while self.running:
                time.sleep(1)
                # Print some basic telemetry
                telemetry = self.flight_controller.get_telemetry()
                altitude = -telemetry['position'][2]
                print(f"Position: N{telemetry['position'][0]:.1f}, E{telemetry['position'][1]:.1f}, Alt{altitude:.1f}m, Mode: {telemetry['flight_mode']}")
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.stop_simulation()
    
    def export_log_data(self, output_file: str = "flight_analysis.csv"):
        """Export current flight data to CSV for analysis"""
        try:
            recent_logs = self.flight_controller.get_recent_logs()
            if recent_logs:
                # Copy the most recent log to the specified location
                import shutil
                shutil.copy2(recent_logs[0], output_file)
                logger.info(f"Flight data exported to: {output_file}")
                return True
            else:
                logger.warning("No flight data available to export")
                return False
        except Exception as e:
            logger.error(f"Error exporting flight data: {e}")
            return False

if HAS_DASH:
    class UAVDashboard:
        """
        Modern minimalistic dashboard for real-time monitoring and control
        Clean, professional design with better UX
        """
        
        def __init__(self, flight_controller: FlightController):
            self.fc = flight_controller
            
            # FIX: Check if dbc is available before using it
            if HAS_DBC:
                self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
            else:
                self.app = dash.Dash(__name__)
                print("Using basic Dash components without Bootstrap styling")
                
            self.setup_layout()
            self.setup_callbacks()
            
            # Data buffers for plotting
            self.position_history = deque(maxlen=200)
            self.attitude_history = deque(maxlen=200)
            self.control_history = deque(maxlen=200)
            self.start_time = time.time()
            
            # Initialize with some data to prevent empty plots
            self._initialize_sample_data()
            
        def _initialize_sample_data(self):
            """Initialize with sample data to prevent empty plots"""
            current_time = time.time()
            for i in range(10):
                sample_time = current_time - (10 - i) * 0.1
                self.position_history.append({
                    'time': sample_time,
                    'x': i * 0.5,
                    'y': i * 0.3, 
                    'z': -10 - i * 0.2
                })
                self.attitude_history.append({
                    'time': sample_time,
                    'roll': np.sin(i * 0.5) * 0.1,
                    'pitch': np.cos(i * 0.3) * 0.08,
                    'yaw': i * 0.02
                })
        
        def setup_layout(self):
            """Setup the modern minimalistic layout"""
            # Layout implementation from original code
            # (This would be the full dashboard layout code)
            pass
            
        def setup_callbacks(self):
            """Setup dashboard callbacks"""
            # Callback implementation from original code
            pass
            
        def run(self, debug: bool = False, port: int = 8050):
            """Start the dashboard server"""
            logger.info(f"🚀 Starting Modern UAV Dashboard on http://localhost:{port}")
            self.app.run(debug=debug, port=port, host='0.0.0.0')