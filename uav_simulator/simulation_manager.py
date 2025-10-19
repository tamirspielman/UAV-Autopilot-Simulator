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
    
    def run_with_dashboard(self, port: int = 8050):
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
            self.dashboard.run(port=port)
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

# Move the UAVDashboard class outside of SimulationManager class
if HAS_DASH:
    class UAVDashboard:
        """
        Simplified UAV Dashboard - Working Version
        """
        
        def __init__(self, flight_controller: FlightController):
            self.fc = flight_controller
            
            # Create the app
            if HAS_DBC:
                self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
            else:
                self.app = dash.Dash(__name__)
            
            # Set up the layout
            self.setup_layout()
            
            # Set up callbacks
            self.setup_callbacks()
            
            # Data buffers
            self.position_history = deque(maxlen=100)
            self.attitude_history = deque(maxlen=100)
            self.start_time = time.time()
        
        def setup_layout(self):
            """Setup a simple but functional layout"""
            if HAS_DBC:
                self.app.layout = dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            html.H1("UAV Autopilot Simulator", 
                                   style={'textAlign': 'center', 'color': 'white', 'marginBottom': '20px'})
                        ], width=12)
                    ]),
                    
                    dbc.Row([
                        # Left column - Controls and Status
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Flight Controls"),
                                dbc.CardBody([
                                    html.Label("Flight Mode"),
                                    dcc.Dropdown(
                                        id='flight-mode-dropdown',
                                        options=[
                                            {'label': 'Manual', 'value': 'manual'},
                                            {'label': 'Stabilize', 'value': 'stabilize'},
                                            {'label': 'Altitude Hold', 'value': 'altitude_hold'},
                                            {'label': 'Position Hold', 'value': 'position_hold'},
                                            {'label': 'Auto', 'value': 'auto'},
                                            {'label': 'Return to Launch', 'value': 'return_to_launch'},
                                            {'label': 'Land', 'value': 'land'},
                                            {'label': 'AI Pilot', 'value': 'ai_pilot'}
                                        ],
                                        value='auto'
                                    ),
                                    html.Br(),
                                    html.Label("Target Altitude (m)"),
                                    dcc.Slider(
                                        id='altitude-slider',
                                        min=1, max=100, step=1, value=10,
                                        marks={i: str(i) for i in range(0, 101, 20)}
                                    ),
                                    html.Br(),
                                    dbc.Button("Takeoff", id='takeoff-btn', color="success", className='w-100 mb-2'),
                                    dbc.Button("Land", id='land-btn', color="warning", className='w-100 mb-2'),
                                    dbc.Button("RTL", id='rtl-btn', color="danger", className='w-100')  # Fixed missing quote
                                ])
                            ], className='mb-3'),
                            
                            dbc.Card([
                                dbc.CardHeader("System Status"),
                                dbc.CardBody([
                                    html.Div(id='status-display')
                                ])
                            ])
                        ], width=4),
                        
                        # Right column - Visualizations
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Position Tracking"),
                                dbc.CardBody([
                                    dcc.Graph(id='position-plot')
                                ])
                            ], className='mb-3'),
                            
                            dbc.Card([
                                dbc.CardHeader("Attitude"),
                                dbc.CardBody([
                                    dcc.Graph(id='attitude-plot')
                                ])
                            ])
                        ], width=8)
                    ]),
                    
                    # Update interval
                    dcc.Interval(
                        id='update-interval',
                        interval=1000,  # Update every second
                        n_intervals=0
                    )
                    
                ], fluid=True, style={'backgroundColor': '#1e1e1e', 'minHeight': '100vh', 'padding': '20px'})
            else:
                # Fallback without Bootstrap
                self.app.layout = html.Div([
                    html.H1("UAV Autopilot Simulator", style={'textAlign': 'center', 'color': 'black'}),
                    
                    html.Div([
                        html.Div([
                            html.H3("Flight Controls"),
                            dcc.Dropdown(
                                id='flight-mode-dropdown',
                                options=[
                                    {'label': 'Manual', 'value': 'manual'},
                                    {'label': 'Stabilize', 'value': 'stabilize'},
                                    {'label': 'Altitude Hold', 'value': 'altitude_hold'},
                                    {'label': 'Position Hold', 'value': 'position_hold'},
                                    {'label': 'Auto', 'value': 'auto'},
                                    {'label': 'Return to Launch', 'value': 'return_to_launch'},
                                    {'label': 'Land', 'value': 'land'},
                                    {'label': 'AI Pilot', 'value': 'ai_pilot'}
                                ],
                                value='auto'
                            ),
                            html.Br(),
                            html.Label("Target Altitude"),
                            dcc.Slider(id='altitude-slider', min=1, max=100, value=10),
                            html.Br(),
                            html.Button("Takeoff", id='takeoff-btn'),
                            html.Button("Land", id='land-btn'),
                            html.Button("RTL", id='rtl-btn')
                        ], style={'width': '30%', 'float': 'left', 'padding': '10px'}),
                        
                        html.Div([
                            dcc.Graph(id='position-plot'),
                            dcc.Graph(id='attitude-plot')
                        ], style={'width': '68%', 'float': 'right'})
                    ]),
                    
                    html.Div(id='status-display', style={'clear': 'both', 'padding': '10px'}),
                    
                    dcc.Interval(
                        id='update-interval',
                        interval=1000,
                        n_intervals=0
                    )
                    
                ], style={'backgroundColor': '#1e1e1e', 'color': 'white', 'fontFamily': 'Arial'})
        
        def setup_callbacks(self):
            """Setup the dashboard callbacks"""
            
            @self.app.callback(
                [Output('status-display', 'children'),
                 Output('position-plot', 'figure'),
                 Output('attitude-plot', 'figure')],
                [Input('update-interval', 'n_intervals')]
            )
            def update_dashboard(n):
                try:
                    # Get telemetry data
                    telemetry = self.fc.get_telemetry()
                    
                    # Update data buffers
                    current_time = time.time()
                    self.position_history.append({
                        'time': current_time,
                        'x': telemetry['position'][0],
                        'y': telemetry['position'][1],
                        'z': telemetry['position'][2]
                    })
                    self.attitude_history.append({
                        'time': current_time,
                        'roll': np.degrees(telemetry['attitude'][0]),
                        'pitch': np.degrees(telemetry['attitude'][1]),
                        'yaw': np.degrees(telemetry['attitude'][2])
                    })
                    
                    # Create status display
                    status_display = html.Div([
                        html.H4("Current Status"),
                        html.P(f"Position: N{telemetry['position'][0]:.1f}m, E{telemetry['position'][1]:.1f}m, Alt{-telemetry['position'][2]:.1f}m"),
                        html.P(f"Velocity: {np.linalg.norm(telemetry['velocity']):.1f} m/s"),
                        html.P(f"Attitude: Roll{np.degrees(telemetry['attitude'][0]):.1f}°, Pitch{np.degrees(telemetry['attitude'][1]):.1f}°, Yaw{np.degrees(telemetry['attitude'][2]):.1f}°"),
                        html.P(f"Flight Mode: {telemetry['flight_mode']}"),
                        html.P(f"Battery: {telemetry['battery']}%"),
                        html.P(f"Waypoint: {telemetry['waypoint_index'] + 1}/{len(self.fc.waypoints) if self.fc.waypoints else 0}"),
                        html.P(f"Mission Complete: {'Yes' if telemetry['mission_complete'] else 'No'}")
                    ])
                    
                    # Create position plot
                    if self.position_history:
                        pos_df = list(self.position_history)
                        times = [p['time'] - self.start_time for p in pos_df]
                        
                        pos_fig = {
                            'data': [
                                {'x': times, 'y': [p['x'] for p in pos_df], 'type': 'line', 'name': 'North', 'line': {'color': '#FF6B6B'}},
                                {'x': times, 'y': [p['y'] for p in pos_df], 'type': 'line', 'name': 'East', 'line': {'color': '#4ECDC4'}},
                                {'x': times, 'y': [-p['z'] for p in pos_df], 'type': 'line', 'name': 'Altitude', 'line': {'color': '#45B7D1'}}
                            ],
                            'layout': {
                                'title': 'Position vs Time',
                                'paper_bgcolor': 'rgba(0,0,0,0)',
                                'plot_bgcolor': 'rgba(0,0,0,0)',
                                'font': {'color': 'white'},
                                'xaxis': {'title': 'Time (s)', 'color': 'white'},
                                'yaxis': {'title': 'Position (m)', 'color': 'white'}
                            }
                        }
                    else:
                        pos_fig = {'data': [], 'layout': {'title': 'Position vs Time'}}
                    
                    # Create attitude plot
                    if self.attitude_history:
                        att_df = list(self.attitude_history)
                        times = [a['time'] - self.start_time for a in att_df]
                        
                        att_fig = {
                            'data': [
                                {'x': times, 'y': [a['roll'] for a in att_df], 'type': 'line', 'name': 'Roll', 'line': {'color': '#FF6B6B'}},
                                {'x': times, 'y': [a['pitch'] for a in att_df], 'type': 'line', 'name': 'Pitch', 'line': {'color': '#4ECDC4'}},
                                {'x': times, 'y': [a['yaw'] for a in att_df], 'type': 'line', 'name': 'Yaw', 'line': {'color': '#45B7D1'}}
                            ],
                            'layout': {
                                'title': 'Attitude vs Time',
                                'paper_bgcolor': 'rgba(0,0,0,0)',
                                'plot_bgcolor': 'rgba(0,0,0,0)',
                                'font': {'color': 'white'},
                                'xaxis': {'title': 'Time (s)', 'color': 'white'},
                                'yaxis': {'title': 'Angle (degrees)', 'color': 'white'}
                            }
                        }
                    else:
                        att_fig = {'data': [], 'layout': {'title': 'Attitude vs Time'}}
                    
                    return status_display, pos_fig, att_fig
                    
                except Exception as e:
                    error_msg = html.Div([
                        html.H4("Error"),
                        html.P(f"Dashboard update error: {str(e)}")
                    ])
                    empty_fig = {'data': [], 'layout': {'title': 'Error'}}
                    return error_msg, empty_fig, empty_fig
            
            # Flight mode callback
            @self.app.callback(
                Output('flight-mode-dropdown', 'value'),
                [Input('takeoff-btn', 'n_clicks'),
                 Input('land-btn', 'n_clicks'),
                 Input('rtl-btn', 'n_clicks'),
                 Input('flight-mode-dropdown', 'value')]
            )
            def handle_flight_controls(takeoff_clicks, land_clicks, rtl_clicks, selected_mode):
                ctx = dash.callback_context
                if not ctx.triggered:
                    return dash.no_update
                
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                
                if button_id == 'takeoff-btn':
                    self.fc.set_flight_mode(FlightMode.ALTITUDE_HOLD)
                    return 'altitude_hold'
                elif button_id == 'land-btn':
                    self.fc.set_flight_mode(FlightMode.LAND)
                    return 'land'
                elif button_id == 'rtl-btn':
                    self.fc.set_flight_mode(FlightMode.RTL)
                    return 'return_to_launch'
                elif button_id == 'flight-mode-dropdown':
                    try:
                        mode = FlightMode(selected_mode)
                        self.fc.set_flight_mode(mode)
                        return selected_mode
                    except ValueError:
                        return dash.no_update
                
                return dash.no_update
            
            # Altitude setpoint callback
            @self.app.callback(
                Output('altitude-slider', 'value'),
                [Input('altitude-slider', 'value')]
            )
            def update_altitude_setpoint(altitude):
                if altitude is not None:
                    self.fc.setpoints['altitude'] = -altitude  # Convert to NED
                return altitude
        
        def run(self, debug: bool = False, port: int = 8050):
            """Start the dashboard server"""
            logger.info(f"🚀 Starting UAV Dashboard on http://localhost:{port}")
            self.app.run(debug=debug, port=port, host='0.0.0.0')