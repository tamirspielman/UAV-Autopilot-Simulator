"""
Simulation Manager and Enhanced Dashboard Interface
"""
import time
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple
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
    from dash import dcc, html, Input, Output, State
    if HAS_DBC:
        import dash_bootstrap_components as dbc
    if HAS_PLOTLY:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

class SimulationManager:
    """
    Manages the complete simulation with enhanced mission capabilities
    """
    
    def __init__(self):
        self.flight_controller = FlightController()
        if HAS_DASH:
            self.dashboard = EnhancedUAVDashboard(self.flight_controller)
        else:
            self.dashboard = None
        
        # Simulation state
        self.running = False
        self.simulation_thread = None
        self.real_time_factor = 1.0
        
        # Performance monitoring
        self.update_times = deque(maxlen=100)
        
        # Mission waypoints
        self.default_waypoints = [
            np.array([0, 0, -10]),      # Start at 10m altitude (Down = -10)
            np.array([15, 10, -20]),    # First waypoint at 20m altitude  
            np.array([25, -5, -30]),    # Second waypoint at 30m altitude
            np.array([10, -15, -25]),   # Third waypoint at 25m altitude
            np.array([0, 0, -10])       # Return to start at 10m altitude
    ]
        
        # Initialize mission
        self._initialize_mission()
    
    def _initialize_mission(self):
        """Initialize with default mission"""
        self.flight_controller.set_waypoints(self.default_waypoints)
        self.flight_controller.set_flight_mode(FlightMode.STABILIZE)
    
    def add_waypoint(self, x: float, y: float, z: float):
        """Add a new waypoint to the mission"""
        new_waypoint = np.array([x, y, -abs(z)])  # Ensure negative for NED
        self.default_waypoints.insert(-1, new_waypoint)  # Insert before final return
        self.flight_controller.set_waypoints(self.default_waypoints)
        logger.info(f"Added waypoint: [{x}, {y}, {z}]")
    
    def clear_waypoints(self):
        """Clear all waypoints except home"""
        self.default_waypoints = [self.default_waypoints[0], self.default_waypoints[-1]]
        self.flight_controller.set_waypoints(self.default_waypoints)
        logger.info("Cleared waypoints")
    
    def set_home_position(self, x: float, y: float, z: float):
        """Set new home position"""
        self.default_waypoints[0] = np.array([x, y, -abs(z)])
        self.default_waypoints[-1] = np.array([x, y, -abs(z)])
        self.flight_controller.set_waypoints(self.default_waypoints)
        logger.info(f"Home position set to: [{x}, {y}, {z}]")
    
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
        """Run simulation with enhanced web dashboard"""
        if not HAS_DASH or self.dashboard is None:
            print("Dash not available. Cannot run dashboard.")
            print("To enable the dashboard, install: pip install dash plotly dash-bootstrap-components")
            return
            
        logger.info("Starting simulation with enhanced dashboard...")
        
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

# Enhanced UAV Dashboard with modern UI and mission planning
if HAS_DASH:
    class EnhancedUAVDashboard:
        """
        Modern UAV Dashboard with Enhanced Features
        """
        
        def __init__(self, flight_controller: FlightController):
            self.fc = flight_controller
            
            # Create the app with modern theme
            if HAS_DBC:
                self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
            else:
                self.app = dash.Dash(__name__)
            
            # Set up the enhanced layout
            self.setup_enhanced_layout()
            
            # Set up callbacks
            self.setup_enhanced_callbacks()
            
            # Data buffers
            self.position_history = deque(maxlen=200)
            self.attitude_history = deque(maxlen=200)
            self.control_history = deque(maxlen=200)
            self.start_time = time.time()
            
            # Mission waypoints for display
            self.mission_waypoints = []
        
        def setup_enhanced_layout(self):
            """Setup modern, professional layout"""
            if HAS_DBC:
                self.app.layout = dbc.Container([
                    # Header
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H1("🚁 Advanced UAV Autopilot Simulator", 
                                       className='text-center mb-2',
                                       style={'color': '#00ff88', 'fontWeight': 'bold'}),
                                html.P("Professional Flight Control System", 
                                      className='text-center mb-4',
                                      style={'color': '#cccccc', 'fontSize': '18px'})
                            ])
                        ], width=12)
                    ], className='mb-4'),
                    
                    # Main Content
                    dbc.Row([
                        # Left Panel - Controls and Mission Planning
                        dbc.Col([
                            # Flight Mode Card
                            dbc.Card([
                                dbc.CardHeader("🎯 Flight Mode Control", 
                                              style={'backgroundColor': '#2a2a2a', 'color': 'white'}),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Current Flight Mode", className='text-light mb-2'),
                                            dcc.Dropdown(
                                                id='flight-mode-dropdown',
                                                options=[
                                                    {'label': '🛸 Stabilize', 'value': 'stabilize'},
                                                    {'label': '📊 Altitude Hold', 'value': 'altitude_hold'},
                                                    {'label': '🎯 Position Hold', 'value': 'position_hold'},
                                                    {'label': '🚀 Auto Mission', 'value': 'auto'},
                                                    {'label': '🏠 Return to Launch', 'value': 'return_to_launch'},
                                                    {'label': '🛬 Land', 'value': 'land'},
                                                    {'label': '🤖 AI Pilot', 'value': 'ai_pilot'}
                                                ],
                                                value='stabilize',
                                                className='mb-3'
                                            )
                                        ], width=12)
                                    ]),
                                    
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Button("🛫 Takeoff to 10m", id='takeoff-btn', 
                                                      color="success", className='w-100 mb-2', size='lg'),
                                        ], width=6),
                                        dbc.Col([
                                            dbc.Button("🛬 Emergency Land", id='land-btn', 
                                                      color="warning", className='w-100 mb-2', size='lg'),
                                        ], width=6)
                                    ]),
                                    
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Button("🏠 Return to Home", id='rtl-btn', 
                                                      color="info", className='w-100 mb-2', size='lg'),
                                        ], width=6),
                                        dbc.Col([
                                            dbc.Button("🔄 Reset Mission", id='reset-btn', 
                                                      color="secondary", className='w-100 mb-2', size='lg'),
                                        ], width=6)
                                    ])
                                ])
                            ], className='mb-4'),
                            
                            # Altitude Control Card
                            dbc.Card([
                                dbc.CardHeader("📈 Altitude Control", 
                                              style={'backgroundColor': '#2a2a2a', 'color': 'white'}),
                                dbc.CardBody([
                                    html.Label("Target Altitude (meters)", className='text-light mb-3'),
                                    dcc.Slider(
                                        id='altitude-slider',
                                        min=1, max=100, step=1, value=10,
                                        marks={i: f'{i}m' for i in range(0, 101, 20)},
                                        className='mb-3'
                                    ),
                                    html.Div(id='altitude-display', className='text-center h5 text-warning')
                                ])
                            ], className='mb-4'),
                            
                            # Mission Planning Card
                            dbc.Card([
                                dbc.CardHeader("🗺️ Mission Planning", 
                                              style={'backgroundColor': '#2a2a2a', 'color': 'white'}),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([html.Label("X (North)", className='text-light')], width=4),
                                        dbc.Col([html.Label("Y (East)", className='text-light')], width=4),
                                        dbc.Col([html.Label("Z (Alt)", className='text-light')], width=4),
                                    ]),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Input(id='waypoint-x', type='number', value=10, 
                                                     className='mb-2')
                                        ], width=4),
                                        dbc.Col([
                                            dbc.Input(id='waypoint-y', type='number', value=10, 
                                                     className='mb-2')
                                        ], width=4),
                                        dbc.Col([
                                            dbc.Input(id='waypoint-z', type='number', value=30, 
                                                     className='mb-2')
                                        ], width=4),
                                    ]),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Button("➕ Add Waypoint", id='add-waypoint-btn', 
                                                      color="primary", className='w-100 mb-2')
                                        ], width=6),
                                        dbc.Col([
                                            dbc.Button("🗑️ Clear All", id='clear-waypoints-btn', 
                                                      color="danger", className='w-100 mb-2')
                                        ], width=6)
                                    ]),
                                    html.Div(id='waypoints-list', className='mt-3')
                                ])
                            ])
                        ], width=4),
                        
                        # Right Panel - Visualizations and Status
                        dbc.Col([
                            # Status Display Card
                            dbc.Card([
                                dbc.CardHeader("📊 Real-time Telemetry", 
                                              style={'backgroundColor': '#2a2a2a', 'color': 'white'}),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([html.Div(id='status-display')], width=12)
                                    ])
                                ])
                            ], className='mb-4'),
                            
                            # 3D Visualization Card
                            dbc.Card([
                                dbc.CardHeader("🌍 3D Flight Path", 
                                              style={'backgroundColor': '#2a2a2a', 'color': 'white'}),
                                dbc.CardBody([
                                    dcc.Graph(id='3d-trajectory-plot')
                                ])
                            ], className='mb-4'),
                            
                            # Charts Row
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader("📈 Position Tracking"),
                                        dbc.CardBody([dcc.Graph(id='position-plot')])
                                    ])
                                ], width=6),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader("🎚️ Attitude"),
                                        dbc.CardBody([dcc.Graph(id='attitude-plot')])
                                    ])
                                ], width=6)
                            ])
                        ], width=8)
                    ]),
                    
                    # Update interval
                    dcc.Interval(
                        id='update-interval',
                        interval=500,  # Update every 500ms for smoother display
                        n_intervals=0
                    ),
                    
                    # Storage for waypoints
                    dcc.Store(id='waypoints-store', data=[])
                    
                ], fluid=True, style={'backgroundColor': '#1a1a1a', 'minHeight': '100vh', 'padding': '20px'})
        
        def setup_enhanced_callbacks(self):
            """Setup enhanced dashboard callbacks"""
            
            @self.app.callback(
                [Output('status-display', 'children'),
                 Output('position-plot', 'figure'),
                 Output('attitude-plot', 'figure'),
                 Output('3d-trajectory-plot', 'figure'),
                 Output('altitude-display', 'children')],
                [Input('update-interval', 'n_intervals')]
            )
            def update_dashboard(n):
                try:
                    # Get telemetry data
                    telemetry = self.fc.get_telemetry()
                    current_altitude = -telemetry['position'][2]
                    
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
                    
                    # Create enhanced status display
                    status_display = dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("📍 Position", className='text-info'),
                                    html.P(f"North: {telemetry['position'][0]:.1f}m", className='text-light'),
                                    html.P(f"East: {telemetry['position'][1]:.1f}m", className='text-light'),
                                    html.P(f"Altitude: {current_altitude:.1f}m", className='text-light')
                                ])
                            ])
                        ], width=4),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("🎯 Attitude", className='text-info'),
                                    html.P(f"Roll: {np.degrees(telemetry['attitude'][0]):.1f}°", className='text-light'),
                                    html.P(f"Pitch: {np.degrees(telemetry['attitude'][1]):.1f}°", className='text-light'),
                                    html.P(f"Yaw: {np.degrees(telemetry['attitude'][2]):.1f}°", className='text-light')
                                ])
                            ])
                        ], width=4),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("⚡ System", className='text-info'),
                                    html.P(f"Mode: {telemetry['flight_mode']}", className='text-light'),
                                    html.P(f"Battery: {telemetry['battery']}%", className='text-light'),
                                    html.P(f"Waypoint: {telemetry['waypoint_index'] + 1}/{(len(self.fc.waypoints) if self.fc.waypoints else 0)}", className='text-light'),
                                    html.P(f"Mission: {'✅ Complete' if telemetry['mission_complete'] else '🟡 Running'}", 
                                          className='text-success' if telemetry['mission_complete'] else 'text-warning')
                                ])
                            ])
                        ], width=4)
                    ])
                    
                    # Create position plot
                    pos_fig = self._create_position_plot()
                    
                    # Create attitude plot  
                    att_fig = self._create_attitude_plot()
                    
                    # Create 3D trajectory plot
                    traj_fig = self._create_3d_trajectory_plot()
                    
                    # Altitude display
                    alt_display = f"Current: {current_altitude:.1f}m | Target: {-self.fc.setpoints['position'][2]:.1f}m"
                    
                    return status_display, pos_fig, att_fig, traj_fig, alt_display
                    
                except Exception as e:
                    error_msg = html.Div([
                        html.H4("❌ Error", className='text-danger'),
                        html.P(f"Dashboard update error: {str(e)}", className='text-light')
                    ])
                    empty_fig = {'data': [], 'layout': {'title': 'Error'}}
                    return error_msg, empty_fig, empty_fig, empty_fig, "Error"
            
            # Flight mode and control callbacks
            @self.app.callback(
                [Output('flight-mode-dropdown', 'value'),
                 Output('waypoints-store', 'data')],
                [Input('takeoff-btn', 'n_clicks'),
                 Input('land-btn', 'n_clicks'),
                 Input('rtl-btn', 'n_clicks'),
                 Input('reset-btn', 'n_clicks'),
                 Input('flight-mode-dropdown', 'value'),
                 Input('add-waypoint-btn', 'n_clicks'),
                 Input('clear-waypoints-btn', 'n_clicks')],
                [State('waypoint-x', 'value'),
                 State('waypoint-y', 'value'), 
                 State('waypoint-z', 'value'),
                 State('waypoints-store', 'data')]
            )
            def handle_controls(takeoff_clicks, land_clicks, rtl_clicks, reset_clicks, 
                              selected_mode, add_clicks, clear_clicks, x, y, z, waypoints_data):
                ctx = dash.callback_context
                if not ctx.triggered:
                    return dash.no_update, waypoints_data
                
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                waypoints = waypoints_data or []
                
                if button_id == 'takeoff-btn':
                    self.fc.set_flight_mode(FlightMode.ALTITUDE_HOLD)
                    self.fc.setpoints['altitude'] = 10.0
                    return 'altitude_hold', waypoints
                    
                elif button_id == 'land-btn':
                    self.fc.set_flight_mode(FlightMode.LAND)
                    return 'land', waypoints
                    
                elif button_id == 'rtl-btn':
                    self.fc.set_flight_mode(FlightMode.RTL)
                    return 'return_to_launch', waypoints
                    
                elif button_id == 'reset-btn':
                    self.fc.set_flight_mode(FlightMode.STABILIZE)
                    self.fc.mission_complete = False
                    self.fc.current_waypoint_index = 0
                    return 'stabilize', waypoints
                    
                elif button_id == 'flight-mode-dropdown':
                    try:
                        mode = FlightMode(selected_mode)
                        self.fc.set_flight_mode(mode)
                        return selected_mode, waypoints
                    except ValueError:
                        return dash.no_update, waypoints
                        
                elif button_id == 'add-waypoint-btn':
                    if x is not None and y is not None and z is not None:
                        # Convert to NED coordinates: z becomes negative for altitude
                        ned_z = -abs(z)  # Ensure negative for proper altitude
                        new_wp = {'x': x, 'y': y, 'z': z, 'ned_z': ned_z, 'id': len(waypoints)}
                        waypoints.append(new_wp)
                        # Convert to numpy array in NED coordinates and update flight controller
                        wp_array = [np.array([wp['x'], wp['y'], wp['ned_z']]) for wp in waypoints]
                        self.fc.set_waypoints(wp_array)
                        return dash.no_update, waypoints               
                elif button_id == 'clear-waypoints-btn':
                    waypoints = []
                    self.fc.set_waypoints([])
                    return dash.no_update, waypoints
                
                return dash.no_update, waypoints
            
            # Waypoints list display
            @self.app.callback(
                Output('waypoints-list', 'children'),
                [Input('waypoints-store', 'data')]
            )
            def update_waypoints_list(waypoints):
                if not waypoints:
                    return html.P("No waypoints set", className='text-muted')
                
                waypoint_items = []
                for i, wp in enumerate(waypoints):
                    waypoint_items.append(
                        dbc.Card([
                            dbc.CardBody([
                                html.P(f"Waypoint {i+1}: N{wp['x']}, E{wp['y']}, Alt{wp['z']}m", 
                                      className='text-light mb-0')
                            ])
                        ], className='mb-2')
                    )
                
                return html.Div(waypoint_items)
            
            # Altitude setpoint callback
            @self.app.callback(
                Output('altitude-slider', 'value'),
                [Input('altitude-slider', 'value')]
            )
            def update_altitude_setpoint(altitude):
                if altitude is not None:
                    self.fc.setpoints['altitude'] = altitude
                return altitude
        
        def _create_position_plot(self):
            """Create enhanced position plot"""
            if not self.position_history:
                return {'data': [], 'layout': {}}
            
            pos_data = list(self.position_history)
            times = [p['time'] - self.start_time for p in pos_data]
            
            fig = {
                'data': [
                    {'x': times, 'y': [p['x'] for p in pos_data], 
                     'type': 'line', 'name': 'North', 'line': {'color': '#FF6B6B', 'width': 3}},
                    {'x': times, 'y': [p['y'] for p in pos_data], 
                     'type': 'line', 'name': 'East', 'line': {'color': '#4ECDC4', 'width': 3}},
                    {'x': times, 'y': [-p['z'] for p in pos_data], 
                     'type': 'line', 'name': 'Altitude', 'line': {'color': '#45B7D1', 'width': 3}}
                ],
                'layout': {
                    'title': {'text': 'Position vs Time', 'font': {'color': 'white'}},
                    'paper_bgcolor': 'rgba(0,0,0,0)',
                    'plot_bgcolor': 'rgba(30,30,30,0.8)',
                    'font': {'color': 'white'},
                    'xaxis': {'title': 'Time (s)', 'color': 'white', 'gridcolor': '#444'},
                    'yaxis': {'title': 'Position (m)', 'color': 'white', 'gridcolor': '#444'},
                    'legend': {'font': {'color': 'white'}}
                }
            }
            return fig
        
        def _create_attitude_plot(self):
            """Create enhanced attitude plot"""
            if not self.attitude_history:
                return {'data': [], 'layout': {}}
            
            att_data = list(self.attitude_history)
            times = [a['time'] - self.start_time for a in att_data]
            
            fig = {
                'data': [
                    {'x': times, 'y': [a['roll'] for a in att_data], 
                     'type': 'line', 'name': 'Roll', 'line': {'color': '#FF6B6B', 'width': 3}},
                    {'x': times, 'y': [a['pitch'] for a in att_data], 
                     'type': 'line', 'name': 'Pitch', 'line': {'color': '#4ECDC4', 'width': 3}},
                    {'x': times, 'y': [a['yaw'] for a in att_data], 
                     'type': 'line', 'name': 'Yaw', 'line': {'color': '#45B7D1', 'width': 3}}
                ],
                'layout': {
                    'title': {'text': 'Attitude vs Time', 'font': {'color': 'white'}},
                    'paper_bgcolor': 'rgba(0,0,0,0)',
                    'plot_bgcolor': 'rgba(30,30,30,0.8)',
                    'font': {'color': 'white'},
                    'xaxis': {'title': 'Time (s)', 'color': 'white', 'gridcolor': '#444'},
                    'yaxis': {'title': 'Angle (degrees)', 'color': 'white', 'gridcolor': '#444'},
                    'legend': {'font': {'color': 'white'}}
                }
            }
            return fig
        
        def _create_3d_trajectory_plot(self):
            """Create 3D trajectory plot"""
            if not self.position_history:
                return {'data': [], 'layout': {}}
            
            pos_data = list(self.position_history)
            
            # Create 3D scatter plot
            trace = go.Scatter3d(
                x=[p['x'] for p in pos_data],
                y=[p['y'] for p in pos_data],
                z=[-p['z'] for p in pos_data],  # Convert to altitude
                mode='lines+markers',
                line=dict(color='#00ff88', width=4),
                marker=dict(size=3, color=[-p['z'] for p in pos_data], 
                          colorscale='Viridis', showscale=True)
            )
            
            # Add current position as larger marker
            current_pos = pos_data[-1]
            current_trace = go.Scatter3d(
                x=[current_pos['x']],
                y=[current_pos['y']],
                z=[-current_pos['z']],
                mode='markers',
                marker=dict(size=8, color='red'),
                name='Current Position'
            )
            
            fig = go.Figure(data=[trace, current_trace])
            fig.update_layout(
                title={'text': '3D Flight Trajectory', 'font': {'color': 'white'}},
                scene=dict(
                    xaxis_title='North (m)',
                    yaxis_title='East (m)',
                    zaxis_title='Altitude (m)',
                    bgcolor='rgba(20,20,20,1)',
                    gridcolor='gray',
                    xaxis=dict(gridcolor='gray'),
                    yaxis=dict(gridcolor='gray'),
                    zaxis=dict(gridcolor='gray'),
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            return fig
        
        def run(self, debug: bool = False, port: int = 8050):
            """Start the enhanced dashboard server"""
            logger.info(f"🚀 Starting Enhanced UAV Dashboard on http://localhost:{port}")
            self.app.run(debug=debug, port=port, host='0.0.0.0')