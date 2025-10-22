# simulation_manager.py
"""
Simulation Manager - MERGED with Enhanced 3D Visualization & Simplified Controls
"""
import time
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque

from .utils import logger, check_imports, FlightMode
from .drone import Drone
from .physics import Physics
from .controller import Controller
from .datalogger import DataLogger

# Check for visualization dependencies
IMPORTS = check_imports()
HAS_DASH = IMPORTS.get('dash', False)
HAS_PLOTLY = IMPORTS.get('plotly', False)
HAS_DBC = IMPORTS.get('dbc', False)

if HAS_DASH:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output, State
    if HAS_DBC:
        import dash_bootstrap_components as dbc
    if HAS_PLOTLY:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

class SimulationManager:
    """
    Merged Simulation Manager with enhanced 3D visualization and simplified controls
    """
    
    def __init__(self):
        # Initialize core components
        self.drone = Drone()
        self.physics = Physics()
        self.controller = Controller()
        self.data_logger = DataLogger()
        
        # Simulation state
        self.running = False
        self.simulation_thread = None
        self.dt = 0.01  # 100Hz update rate
        
        # Timing
        self.last_update = time.time()
        self.last_log_time = time.time()
        self.log_interval = 0.1
        
        # Initialize dashboard
        if HAS_DASH and HAS_PLOTLY:
            try:
                self.dashboard = MergedDashboard(self)
                logger.info("Merged dashboard initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize dashboard: {e}")
                self.dashboard = None
        else:
            self.dashboard = None
            if not HAS_DASH:
                logger.warning("Dash not available - dashboard disabled")
            if not HAS_PLOTLY:
                logger.warning("Plotly not available - dashboard disabled")
        
        logger.info("✓ Simulation Manager initialized")
    
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
        while self.running:
            current_time = time.time()
            dt = current_time - self.last_update
            
            if dt >= self.dt:
                try:
                    # Compute control outputs
                    control_output = self.controller.compute_control(self.drone, dt)
                    self.controller.control_output = control_output
                    
                    # Update physics
                    new_state = self.physics.update(self.drone, control_output, dt)
                    self.drone.true_state = new_state
                    
                    # For now, estimated state = true state (no sensor noise)
                    self.drone.estimated_state = new_state
                    
                    # Log data periodically - FIXED: This was missing!
                    if current_time - self.last_log_time >= self.log_interval:
                        self.data_logger.log_data(self.drone, self.controller)
                        self.last_log_time = current_time
                    
                    self.last_update = current_time
                    
                except Exception as e:
                    logger.error(f"Simulation loop error: {e}")
            
            time.sleep(0.001)
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=2.0)
        
        self.data_logger.stop_logging()
        logger.info("Simulation stopped")
    
    def run_with_dashboard(self, port: int = 8050):
        """Run simulation with dashboard"""
        if self.dashboard is None:
            logger.warning("Dashboard not available, running headless")
            self.run_headless()
            return
        
        logger.info("Starting simulation with dashboard...")
        self.start_simulation()
        
        try:
            self.dashboard.run(port=port)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.stop_simulation()
    
    def run_headless(self):
        """Run without dashboard"""
        logger.info("Starting headless simulation...")
        self.start_simulation()
        
        try:
            while self.running:
                time.sleep(1)
                telemetry = self.get_telemetry()
                print(f"Alt: {telemetry['altitude']:.1f}m | "
                      f"Pos: [{telemetry['position'][0]:.1f}, {telemetry['position'][1]:.1f}] | "
                      f"Mode: {telemetry['flight_mode']}")
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.stop_simulation()
    
    def get_telemetry(self) -> Dict:
        """Get telemetry data for dashboard - returns NED position (z negative = below origin)"""
        drone_telemetry = self.drone.get_telemetry()
        controller_status = self.controller.get_status()
        
        # NED position (keep sign convention consistent with dynamics)
        ned_pos = self.drone.estimated_state.position.copy()
        ned_vel = self.drone.estimated_state.velocity.copy()

        current_altitude = -ned_pos[2]  # positive altitude (for displays)
        vertical_velocity = -ned_vel[2]

        return {
            # Keep position as NED vector
            'position': [float(ned_pos[0]), float(ned_pos[1]), float(ned_pos[2])],
            # velocity in NED (vx, vy, vz)
            'velocity': [float(ned_vel[0]), float(ned_vel[1]), float(ned_vel[2])],
            # also expose altitude explicitly so consumers don't guess sign
            'altitude': float(current_altitude),
            'vertical_velocity': float(vertical_velocity),
            'attitude': self.drone.estimated_state.orientation.tolist(),  # roll, pitch, yaw (radians)
            'flight_mode': controller_status['flight_mode'],
            'battery': 85.0,
            'gps_fix': True,
            'waypoint_index': controller_status['waypoint_index'],
            'mission_complete': controller_status['mission_complete'],
            'control_output': controller_status['control_output'],
            'is_launched': controller_status['is_launched'],
            # add estimated_state vector for debugging/tracing if needed
            'estimated_position': self.drone.estimated_state.position.tolist(),
            'true_position': self.drone.true_state.position.tolist()  # true (dynamics) state for cross-check
        }
    
    # Controller proxy methods for dashboard
    def launch(self, target_altitude: float = 2.0):
        """Launch the drone"""
        self.controller.launch(target_altitude)
    
    def add_waypoint(self, north: float, east: float, altitude: float):
        """Add a waypoint to the mission"""
        self.controller.add_waypoint(north, east, altitude)
    
    def clear_waypoints(self):
        """Clear all waypoints"""
        self.controller.clear_waypoints()
    
    def start_mission(self):
        """Start autonomous mission"""
        self.controller.start_mission()
    
    def set_flight_mode(self, mode: FlightMode):
        """Change flight mode"""
        self.controller.set_flight_mode(mode)
    
    def emergency_land(self):
        """Emergency landing"""
        self.controller.emergency_land()
    
    @property
    def waypoints(self):
        """Get waypoints from controller"""
        return self.controller.waypoints


if HAS_DASH and HAS_PLOTLY:
    class MergedDashboard:
        """
        Merged Dashboard with Enhanced 3D Visualization & Simplified Controls
        """
        
        def __init__(self, sim_manager: SimulationManager):
            self.sim = sim_manager
            
            if HAS_DBC:
                self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
            else:
                self.app = dash.Dash(__name__)
            
            self.setup_layout()
            self.setup_callbacks()
            
            # Data buffers (from old version)
            self.position_history = deque(maxlen=200)
            self.attitude_history = deque(maxlen=200)
            self.control_history = deque(maxlen=200)
            self.start_time = time.time()
            
            logger.info("Merged dashboard layout initialized successfully")
        
        def setup_layout(self):
            """Setup merged dashboard layout"""
            if HAS_DBC:
                self.app.layout = dbc.Container([
                    # Header (from old version)
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
                        # Left Panel - Simplified Controls (from new version)
                        dbc.Col([
                            # Launch Control (from new version)
                            dbc.Card([
                                dbc.CardHeader("🚀 Launch Control", 
                                              style={'backgroundColor': '#2a2a2a', 'color': 'white'}),
                                dbc.CardBody([
                                    dbc.Button("🛫 LAUNCH (2m)", id='launch-btn', 
                                              color="success", className='w-100 mb-3', size='lg',
                                              disabled=False),
                                    html.Div(id='launch-status', className='text-center')
                                ])
                            ], className='mb-4'),
                            
                            # Waypoint Input (from new version)
                            dbc.Card([
                                dbc.CardHeader("📍 Waypoint Input (NED)", 
                                              style={'backgroundColor': '#2a2a2a', 'color': 'white'}),
                                dbc.CardBody([
                                    html.Label("North (m)", className='text-light'),
                                    dbc.Input(id='wp-north', type='number', value=10, className='mb-2'),
                                    
                                    html.Label("East (m)", className='text-light'),
                                    dbc.Input(id='wp-east', type='number', value=0, className='mb-2'),
                                    
                                    html.Label("Altitude (m)", className='text-light'),
                                    dbc.Input(id='wp-alt', type='number', value=5, className='mb-3'),
                                    
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Button("➕ Add Waypoint", id='add-wp-btn', 
                                                      color="primary", className='w-100')
                                        ], width=6),
                                        dbc.Col([
                                            dbc.Button("🗑️ Clear All", id='clear-wp-btn', 
                                                      color="danger", className='w-100')
                                        ], width=6)
                                    ]),
                                    
                                    html.Div(id='waypoint-list', className='mt-3')
                                ])
                            ], className='mb-4'),
                            
                            # Flight Modes (from new version)
                            dbc.Card([
                                dbc.CardHeader("✈️ Flight Modes", 
                                              style={'backgroundColor': '#2a2a2a', 'color': 'white'}),
                                dbc.CardBody([
                                    dbc.Button("🎯 Start Mission", id='mission-btn', 
                                              color="info", className='w-100 mb-2', size='lg'),
                                    dbc.Button("🏠 RTL (Return & Land Home)", id='rtl-btn', 
                                              color="warning", className='w-100 mb-2', size='lg'),
                                    dbc.Button("🛬 Land Here", id='land-btn', 
                                              color="secondary", className='w-100 mb-2', size='lg'),
                                    html.Hr(),
                                    dbc.Button("⚠️ EMERGENCY LAND", id='emergency-btn', 
                                              color="danger", className='w-100', size='lg')
                                ])
                            ])
                        ], width=4),
                        
                        # Right Panel - Enhanced Visualizations (from old version)
                        dbc.Col([
                            # Status Display Card (from old version)
                            dbc.Card([
                                dbc.CardHeader("📊 Real-time Telemetry", 
                                              style={'backgroundColor': '#2a2a2a', 'color': 'white'}),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([html.Div(id='status-display')], width=12)
                                    ])
                                ])
                            ], className='mb-4'),
                            
                            # 3D Visualization Card (from old version - enhanced)
                            dbc.Card([
                                dbc.CardHeader("🌍 3D Flight Path", 
                                              style={'backgroundColor': '#2a2a2a', 'color': 'white'}),
                                dbc.CardBody([
                                    dcc.Graph(id='3d-trajectory-plot')
                                ])
                            ], className='mb-4'),
                            
                            # Charts Row (from old version - with proper attitude graphs)
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader("📈 Position vs Time", style={'color': 'white'}),
                                        dbc.CardBody([dcc.Graph(id='position-plot')])
                                    ])
                                ], width=6),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader("🎚️ Attitude vs Time", style={'color': 'white'}),
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
            else:
                # Simple fallback
                self.app.layout = html.Div([
                    html.H1("UAV Control"),
                    html.Button('Launch', id='launch-btn'),
                    html.Div(id='status-display'),
                    dcc.Graph(id='3d-trajectory-plot'),
                    dcc.Graph(id='position-plot'),
                    dcc.Graph(id='attitude-plot'),
                    dcc.Interval(id='update-interval', interval=1000, n_intervals=0)
                ])
        
        def setup_callbacks(self):
            """Setup merged dashboard callbacks"""
            
            @self.app.callback(
                [Output('status-display', 'children'),
                 Output('position-plot', 'figure'),
                 Output('attitude-plot', 'figure'),
                 Output('3d-trajectory-plot', 'figure'),
                 Output('launch-btn', 'disabled')],
                [Input('update-interval', 'n_intervals')]
            )
            def update_dashboard(n):
                try:
                    # Get telemetry data
                    telemetry = self.sim.get_telemetry()
                    current_altitude = telemetry['altitude']
                    
                    # Update data buffers (from old version)
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
                    
                    # Create enhanced status display (from old version)
                    if HAS_DBC:
                        status_display = dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H4("📍 Position", className='text-white'),
                                        html.P(f"North: {telemetry['position'][0]:.1f}m", className='text-white'),
                                        html.P(f"East: {telemetry['position'][1]:.1f}m", className='text-white'),
                                        html.P(f"Altitude: {current_altitude:.1f}m", className='text-white')
                                    ], style={'backgroundColor': '#2a2a2a'})
                                ], style={'border': '1px solid #444'})
                            ], width=4),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H4("🎯 Attitude", className='text-white'),
                                        html.P(f"Roll: {np.degrees(telemetry['attitude'][0]):.1f}°", className='text-white'),
                                        html.P(f"Pitch: {np.degrees(telemetry['attitude'][1]):.1f}°", className='text-white'),
                                        html.P(f"Yaw: {np.degrees(telemetry['attitude'][2]):.1f}°", className='text-white')
                                    ], style={'backgroundColor': '#2a2a2a'})
                                ], style={'border': '1px solid #444'})
                            ], width=4),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H4("⚡ System", className='text-white'),
                                        html.P(f"Mode: {telemetry['flight_mode']}", className='text-white'),
                                        html.P(f"Waypoint: {telemetry['waypoint_index'] + 1}/{(len(self.sim.waypoints) if self.sim.waypoints else 0)}", className='text-white'),
                                        html.P(f"Mission: {'✅ Complete' if telemetry['mission_complete'] else '🟡 Running'}", 
                                              className='text-success' if telemetry['mission_complete'] else 'text-warning')
                                    ], style={'backgroundColor': '#2a2a2a'})
                                ], style={'border': '1px solid #444'})
                            ], width=4)
                        ])
                    else:
                        status_display = html.Div([
                            html.P(f"Position: N{telemetry['position'][0]:.1f} E{telemetry['position'][1]:.1f} Alt{current_altitude:.1f}m"),
                            html.P(f"Attitude: Roll{np.degrees(telemetry['attitude'][0]):.1f}° Pitch{np.degrees(telemetry['attitude'][1]):.1f}° Yaw{np.degrees(telemetry['attitude'][2]):.1f}°"),
                            html.P(f"Mode: {telemetry['flight_mode']} | WP: {telemetry['waypoint_index'] + 1}/{(len(self.sim.waypoints) if self.sim.waypoints else 0)}")
                        ])
                    
                    # Create enhanced plots (from old version)
                    pos_fig = self._create_position_plot()
                    att_fig = self._create_attitude_plot()
                    traj_fig = self._create_3d_trajectory_plot()
                    
                    # Disable launch button if already launched (from new version)
                    launch_disabled = telemetry['is_launched']
                    
                    return status_display, pos_fig, att_fig, traj_fig, launch_disabled
                    
                except Exception as e:
                    logger.error(f"Dashboard update error: {e}")
                    error_msg = html.Div([
                        html.H4("❌ Error", className='text-danger'),
                        html.P(f"Dashboard update error: {str(e)}", className='text-light')
                    ])
                    empty_fig = {'data': [], 'layout': {'title': 'Error'}}
                    return error_msg, empty_fig, empty_fig, empty_fig, False
            
            # Control callbacks (from new version)
            @self.app.callback(
                Output('waypoints-store', 'data'),
                [Input('launch-btn', 'n_clicks'),
                 Input('add-wp-btn', 'n_clicks'),
                 Input('clear-wp-btn', 'n_clicks'),
                 Input('mission-btn', 'n_clicks'),
                 Input('rtl-btn', 'n_clicks'),
                 Input('land-btn', 'n_clicks'),
                 Input('emergency-btn', 'n_clicks')],
                [State('wp-north', 'value'),
                 State('wp-east', 'value'),
                 State('wp-alt', 'value'),
                 State('waypoints-store', 'data')]
            )
            def handle_buttons(launch_n, add_n, clear_n, mission_n, rtl_n, land_n, emergency_n,
                             north, east, alt, waypoints):
                ctx = dash.callback_context
                if not ctx.triggered:
                    return waypoints or []
                
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                waypoints = waypoints or []
                
                try:
                    if button_id == 'launch-btn':
                        self.sim.launch(target_altitude=2.0)
                        logger.info("🚀 Launch button pressed")
                    
                    elif button_id == 'add-wp-btn':
                        if north is not None and east is not None and alt is not None:
                            self.sim.add_waypoint(north, east, alt)
                            waypoints.append({'n': north, 'e': east, 'a': alt})
                    
                    elif button_id == 'clear-wp-btn':
                        self.sim.clear_waypoints()
                        waypoints = []
                    
                    elif button_id == 'mission-btn':
                        self.sim.start_mission()
                    
                    elif button_id == 'rtl-btn':
                        self.sim.set_flight_mode(FlightMode.RTL)
                    
                    elif button_id == 'land-btn':
                        self.sim.set_flight_mode(FlightMode.LAND)
                    
                    elif button_id == 'emergency-btn':
                        self.sim.emergency_land()
                
                except Exception as e:
                    logger.error(f"Button handler error: {e}")
                
                return waypoints
            
            @self.app.callback(
                Output('waypoint-list', 'children'),
                [Input('waypoints-store', 'data')]
            )
            def update_waypoint_list(waypoints):
                if not waypoints:
                    return html.P("No waypoints", className='text-muted')
                
                items = []
                for i, wp in enumerate(waypoints):
                    items.append(
                        html.P(f"WP{i+1}: N{wp['n']} E{wp['e']} Alt{wp['a']}m", 
                              style={'color': '#00ff88', 'margin': '5px'})
                    )
                return html.Div(items)
        
        def _create_position_plot(self):
            """Create enhanced position plot (from old version)"""
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
                    {'x': times, 'y': [-p['z'] for p in pos_data],  # Convert NED to altitude
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
            """Create enhanced attitude plot (from old version) - FIXED with roll, pitch, yaw"""
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
            """Create 3D trajectory plot - FIXED VERSION (from old version)"""
            if not self.position_history:
                return {'data': [], 'layout': {}}
            
            pos_data = list(self.position_history)
            
            # Create 3D scatter plot with proper altitude conversion
            trace = go.Scatter3d(
                x=[p['x'] for p in pos_data],
                y=[p['y'] for p in pos_data],
                z=[-p['z'] for p in pos_data],  # Convert NED to altitude (negative Z = positive altitude)
                mode='lines+markers',
                line=dict(color='#00ff88', width=4),
                marker=dict(
                    size=3, 
                    color=[-p['z'] for p in pos_data],  # Color by altitude
                    colorscale='Viridis', 
                    showscale=True,
                    colorbar=dict(title="Altitude (m)")
                ),
                name='Flight Path'
            )
            
            # Add current position as larger marker
            current_pos = pos_data[-1]
            current_trace = go.Scatter3d(
                x=[current_pos['x']],
                y=[current_pos['y']],
                z=[-current_pos['z']],  # Convert to altitude
                mode='markers',
                marker=dict(size=8, color='red', symbol='diamond'),
                name='Current Position'
            )
            
            # Add waypoints if available
            waypoints_trace = None
            if self.sim.waypoints and len(self.sim.waypoints) > 0:
                waypoints_trace = go.Scatter3d(
                    x=[wp[0] for wp in self.sim.waypoints],
                    y=[wp[1] for wp in self.sim.waypoints],
                    z=[-wp[2] for wp in self.sim.waypoints],  # Convert NED to altitude
                    mode='markers',
                    marker=dict(size=8, color='yellow', symbol='square', line=dict(width=2, color='orange')),
                    name='Waypoints'
                )
            
            data = [trace, current_trace]
            if waypoints_trace:
                data.append(waypoints_trace)
            
            fig = go.Figure(data=data)
            fig.update_layout(
                title={'text': '3D Flight Trajectory', 'font': {'color': 'white'}},
                scene=dict(
                    xaxis_title='North (m)',
                    yaxis_title='East (m)',
                    zaxis_title='Altitude (m)',
                    bgcolor='rgba(20,20,20,1)',
                    xaxis=dict(
                        gridcolor='gray', 
                        showbackground=True,
                        backgroundcolor='rgb(20,20,20)'
                    ),
                    yaxis=dict(
                        gridcolor='gray', 
                        showbackground=True,
                        backgroundcolor='rgb(20,20,20)'
                    ),
                    zaxis=dict(
                        gridcolor='gray', 
                        showbackground=True,
                        backgroundcolor='rgb(20,20,20)'
                    ),
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=True,
                legend=dict(
                    font=dict(color='white'),
                    bgcolor='rgba(0,0,0,0.5)'
                )
            )
            
            return fig
        
        def run(self, debug: bool = False, port: int = 8050):
            """Start the dashboard server"""
            logger.info(f"UAV Dashboard on http://localhost:{port}")
            try:
                self.app.run(debug=debug, port=port, host='0.0.0.0')
            except Exception as e:
                logger.error(f"Failed to start dashboard: {e}")