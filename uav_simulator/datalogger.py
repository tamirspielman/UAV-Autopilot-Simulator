# datalogger.py
"""
Data Logger - logs simulation data to files
"""
import os
import csv
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque

from .drone import Drone
from .controller import Controller
from .utils import logger, normalize_angles

class DataLogger:
    """
    Comprehensive data logging system for drone telemetry and analysis
    Logs all flight data to CSV for post-flight analysis and debugging
    """
    
    def __init__(self, log_directory: str = "flight_logs"):
        self.log_directory = log_directory
        self.current_log_file = None
        self.csv_writer = None
        self.csv_file = None
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_directory, exist_ok=True)
        
        # Field names for CSV logging
        self.fieldnames = [
            'timestamp', 'simulation_time',
            'position_x', 'position_y', 'position_z',
            'velocity_x', 'velocity_y', 'velocity_z',
            'attitude_roll', 'attitude_pitch', 'attitude_yaw',
            'control_throttle', 'control_roll', 'control_pitch', 'control_yaw',
            'estimated_position_x', 'estimated_position_y', 'estimated_position_z',
            'flight_mode', 'battery_level',
            'setpoint_altitude', 'setpoint_position_x', 'setpoint_position_y', 'setpoint_position_z',
            'position_error', 'attitude_error',
            'waypoint_index', 'mission_complete'
        ]
        
        self.start_time = time.time()
        self.logging_enabled = True
        
        # Auto-start logging when initialized
        self.start_new_log()
        
    def start_new_log(self):
        """Start a new log file with timestamp"""
        if not self.logging_enabled:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_log_file = os.path.join(self.log_directory, f"flight_log_{timestamp}.csv")
        
        try:
            self.csv_file = open(self.current_log_file, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
            self.csv_writer.writeheader()
            self.csv_file.flush()
            logger.info(f"Started new flight log: {self.current_log_file}")
        except Exception as e:
            logger.error(f"Failed to create log file: {e}")
            self.logging_enabled = False
    
    def log_data(self, drone: Drone, controller: Controller):
        """Log current drone and controller data"""
        if not self.logging_enabled or self.csv_writer is None or self.csv_file is None:
            return
        
        try:
            current_time = time.time()
            simulation_time = current_time - self.start_time

            # Use proper altitude calculations
            altitude = -drone.true_state.position[2]  # Convert NED to altitude
            vertical_velocity = -drone.true_state.velocity[2]  # Convert NED to climb rate
            estimated_altitude = -drone.estimated_state.position[2]  # Convert NED to altitude

            # Position error (Euclidean) - Use estimated state vs setpoint
            position_error = float(np.linalg.norm(drone.estimated_state.position - controller.setpoints['position']))

            # Attitude error (normalize angles to [-pi, pi])
            try:
                attitude_diff = normalize_angles(
                    drone.estimated_state.orientation - np.array([0, 0, controller.setpoints['yaw']])
                )
                attitude_error = float(np.linalg.norm(attitude_diff))
            except Exception:
                attitude_error = 0.0

            # Build row with CORRECT state assignments
            row = {
                'timestamp': current_time,
                'simulation_time': simulation_time,

                # TRUE STATE (from dynamics)
                'position_x': drone.true_state.position[0],
                'position_y': drone.true_state.position[1], 
                'position_z': altitude,  # This is the TRUE altitude
                'velocity_x': drone.true_state.velocity[0],
                'velocity_y': drone.true_state.velocity[1],
                'velocity_z': vertical_velocity,
                'attitude_roll': drone.true_state.orientation[0],
                'attitude_pitch': drone.true_state.orientation[1],
                'attitude_yaw': drone.true_state.orientation[2],

                # ESTIMATED STATE
                'estimated_position_x': drone.estimated_state.position[0],
                'estimated_position_y': drone.estimated_state.position[1],
                'estimated_position_z': estimated_altitude,  # Convert NED to altitude

                # Control outputs
                'control_throttle': controller.control_output[0],
                'control_roll': controller.control_output[1],
                'control_pitch': controller.control_output[2],
                'control_yaw': controller.control_output[3],

                # System state
                'flight_mode': controller.flight_mode.value,
                'battery_level': 85.0,

                # Setpoints
                'setpoint_altitude': controller.setpoints['altitude'],
                'setpoint_position_x': float(controller.setpoints['position'][0]),
                'setpoint_position_y': float(controller.setpoints['position'][1]),
                'setpoint_position_z': float(controller.setpoints['position'][2]),

                # Errors
                'position_error': position_error,
                'attitude_error': attitude_error,
                
                # Mission status
                'waypoint_index': controller.current_waypoint_index,
                'mission_complete': int(controller.mission_complete)
            }

            self.csv_writer.writerow(row)
            self.csv_file.flush()

        except Exception as e:
            # Only log actual errors, not routine telemetry
            if "Error writing to log file" in str(e):
                logger.error(f"Error writing to log file: {e}")
            
    def stop_logging(self):
        """Stop logging and close the file"""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
            logger.info(f"Closed flight log: {self.current_log_file}")
    
    def analyze_log(self, log_file_path: str) -> Dict[str, Any]:
        """
        Analyze a flight log CSV file and compute performance statistics.
        Handles missing data and NaN values gracefully.
        """
        try:
            df = pd.read_csv(log_file_path)
        except Exception as e:
            logger.error(f"Failed to read log file: {e}")
            return {}

        if df.empty:
            return {}

        # Attitude stats with NaN guards
        if 'attitude_error' in df.columns and not df['attitude_error'].dropna().empty:
            max_att_rad = float(df['attitude_error'].dropna().max())
            avg_att_rad = float(df['attitude_error'].dropna().mean())
        else:
            max_att_rad = avg_att_rad = 0.0

        analysis = {
            'file_path': log_file_path,
            'duration_seconds': float(df['simulation_time'].max()) if 'simulation_time' in df.columns else 0.0,
            'total_samples': len(df),
            'average_update_rate': (
                (len(df) / df['simulation_time'].max())
                if ('simulation_time' in df.columns and df['simulation_time'].max() > 0)
                else 0.0
            ),

            # Position statistics
            'max_position_error': float(df['position_error'].max()) if 'position_error' in df.columns else 0.0,
            'avg_position_error': float(df['position_error'].mean()) if 'position_error' in df.columns else 0.0,
            'final_position_error': float(df['position_error'].iloc[-1]) if ('position_error' in df.columns and len(df) > 0) else 0.0,

            # Attitude statistics (radians → degrees)
            'max_attitude_error_deg': float(np.degrees(max_att_rad)),
            'avg_attitude_error_deg': float(np.degrees(avg_att_rad)),

            # Control performance
            'max_throttle': float(df['control_throttle'].max()) if 'control_throttle' in df.columns else 0.0,
            'min_throttle': float(df['control_throttle'].min()) if 'control_throttle' in df.columns else 0.0,
            'max_roll_command': float(df['control_roll'].max()) if 'control_roll' in df.columns else 0.0,
            'max_pitch_command': float(df['control_pitch'].max()) if 'control_pitch' in df.columns else 0.0,

            # Mission completion
            'mission_completed': bool(df['mission_complete'].iloc[-1]) if ('mission_complete' in df.columns and len(df) > 0) else False,
            'waypoints_reached': int(df['waypoint_index'].max()) if ('waypoint_index' in df.columns and len(df) > 0) else 0
        }

        return analysis
    
    def generate_report(self, log_file_path: Optional[str] = None) -> str:
        """Generate a comprehensive flight report"""
        if log_file_path is None:
            return "No log file specified"
            
        analysis = self.analyze_log(log_file_path)
        if not analysis:
            return "No data available for report"
            
        report = f"""
FLIGHT ANALYSIS REPORT
======================
File: {analysis['file_path']}
Duration: {analysis['duration_seconds']:.1f} seconds
Samples: {analysis['total_samples']}
Update Rate: {analysis['average_update_rate']:.1f} Hz

POSITION PERFORMANCE
--------------------
Maximum Position Error: {analysis['max_position_error']:.2f} m
Average Position Error: {analysis['avg_position_error']:.2f} m
Final Position Error: {analysis['final_position_error']:.2f} m

ATTITUDE PERFORMANCE
--------------------
Maximum Attitude Error: {analysis['max_attitude_error_deg']:.1f}°
Average Attitude Error: {analysis['avg_attitude_error_deg']:.1f}°

CONTROL PERFORMANCE
-------------------
Throttle Range: {analysis['min_throttle']:.2f} - {analysis['max_throttle']:.2f}
Max Roll Command: {analysis['max_roll_command']:.2f}
Max Pitch Command: {analysis['max_pitch_command']:.2f}

MISSION STATUS
--------------
Mission Completed: {'Yes' if analysis['mission_completed'] else 'No'}
Waypoints Reached: {analysis['waypoints_reached']}
"""
        return report

    def get_recent_logs(self, count: int = 5) -> List[str]:
        """Get list of recent log files"""
        try:
            log_files = []
            for file in os.listdir(self.log_directory):
                if file.startswith("flight_log_") and file.endswith(".csv"):
                    file_path = os.path.join(self.log_directory, file)
                    mod_time = os.path.getmtime(file_path)
                    log_files.append((file_path, mod_time))
            
            # Sort by modification time (newest first)
            log_files.sort(key=lambda x: x[1], reverse=True)
            return [file[0] for file in log_files[:count]]
            
        except Exception as e:
            logger.error(f"Error getting recent logs: {e}")
            return []