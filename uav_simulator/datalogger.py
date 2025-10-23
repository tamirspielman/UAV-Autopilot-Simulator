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
    def __init__(self, log_directory: str = "flight_logs"):
        self.log_directory = log_directory
        self.current_log_file = None
        self.csv_writer = None
        self.csv_file = None
        os.makedirs(self.log_directory, exist_ok=True)
        self.fieldnames = [
            'timestamp', 'simulation_time',
            'position_x', 'position_y', 'position_z',
            'velocity_x', 'velocity_y', 'velocity_z',
            'attitude_roll', 'attitude_pitch', 'attitude_yaw',
            'control_throttle', 'control_roll', 'control_pitch', 'control_yaw',
            'flight_mode'
        ]
        self.start_time = time.time()
        self.logging_enabled = True
        self.start_new_log()  
    def start_new_log(self):
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
        if not self.logging_enabled or self.csv_writer is None or self.csv_file is None:
            return
        try:
            current_time = time.time()
            simulation_time = current_time - self.start_time
            altitude = -drone.true_state.position[2]  
            vertical_velocity = -drone.true_state.velocity[2]  
            row = {
                'timestamp': current_time,
                'simulation_time': simulation_time,
                'position_x': drone.true_state.position[0],
                'position_y': drone.true_state.position[1], 
                'position_z': altitude,  
                'velocity_x': drone.true_state.velocity[0],
                'velocity_y': drone.true_state.velocity[1],
                'velocity_z': vertical_velocity,
                'attitude_roll': drone.true_state.orientation[0],
                'attitude_pitch': drone.true_state.orientation[1],
                'attitude_yaw': drone.true_state.orientation[2],
                'control_throttle': controller.control_output[0],
                'control_roll': controller.control_output[1],
                'control_pitch': controller.control_output[2],
                'control_yaw': controller.control_output[3],
                'flight_mode': controller.flight_mode.value
            }
            self.csv_writer.writerow(row)
            self.csv_file.flush()
        except Exception as e:
            if "Error writing to log file" in str(e):
                logger.error(f"Error writing to log file: {e}")
    def stop_logging(self):
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
            logger.info(f"Closed flight log: {self.current_log_file}")
    def analyze_log(self, log_file_path: str) -> Dict[str, Any]:
        try:
            df = pd.read_csv(log_file_path)
        except Exception as e:
            logger.error(f"Failed to read log file: {e}")
            return {}
        if df.empty:
            return {}
        analysis = {
            'file_path': log_file_path,
            'duration_seconds': float(df['simulation_time'].max()) if 'simulation_time' in df.columns else 0.0,
            'total_samples': len(df),
            'average_update_rate': (
                (len(df) / df['simulation_time'].max())
                if ('simulation_time' in df.columns and df['simulation_time'].max() > 0)
                else 0.0
            ),
            'max_altitude': float(df['position_z'].max()) if 'position_z' in df.columns else 0.0,
            'min_altitude': float(df['position_z'].min()) if 'position_z' in df.columns else 0.0,
            'final_altitude': float(df['position_z'].iloc[-1]) if ('position_z' in df.columns and len(df) > 0) else 0.0,
            'max_roll_deg': float(np.degrees(df['attitude_roll'].max())) if 'attitude_roll' in df.columns else 0.0,
            'max_pitch_deg': float(np.degrees(df['attitude_pitch'].max())) if 'attitude_pitch' in df.columns else 0.0,
            'avg_roll_deg': float(np.degrees(df['attitude_roll'].mean())) if 'attitude_roll' in df.columns else 0.0,
            'avg_pitch_deg': float(np.degrees(df['attitude_pitch'].mean())) if 'attitude_pitch' in df.columns else 0.0,
            'max_throttle': float(df['control_throttle'].max()) if 'control_throttle' in df.columns else 0.0,
            'min_throttle': float(df['control_throttle'].min()) if 'control_throttle' in df.columns else 0.0,
            'max_roll_command': float(df['control_roll'].max()) if 'control_roll' in df.columns else 0.0,
            'max_pitch_command': float(df['control_pitch'].max()) if 'control_pitch' in df.columns else 0.0,
            'flight_modes_used': df['flight_mode'].unique().tolist() if 'flight_mode' in df.columns else []
        }
        return analysis
    
    def generate_report(self, log_file_path: Optional[str] = None) -> str:
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

ALTITUDE PERFORMANCE
--------------------
Maximum Altitude: {analysis['max_altitude']:.1f} m
Minimum Altitude: {analysis['min_altitude']:.1f} m
Final Altitude: {analysis['final_altitude']:.1f} m

ATTITUDE PERFORMANCE
--------------------
Maximum Roll: {analysis['max_roll_deg']:.1f}°
Maximum Pitch: {analysis['max_pitch_deg']:.1f}°
Average Roll: {analysis['avg_roll_deg']:.1f}°
Average Pitch: {analysis['avg_pitch_deg']:.1f}°

CONTROL PERFORMANCE
-------------------
Throttle Range: {analysis['min_throttle']:.2f} - {analysis['max_throttle']:.2f}
Max Roll Command: {analysis['max_roll_command']:.2f}
Max Pitch Command: {analysis['max_pitch_command']:.2f}

FLIGHT MODES
------------
Modes Used: {', '.join(analysis['flight_modes_used'])}
"""
        return report
    def get_recent_logs(self, count: int = 5) -> List[str]:
        try:
            log_files = []
            for file in os.listdir(self.log_directory):
                if file.startswith("flight_log_") and file.endswith(".csv"):
                    file_path = os.path.join(self.log_directory, file)
                    mod_time = os.path.getmtime(file_path)
                    log_files.append((file_path, mod_time))
            log_files.sort(key=lambda x: x[1], reverse=True)
            return [file[0] for file in log_files[:count]]
        except Exception as e:
            logger.error(f"Error getting recent logs: {e}")
            return []