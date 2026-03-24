import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def compute_global_timestamps(csv_path):
    """
    Applies offline linear regression to map hardware sensor timestamps 
    to the host's time-of-arrival clock, eliminating USB jitter.
    
    Arguments:
    - csv_path (str): String path to the camera's timestamps.csv file.
    
    Returns:
    - pd.DataFrame: A pandas DataFrame containing the original data plus a new 
                    'global_timestamp_ms' column.
    """
    df = pd.read_csv(csv_path)
    
    # Extract the X and Y arrays for the regression
    sensor_ts = df['sensor_timestamp_ms'].values
    arrival_ts = df['time_of_arrival_ms'].values
    
    # numpy.polyfit calculates the 'a' (slope) and 'b' (intercept) 
    # coefficients from Equation 1 in the paper.
    slope, intercept = np.polyfit(sensor_ts, arrival_ts, 1)
    
    # Apply the regression to generate the smooth global timeline
    df['global_timestamp_ms'] = (sensor_ts * slope) + intercept
    
    return df


def plot_synchronization_dashboard(dataframes, labels):
    """
    Generates a 3-panel matplotlib dashboard replicating Figures 3 and 8 
    from the reference paper to visually verify synchronization.
    
    Arguments:
    - dataframes (list): List of pandas DataFrames returned by compute_global_timestamps.
    - labels (list): List of string labels for the cameras (e.g., ['Cam 0', 'Cam 1']).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    ax_sensor, ax_arrival, ax_global = axes
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', '^', 'x']

    for i, (df, label) in enumerate(zip(dataframes, labels)):
        frames = df['frame_index'].values
        
        # To make the graphs readable like the paper, we normalize the timestamps 
        # by subtracting the very first timestamp so the graphs start near 0.
        norm_sensor = df['sensor_timestamp_ms'].values - df['sensor_timestamp_ms'].values[0]
        norm_arrival = df['time_of_arrival_ms'].values - df['time_of_arrival_ms'].values[0]
        norm_global = df['global_timestamp_ms'].values - df['global_timestamp_ms'].values[0]
        
        # Panel 1: Sensor Timestamps (Figure 3a replica)
        # Expected: Smooth lines, but starting at different offsets due to hardware clock differences.
        ax_sensor.plot(frames, norm_sensor, marker=markers[i], color=colors[i], 
                       linestyle='-', linewidth=1, markersize=4, label=label, alpha=0.7)
        
        # Panel 2: Time-of-Arrival (Figure 3b replica)
        # Expected: Lines are overlapping, but you should see jagged stutters from USB jitter.
        ax_arrival.plot(frames, norm_arrival, marker=markers[i], color=colors[i], 
                        linestyle='-', linewidth=1, markersize=4, label=label, alpha=0.7)
        
        # Panel 3: Global Timestamps (Figure 8b replica)
        # Expected: Lines are perfectly smooth AND perfectly overlapping.
        ax_global.plot(frames, norm_global, marker=markers[i], color=colors[i], 
                       linestyle='-', linewidth=1, markersize=4, label=label, alpha=0.7)

    # Formatting Panel 1
    ax_sensor.set_title('Sensor Timestamp (Hardware Clock)')
    ax_sensor.set_xlabel('Frame')
    ax_sensor.set_ylabel('Normalized Time (ms)')
    ax_sensor.legend()
    ax_sensor.grid(True, linestyle='--', alpha=0.6)

    # Formatting Panel 2
    ax_arrival.set_title('Time-of-Arrival (USB Jitter)')
    ax_arrival.set_xlabel('Frame')
    ax_arrival.set_ylabel('Normalized Time (ms)')
    ax_arrival.legend()
    ax_arrival.grid(True, linestyle='--', alpha=0.6)

    # Formatting Panel 3
    ax_global.set_title('Global Timestamp (Regressed / Aligned)')
    ax_global.set_xlabel('Frame')
    ax_global.set_ylabel('Normalized Time (ms)')
    ax_global.legend()
    ax_global.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def align_camera_data(master_df, slave_df, slave_yolo_data):
    """
    Interpolates the slave camera's YOLO keypoints to match the 
    exact global timestamps of the master camera.
    
    Arguments:
    - master_df (pd.DataFrame): DataFrame containing the master camera's global timestamps.
    - slave_df (pd.DataFrame): DataFrame containing the slave camera's global timestamps.
    - slave_yolo_data (np.ndarray): Numpy array of shape (num_frames, num_joints, 2) 
                                    containing the extracted 2D coordinates.
                       
    Returns:
    - np.ndarray: A resampled Numpy array of the slave's YOLO data, perfectly aligned 
                  to the master's timeline.
    """
    master_timeline = master_df['global_timestamp_ms'].values
    slave_timeline = slave_df['global_timestamp_ms'].values
    
    num_frames, num_joints, coords = slave_yolo_data.shape
    aligned_yolo_data = np.zeros((len(master_timeline), num_joints, coords))
    
    # Interpolate each joint's X and Y coordinates independently
    for joint_idx in range(num_joints):
        for axis in range(coords):
            valid_data = slave_yolo_data[:, joint_idx, axis]
            
            # Create the mathematical spline based on the slave's timeline
            interpolator = interp1d(
                slave_timeline, 
                valid_data, 
                kind='linear', 
                bounds_error=False, 
                fill_value='extrapolate'
            )
            
            # Predict where the joint was at the exact millisecond the master fired
            aligned_yolo_data[:, joint_idx, axis] = interpolator(master_timeline)
            
    return aligned_yolo_data


if __name__ == "__main__":
    print("--- Testing Offline Linear Regression Synchronization ---")
    
    # IMPORTANT: Update these paths to the actual CSVs generated by your recording script
    # cam0_csv = 'realsense_cam_0/timestamps.csv'
    # cam1_csv = 'realsense_cam_1/timestamps.csv'
    
    try:
        # 1. Calculate the smooth global clocks
        # df0 = compute_global_timestamps(cam0_csv)
        # df1 = compute_global_timestamps(cam1_csv)
        
        # 2. Plot the graphs from the paper to visually verify
        # plot_synchronization_dashboard([df0, df1], ['Cam 0', 'Cam 1'])
        
        print("Uncomment the lines above and provide your actual CSV paths to run.")
        
    except FileNotFoundError:
        print("Error: Could not find the CSV files. Please check the paths.")