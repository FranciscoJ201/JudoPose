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
    slope, intercept = np.polyfit(sensor_ts, arrival_ts, 1)
    
    # Apply the regression to generate the smooth global timeline
    df['global_timestamp_ms'] = (sensor_ts * slope) + intercept
    
    return df


def evaluate_synchronization_accuracy(df_master, df_slave):
    """
    Numerically tests the alignment accuracy between two cameras before 
    and after applying the offline linear regression.
    
    Arguments:
    - df_master (pd.DataFrame): The calculated timestamps for the master camera.
    - df_slave (pd.DataFrame): The calculated timestamps for the slave camera.
    """
    # Merge the dataframes strictly on matching frame indices
    merged = pd.merge(df_master, df_slave, on='frame_index', suffixes=('_master', '_slave'))
    
    # Calculate the absolute difference in arrival times (Raw USB Jitter)
    raw_error = np.abs(merged['time_of_arrival_ms_master'] - merged['time_of_arrival_ms_slave'])
    
    # Calculate the absolute difference in regressed global times (Cleaned)
    sync_error = np.abs(merged['global_timestamp_ms_master'] - merged['global_timestamp_ms_slave'])
    
    print("\n" + "="*50)
    print(" NUMERICAL SYNCHRONIZATION TEST")
    print("="*50)
    print(f"Total overlapping frames analyzed : {len(merged)}")
    print(f"\n[BEFORE] Raw Hardware (Time of Arrival Jitter):")
    print(f"  Mean Delay : {raw_error.mean():.4f} ms")
    print(f"  Max Delay  : {raw_error.max():.4f} ms")
    
    print(f"\n[AFTER] Regressed Alignment (Global Timestamps):")
    print(f"  Mean Delay : {sync_error.mean():.6f} ms")
    print(f"  Max Delay  : {sync_error.max():.6f} ms")
    print("="*50 + "\n")


def plot_synchronization_dashboard(dataframes, labels, zoom_window=(5000, 5100)):
    """
    Generates a 3-panel matplotlib dashboard replicating the paper's figures.
    Uses a zoom window and thin markers to make the micro-stutters visible.
    
    Arguments:
    - dataframes (list): List of pandas DataFrames returned by compute_global_timestamps.
    - labels (list): List of string labels for the cameras (e.g., ['Cam 0', 'Cam 1']).
    - zoom_window (tuple): (start_frame, end_frame) to slice the data for visibility.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_sensor, ax_arrival, ax_global = axes
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', '^', 'x']

    start_idx, end_idx = zoom_window

    for i, (df, label) in enumerate(zip(dataframes, labels)):
        # Slice the dataframe to the zoom window so markers don't crush together
        df_sliced = df[(df['frame_index'] >= start_idx) & (df['frame_index'] <= end_idx)]
        frames = df_sliced['frame_index'].values
        
        # Normalize timestamps strictly based on the FIRST frame of the sliced window
        norm_sensor = df_sliced['sensor_timestamp_ms'].values - df_sliced['sensor_timestamp_ms'].values[0]
        norm_arrival = df_sliced['time_of_arrival_ms'].values - df_sliced['time_of_arrival_ms'].values[0]
        norm_global = df_sliced['global_timestamp_ms'].values - df_sliced['global_timestamp_ms'].values[0]
        
        # Adjusted plot settings: thin lines, visible distinct markers
        plot_kwargs = {
            'marker': markers[i], 
            'color': colors[i], 
            'linestyle': '-', 
            'linewidth': 0.8,    # Thinner lines
            'markersize': 5,     # Smaller but distinct markers
            'alpha': 0.8,
            'label': label
        }
        
        ax_sensor.plot(frames, norm_sensor, **plot_kwargs)
        ax_arrival.plot(frames, norm_arrival, **plot_kwargs)
        ax_global.plot(frames, norm_global, **plot_kwargs)

    # Formatting Panel 1
    ax_sensor.set_title(f'Sensor Timestamp (Frames {start_idx}-{end_idx})')
    ax_sensor.set_xlabel('Frame')
    ax_sensor.set_ylabel('Normalized Time (ms)')
    ax_sensor.legend()
    ax_sensor.grid(True, linestyle='--', alpha=0.5)

    # Formatting Panel 2
    ax_arrival.set_title('Time-of-Arrival (USB Jitter)')
    ax_arrival.set_xlabel('Frame')
    ax_arrival.set_ylabel('Normalized Time (ms)')
    ax_arrival.legend()
    ax_arrival.grid(True, linestyle='--', alpha=0.5)

    # Formatting Panel 3
    ax_global.set_title('Global Timestamp (Regressed / Aligned)')
    ax_global.set_xlabel('Frame')
    ax_global.set_ylabel('Normalized Time (ms)')
    ax_global.legend()
    ax_global.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("--- Running Synchronization Test & Visualizer ---")
    
    # 1. Update these to point to your actual captured CSVs
    cam1_csv = 'realsense_cam_1/timestamps.csv' # Set the flawless camera as master
    cam0_csv = 'realsense_cam_0/timestamps.csv' 
    
    try:
        # Calculate the smooth global clocks
        df1_master = compute_global_timestamps(cam1_csv)
        df0_slave = compute_global_timestamps(cam0_csv)
        
        # Run the numerical test to measure millisecond accuracy
        evaluate_synchronization_accuracy(df1_master, df0_slave)
        
        # Plot the 100-frame zoomed dashboard to visually verify
        # Adjust the zoom_window to focus on the exact frame where cam_0 dropped earlier!
        plot_synchronization_dashboard(
            dataframes=[df1_master, df0_slave], 
            labels=['Cam 1 (Master)', 'Cam 0 (Slave)'],
            zoom_window=(9750, 9850) 
        )
        
    except FileNotFoundError:
        print("Error: Could not find the CSV files. Please check the paths.")