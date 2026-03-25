import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_global_timestamps(csv_path):
    """
    Applies offline linear regression to map hardware sensor timestamps 
    to the host's time-of-arrival clock, eliminating USB jitter.
    """
    df = pd.read_csv(csv_path)
    sensor_ts = df['sensor_timestamp_ms'].values
    arrival_ts = df['time_of_arrival_ms'].values
    
    slope, intercept = np.polyfit(sensor_ts, arrival_ts, 1)
    df['global_timestamp_ms'] = (sensor_ts * slope) + intercept
    return df


def evaluate_synchronization_accuracy(df_master, df_slave):
    """
    Numerically tests alignment and locates the exact frame of the worst hardware drop.
    """
    merged = pd.merge(df_master, df_slave, on='frame_index', suffixes=('_master', '_slave'))
    
    raw_error = np.abs(merged['time_of_arrival_ms_master'] - merged['time_of_arrival_ms_slave'])
    sync_error = np.abs(merged['global_timestamp_ms_master'] - merged['global_timestamp_ms_slave'])
    
    # --- NEW: LOCATE THE BIGGEST OFFSET ---
    max_error_idx = raw_error.idxmax()
    max_error_frame = merged.loc[max_error_idx, 'frame_index']
    
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
    
    print(f"\n[TARGET LOCATOR] Worst Hardware Drop Found:")
    print(f"  >> Frame Index : {max_error_frame}")
    print(f"  >> Copy/Paste  : zoom_window=({max_error_frame - 50}, {max_error_frame + 50})")
    print("="*50 + "\n")


def plot_synchronization_dashboard(dataframes, labels, zoom_window=(0, 100)):
    """
    Generates a 3-panel matplotlib dashboard.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_sensor, ax_arrival, ax_global = axes
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', '^', 'x']

    start_idx, end_idx = zoom_window

    for i, (df, label) in enumerate(zip(dataframes, labels)):
        df_sliced = df[(df['frame_index'] >= start_idx) & (df['frame_index'] <= end_idx)]
        
        if df_sliced.empty:
            print(f"Warning: No data in zoom_window {zoom_window} for {label}.")
            continue
            
        frames = df_sliced['frame_index'].values
        
        norm_sensor = df_sliced['sensor_timestamp_ms'].values - df_sliced['sensor_timestamp_ms'].values[0]
        norm_arrival = df_sliced['time_of_arrival_ms'].values - df_sliced['time_of_arrival_ms'].values[0]
        norm_global = df_sliced['global_timestamp_ms'].values - df_sliced['global_timestamp_ms'].values[0]
        
        plot_kwargs = {
            'marker': markers[i], 
            'color': colors[i], 
            'linestyle': '-', 
            'linewidth': 0.8,
            'markersize': 5,
            'alpha': 0.8,
            'label': label
        }
        
        ax_sensor.plot(frames, norm_sensor, **plot_kwargs)
        ax_arrival.plot(frames, norm_arrival, **plot_kwargs)
        ax_global.plot(frames, norm_global, **plot_kwargs)

    ax_sensor.set_title(f'Sensor Timestamp (Frames {start_idx}-{end_idx})')
    ax_sensor.set_xlabel('Frame')
    ax_sensor.set_ylabel('Normalized Time (ms)')
    ax_sensor.legend()
    ax_sensor.grid(True, linestyle='--', alpha=0.5)

    ax_arrival.set_title('Time-of-Arrival (USB Jitter)')
    ax_arrival.set_xlabel('Frame')
    ax_arrival.set_ylabel('Normalized Time (ms)')
    ax_arrival.legend()
    ax_arrival.grid(True, linestyle='--', alpha=0.5)

    ax_global.set_title('Global Timestamp (Regressed / Aligned)')
    ax_global.set_xlabel('Frame')
    ax_global.set_ylabel('Normalized Time (ms)')
    ax_global.legend()
    ax_global.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("--- Running Synchronization Test & Visualizer ---")
    
    cam0_csv = 'c:\\Users\\vrspr\\OneDrive\\Desktop\\realsense_cam_0\\timestamps.csv'
    cam1_csv = 'c:\\Users\\vrspr\\OneDrive\\Desktop\\realsense_cam_1\\timestamps.csv'
    
    try:
        df1_master = compute_global_timestamps(cam1_csv)
        df0_slave = compute_global_timestamps(cam0_csv)
        
        evaluate_synchronization_accuracy(df1_master, df0_slave)
        
        # Paste the output from the terminal right here to see the drop!
        plot_synchronization_dashboard(
            dataframes=[df1_master, df0_slave], 
            labels=['Cam 1 (Main)', 'Cam 0 (Supplementary)'],
            zoom_window=(18731, 18831) # UPDATE THIS AFTER RUNNING ONCE
        )
        
    except FileNotFoundError:
        print("Error: Could not find the CSV files. Please check the paths.")





   