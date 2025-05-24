# Read csv file with time and speed, find pedestrian and vehicle events
# 2025-05-15 J.Beale

"""
Tunable  parameters:
1. Vehicle event detection:
   - `eps=0.3` in DBSCAN for initial clustering sensitivity
   - `min_samples=5` in DBSCAN for minimum points to start a cluster

2. Split thresholds:
   - `time_gap_threshold=2.0` seconds for temporal separation
   - `max_duration=30.0` seconds for maximum event length
   - `velocity_gap=10.0` km/h for velocity band separation
   - `jump_threshold=5.0` km/h for sudden velocity increases

3. Final filtering:
   - `min_points=25` for minimum data points per valid vehicle event
"""   

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
from matplotlib import dates as mdates
import os

# =============================================================



# Configuration parameters
CONFIG = {
    # Event detection parameters
    'dbscan': {
        'eps': 0.05,           # clustering sensitivity
        'min_samples': 5      # minimum points to start cluster
    },
    
    # Density filtering
    'density': {
        'min_points': 15,     # minimum points in time window
        'time_window': 8,     # seconds to check density
        'max_rate': 22,       # maximum sample rate in Hz
        'required_density': 4,  # unique points per second
        'speed_threshold': 20   # km/h for speed threshold above which to keep
    },
    
    # Split thresholds
    'splits': {
        'time_gap': 2.0,      # seconds between events
        'max_duration': 60.0,  # maximum event length in seconds
        'velocity_gap': 10.0,  # km/h for velocity band separation
        'jump_threshold': 8.0, # km/h for sudden velocity increases
        'time_window': 0.8    # seconds for velocity jump detection
    },
    
    # Final filtering
    'filter': {
        'min_points': 25      # minimum points per valid event
    }
}

def filter_low_density_regions(df, config):
    """Filter out regions with too few unique points per unit time using rolling window.
    
    Args:
        df: DataFrame with 'epoch' column, 'kmh' column
        config: Configuration dictionary
    
    Returns:
        Tuple of (kept_df, rejected_df) with filtered and rejected points
    """
    df = df.copy()
    times = df['epoch'].values
    speeds = df['kmh'].values
    n = len(times)
    max_window_points = int(config['density']['max_rate'] * config['density']['time_window'])
    keep_mask = np.zeros(n, dtype=bool)
    
    # Use sliding window of fixed size in index space
    for i in range(n):
        # Look at points within max_window_points indices
        end_idx = min(i + max_window_points, n)
        window_times = times[i:end_idx]
        window_speeds = speeds[i:end_idx]
        
        # Create mask for points within time window
        time_mask = window_times - times[i] < config['density']['time_window']
        
        # Get unique speed values within the time window
        unique_speeds = np.unique(window_speeds[time_mask])
        points_in_window = len(unique_speeds)
        avg_speed = np.mean(np.abs(window_speeds[time_mask]))
        
        if points_in_window >= config['density']['min_points'] or avg_speed > config['density']['speed_threshold']:
            # Mark all points in this valid window as kept
            time_indices = np.where(time_mask)[0] + i
            keep_mask[time_indices] = True
    
    # Add statistics about filtered points
    kept_points = np.sum(keep_mask)
    rejected_points = n - kept_points
    pctRejected = rejected_points / n * 100
    # print(f"Density filter: kept {kept_points} points, removed {rejected_points} points ({rejected_points/n*100:.1f}%)")    

    return df[keep_mask].reset_index(drop=True), df[~keep_mask].reset_index(drop=True), pctRejected

def split_clusters_by_time_gap(df, time_gap_threshold=2.0, max_duration=30.0):
    """Split clusters that have time gaps or exceed maximum duration.
    
    Args:
        df: DataFrame with 'epoch' and 'cluster' columns
        time_gap_threshold: Maximum allowed time gap in seconds
        max_duration: Maximum allowed duration for a single vehicle event
    
    Returns:
        DataFrame with updated cluster labels
    """
    df = df.copy()
    new_cluster_label = df['cluster'].max() + 1
    
    # Process each original cluster
    for cluster in sorted(set(df['cluster'].unique()) - {-1}):
        # Get contiguous chunk of data for this cluster
        mask = df['cluster'] == cluster
        cluster_df = df[mask].sort_values('epoch')
        
        # Initialize segment start with first point
        current_segment = []
        current_start_time = None
        
        for idx, row in cluster_df.iterrows():
            if current_start_time is None:
                current_start_time = row['epoch']
                current_segment = [idx]
                continue
                
            time_gap = row['epoch'] - df.loc[current_segment[-1], 'epoch']
            segment_duration = row['epoch'] - current_start_time
            
            # Split if we exceed max duration or have a large time gap
            if segment_duration > max_duration or time_gap > time_gap_threshold:
                # Assign new label to completed segment
                df.loc[current_segment, 'cluster'] = new_cluster_label
                new_cluster_label += 1
                
                # Start new segment
                current_segment = [idx]
                current_start_time = row['epoch']
            else:
                current_segment.append(idx)
        
        # Handle final segment
        if current_segment:
            df.loc[current_segment, 'cluster'] = new_cluster_label
            new_cluster_label += 1
    
    return df

def split_clusters_by_direction(df):
    """Split clusters that contain both positive and negative velocities.
    
    Args:
        df: DataFrame with 'cluster' and 'kmh' columns
    
    Returns:
        DataFrame with updated cluster labels
    """
    df = df.copy()
    new_cluster_label = df['cluster'].max() + 1
    
    # Process each cluster
    for cluster in sorted(set(df['cluster'].unique()) - {-1}):
        cluster_df = df[df['cluster'] == cluster]
        
        # Check if cluster has both positive and negative velocities
        if (cluster_df['kmh'] > 0).any() and (cluster_df['kmh'] < 0).any():
            # Split into two clusters based on velocity sign
            df.loc[(df['cluster'] == cluster) & (df['kmh'] > 0), 'cluster'] = new_cluster_label
            new_cluster_label += 1
            df.loc[(df['cluster'] == cluster) & (df['kmh'] < 0), 'cluster'] = new_cluster_label
            new_cluster_label += 1
    
    return df

def split_clusters_by_velocity_bands(df, velocity_gap=10.0):
    """Split clusters that have distinct velocity bands separated by gaps.
    
    Args:
        df: DataFrame with 'cluster' and 'kmh' columns
        velocity_gap: Minimum gap in km/h between distinct velocity bands
    
    Returns:
        DataFrame with updated cluster labels
    """
    df = df.copy()
    new_cluster_label = df['cluster'].max() + 1
    
    # Process each cluster
    for cluster in sorted(set(df['cluster'].unique()) - {-1}):
        # Get this cluster's data
        cluster_df = df[df['cluster'] == cluster]
        
        # Skip small clusters
        if len(cluster_df) < 25:
            continue
            
        # Sort speeds and find gaps
        speeds = np.array(sorted(cluster_df['kmh']))
        gaps = np.diff(speeds)
        significant_gaps = np.where(gaps > velocity_gap)[0]
        
        if len(significant_gaps) > 0:
            # There are distinct velocity bands
            split_points = [speeds[i] for i in significant_gaps]
            
            # Create index mask for this cluster's data
            cluster_indices = cluster_df.index
            
            # Assign new labels to each band
            for i, split_point in enumerate(split_points):
                if i == 0:
                    # First band
                    band_mask = cluster_df['kmh'] <= split_point
                else:
                    # Middle bands
                    band_mask = (cluster_df['kmh'] > split_points[i-1]) & \
                              (cluster_df['kmh'] <= split_point)
                
                # Apply mask only to this cluster's indices
                df.loc[cluster_indices[band_mask], 'cluster'] = new_cluster_label
                new_cluster_label += 1
            
            # Last band
            band_mask = cluster_df['kmh'] > split_points[-1]
            df.loc[cluster_indices[band_mask], 'cluster'] = new_cluster_label
            new_cluster_label += 1
    
    return df

def split_clusters_by_velocity_jump(df, jump_threshold=5.0, time_window=0.8, debug=False):
    """Split clusters where velocity has sudden significant increase.
    
    Args:
        df: DataFrame with 'epoch', 'kmh', and 'cluster' columns
        jump_threshold: Minimum velocity increase in km/h to trigger split
        time_window: Time window in seconds to look for velocity changes
        debug: Print debug info for large velocity jumps
    """
    df = df.copy()
    new_cluster_label = df['cluster'].max() + 1
    
    # Process each cluster
    for cluster in sorted(set(df['cluster'].unique()) - {-1}):
        cluster_df = df[df['cluster'] == cluster].sort_values('epoch')
        
        if len(cluster_df) < 25:  # Skip small clusters
            continue
            
        # Calculate point-to-point velocity changes
        velocities = cluster_df['kmh'].values
        vel_jumps = np.diff(velocities)
        times = cluster_df['epoch'].values
        
        # Find significant jumps
        split_points = []
        for i in range(len(vel_jumps)):
            if vel_jumps[i] > jump_threshold:
                if debug and vel_jumps[i] > 10:
                    print(f"Found jump of {vel_jumps[i]:.2f} km/h at time {times[i]:.2f}")
                    print(f"Velocities: {velocities[i]:.2f} -> {velocities[i+1]:.2f}")
                    print(f"Index in cluster_df: {i+1}")  # Debug: show where split will occur
                split_points.append(i + 1)  # Index in cluster_df where split should occur
        
        # Apply splits
        if split_points:
            # First section keeps original label
            for split_idx in sorted(split_points):
                # Get actual DataFrame indices for points after split
                indices_to_split = cluster_df.index[split_idx:]
                
                # Assign new label to these points
                df.loc[indices_to_split, 'cluster'] = new_cluster_label
                if debug:
                    print(f"Split applied at index {split_idx}, new label {new_cluster_label}")
                new_cluster_label += 1
    
    return df

def median_filter(vec):    
    padded = np.pad(vec, pad_width=1, mode='edge')
    filtered = np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]]), axis=0)
    return filtered

def get_smooth_max_speed(speeds, window_size=9):
    """Calculate smoothed maximum speed using rolling window.
    
    Args:
        speeds: numpy array of speed values
        window_size: size of rolling window for smoothing
    
    Returns:
        Smoothed maximum speed value
    """
    if len(speeds) < window_size:
        return np.abs(speeds).max()
                
    even_idx = speeds[::2]  # all even-indexed elements (0, 2, 4, ...)
    odd_idx = speeds[1::2]  # all odd-indexed elements (1, 3, 5, ...)
    evenf = median_filter(even_idx)
    oddf = median_filter(odd_idx)   

    combined = np.empty(len(speeds), dtype=np.float32)  
    combined[::2] = evenf
    combined[1::2] = oddf
    speeds = median_filter(np.abs(combined))


    # Use rolling window to get local averages
    smooth_speeds = np.convolve(speeds, 
                               np.ones(window_size)/window_size, 
                               mode='valid')

    #plt.plot(speeds, label='median') # debug
    #plt.plot(smooth_speeds, label='smoothed') # debug
    #plt.show()

    return smooth_speeds.max()

# ==========================================================
# Main script

if __name__ == "__main__":
    import sys
    import argparse

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Process doppler radar data')
    parser.add_argument('filename', help='Input CSV file to process')
    parser.add_argument('-n', '--no-plot', action='store_true', 
                      help='Disable plotting (run in non-interactive mode)')

    args = parser.parse_args()
    file = args.filename
    doPlot = not args.no_plot  # True by default, False if -n is specified

    #file = r"20250519_0000_DpCh1.csv" # fair amount of rain
    #file =  r"20250516_000003_SerialLog.csv" # no rain

    #indir = r"/home/john/Documents/doppler"
    indir = r"C:\Users\beale\Documents\doppler"
    infile = os.path.join(indir, file)

    # Remove or comment out the hardcoded doPlot assignment
    #doPlot = False  # Set to True to display events in a plot

    # --- Step 1: Load and preprocess the CSV file ---
    chunk_size = 10000  # Adjust based on available memory
    df_chunks = pd.read_csv(infile, chunksize=chunk_size)
    df = pd.concat([chunk[chunk['kmh'] != 0.0] for chunk in df_chunks])

    #df['epoch'] = df['epoch'].astype(np.float32)  # instead of float64
    #df['kmh'] = df['kmh'].astype(np.float32)

    # Filter out zero velocity readings
    df = df[df['kmh'] != 0.0].reset_index(drop=True)

    # Filter out low density regions (e.g., rain)
    kept_df, rejected_df, pctRejected = filter_low_density_regions(df, CONFIG)
    df = kept_df  # Continue with kept points

    # Add visualization of rejected points
    if doPlot:
        plt.figure(figsize=(15, 8))
        
        # Convert epoch times to local datetime objects
        local_tz = pytz.timezone('America/Los_Angeles')
        utc_tz = pytz.UTC
        
        # Plot rejected points
        rejected_times = [datetime.fromtimestamp(ts, tz=utc_tz).astimezone(local_tz) 
                        for ts in rejected_df['epoch']]
        plt.scatter(rejected_times, rejected_df['kmh'],
                color='red', alpha=0.2, label='Filtered Out', s=10)
        
        plt.xlabel("Local Time (PDT)")
        plt.ylabel("Speed (km/h)")
        plt.title("Points Removed by Density Filter")
        plt.grid(True)
        plt.legend()
        plt.gcf().autofmt_xdate()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S', tz=local_tz))
        plt.tight_layout()
        plt.show()

    # --- Step 2: Preprocess the data ---
    # Normalize time to seconds from start
    df['time'] = df['epoch'] - df['epoch'].min()

    # Process in chunks for DBSCAN
    chunk_size = 5000  # Adjust based on available memory
    overlap = 1000     # Number of points to overlap between chunks
    all_labels = np.full(len(df), -1, dtype=np.int32)
    current_max_label = -1

    # Process in chunks for DBSCAN
    for start_idx in range(0, len(df), chunk_size - overlap):
        end_idx = min(start_idx + chunk_size, len(df))
        
        # Scale features for this chunk
        chunk_data = df.iloc[start_idx:end_idx][['time', 'kmh']].values
        scaler = StandardScaler()
        X_scaled_chunk = scaler.fit_transform(chunk_data)
        
        # print(f"Processing DBSCAN chunk {start_idx//(chunk_size-overlap) + 1}")
        # Run DBSCAN on chunk
        dbscan = DBSCAN(
            eps=CONFIG['dbscan']['eps'],
            min_samples=CONFIG['dbscan']['min_samples'],
            metric='euclidean'
        ).fit(X_scaled_chunk)
        
        # Adjust labels to not overlap with previous chunks
        valid_labels = dbscan.labels_[dbscan.labels_ != -1]
        if len(valid_labels) > 0:
            dbscan.labels_[dbscan.labels_ != -1] += (current_max_label + 1)
            current_max_label = dbscan.labels_.max()
        
        # Store labels, handling overlap region
        if start_idx == 0:
            # First chunk - store all labels
            all_labels[start_idx:end_idx] = dbscan.labels_
        else:
            # For subsequent chunks:
            # 1. Find matching clusters in overlap region
            overlap_start = start_idx
            overlap_end = start_idx + overlap
            overlap_old_labels = all_labels[overlap_start:overlap_end]
            overlap_new_labels = dbscan.labels_[:overlap]
            
            # 2. Create mapping between old and new labels in overlap
            label_map = {}
            for old, new in zip(overlap_old_labels, overlap_new_labels):
                if old != -1 and new != -1:
                    if new + current_max_label + 1 not in label_map:
                        label_map[new + current_max_label + 1] = old
            
            # 3. Apply mapping to new labels
            mapped_labels = dbscan.labels_.copy()
            for new, old in label_map.items():
                mapped_labels[mapped_labels == new - current_max_label - 1] = old
            
            # 4. Store non-overlap region
            all_labels[start_idx + overlap:end_idx] = mapped_labels[overlap:]
        
        del X_scaled_chunk  # Free memory

    # Add cluster labels to DataFrame
    df['cluster'] = all_labels
    # print("DBSCAN clustering complete.")

    # Split clusters with large time gaps

    df = split_clusters_by_direction(df)
    df = split_clusters_by_velocity_bands(df, velocity_gap=CONFIG['splits']['velocity_gap'])
    df = split_clusters_by_velocity_jump(df, 
        jump_threshold=CONFIG['splits']['jump_threshold'],
        time_window=CONFIG['splits']['time_window'])
    df = split_clusters_by_time_gap(df, 
        time_gap_threshold=CONFIG['splits']['time_gap'],
        max_duration=CONFIG['splits']['max_duration'])

    # Filter out clusters with too few points (moved after all splitting)
    min_points = CONFIG['filter']['min_points']
    valid_clusters = []
    for label in sorted(set(df['cluster'].unique()) - {-1}):
        cluster_df = df[df['cluster'] == label]
        if len(cluster_df) >= min_points:
            valid_clusters.append(label)
        else:
            df.loc[df['cluster'] == label, 'cluster'] = -1  # mark as noise

    # print(f"\nDiscarded {len(set(df['cluster'].unique()) - {-1} - set(valid_clusters))} vehicles with < {min_points} points")
    # print(f"Final vehicle count after filtering: {len(valid_clusters)}")

    if doPlot:
        # --- Step 4: Plot events grouped by color ---
        plt.figure(figsize=(15, 8))
        
        # Convert epoch times to local datetime objects
        local_tz = pytz.timezone('America/Los_Angeles')  # PDT/PST timezone
        utc_tz = pytz.UTC

        # Plot noise points first so they're in the background
        noise = df[df['cluster'] == -1]
        noise_times = [datetime.fromtimestamp(ts, tz=utc_tz).astimezone(local_tz) 
                    for ts in noise['epoch']]
        plt.scatter(noise_times, noise['kmh'], 
                color='lightgray', alpha=0.5, label='Noise', s=10)

        # Plot clusters with lines connecting points
        for label in sorted(set(df['cluster'].unique()) - {-1}):  # exclude noise
            cluster_df = df[df['cluster'] == label].sort_values('epoch')
            cluster_times = [datetime.fromtimestamp(ts, tz=utc_tz).astimezone(local_tz) 
                            for ts in cluster_df['epoch']]
            plt.scatter(cluster_times, cluster_df['kmh'],
                    label=f'Vehicle {label}', s=30, alpha=0.6)
            # Connect points within each cluster
            plt.plot(cluster_times, cluster_df['kmh'],
                    alpha=0.4)

        # Format x-axis
        plt.gcf().autofmt_xdate()  # Angle and align the tick labels
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S', tz=local_tz))
        
        plt.xlabel("Local Time (PDT)")
        plt.ylabel("Speed (km/h)")
        plt.title("Traffic Events")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Print cluster statistics
    final_vehicles = len(valid_clusters)
    # print(f"Final vehicle count after splitting: {final_vehicles}")

    # Create sequential label mapping
    label_map = {label: idx for idx, label in enumerate(sorted(valid_clusters))}
    ped = 0
    shortPed = 0

    # print("ID, Points, Duration(s), Dir, AvgSpd, MaxSpd, SmoothMax, Accel, Type")
    # Create DataFrame to store event statistics
    event_stats = pd.DataFrame(columns=['event_id', 'points', 'duration', 'direction', 
                                    'avg_speed', 'max_speed', 'smooth_max', 'accel', 
                                    'type', 'start_time'])

    for slabel in valid_clusters:
        isVehicle = 1
        cluster_df = df[df['cluster'] == slabel]
        duration = cluster_df['epoch'].max() - cluster_df['epoch'].min()
        speeds = cluster_df['kmh'].to_numpy()
        dir = int(np.sign(speeds.mean()))
        accel = np.abs(np.diff(speeds))
        aAvg = accel.mean()
        avg_speed = np.abs(speeds).mean()
        max_speed = np.abs(speeds).max()
        
        
        aAvg *= 20 / avg_speed  # normalize by average speed
        if (aAvg > 0.5) and (avg_speed < 10.0):        
            if (duration > 5.0):
                isVehicle = 0
                ped += 1
            else:
                isVehicle = -1
                shortPed += 1           

        # Calculate smooth max only for likely vehicles
        smooth_max = get_smooth_max_speed(speeds) if (isVehicle==1) else max_speed

        #print(f"{label_map[slabel]}, {len(cluster_df)}, {duration:.1f}, {dir}, "
        #      f"{avg_speed:.1f}, {max_speed:.1f}, {smooth_max:.1f}, {aAvg:.1f}, {isVehicle}")
        
        # Get start time of event
        start_time = cluster_df['epoch'].min()
        
        # Add row to event_stats DataFrame
        event_stats.loc[len(event_stats)] = {
            'event_id': label_map[slabel],
            'points': len(cluster_df),
            'duration': duration,
            'direction': dir,
            'avg_speed': avg_speed,
            'max_speed': max_speed,
            'smooth_max': smooth_max,
            'accel': aAvg,
            'type': isVehicle,
            'start_time': pd.Timestamp(start_time, unit='s', tz='UTC').tz_convert('America/Los_Angeles')
        }

    # Print summary of DataFrame
    #print("\nEvent Statistics Summary:")
    #pd.set_option('display.float_format', lambda x: '%.2f' % x)
    #print(event_stats.describe())

    print("%30s,%4.1f,%3d,%3d,%3d, " % (file,pctRejected,ped,shortPed,(final_vehicles-(ped+shortPed))), end="")

    # Find and print details of fastest event
    fastest_event = event_stats.loc[event_stats['smooth_max'].idxmax()]
    maxT = fastest_event['start_time'].strftime('%Y-%m-%d %H:%M:%S')
    maxKMH = fastest_event['smooth_max']
    maxMPH = maxKMH * 0.621371  # Convert to mph
    maxD = fastest_event['duration']
    maxP = fastest_event['points']

    print(f"%s, %.2f, %.2f, %4.1f, %3d" % (maxT,maxKMH,maxMPH,maxD,maxP))

    summaryPath = os.path.join(indir, file[:-4]+"_summary.csv")  
    event_stats.to_csv(summaryPath, index=False, float_format='%.4f') # save event statistics to CSV
    print(f"Event statistics saved to {summaryPath}")