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

infile =  r"C:\Users\beale\Documents\doppler\20250518_000004_SerialLog.csv"
#infile =  r"C:\Users\beale\Documents\doppler\20250518_clip1.csv"
#infile =  r"C:\Users\beale\Documents\doppler\20250518_clip2.csv"

doPlot = True  # Set to True to display events in a plot

# --- Step 1: Load and preprocess the CSV file ---
df = pd.read_csv(infile)
df.columns = ['epoch', 'kmh']

# Filter out zero velocity readings
df = df[df['kmh'] != 0.0].reset_index(drop=True)

# --- Step 2: Preprocess the data ---
# Normalize time to seconds from start
df['time'] = df['epoch'] - df['epoch'].min()
df['velocity'] = df['kmh']

# Scale features to comparable ranges
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['time', 'velocity']].values)

# --- Step 3: Run DBSCAN clustering ---
# Adjusted parameters:
# eps: increased to allow more temporal connection
# min_samples: reduced to catch shorter tracks
dbscan = DBSCAN(
    eps=0.3,           # reduced because we're using scaled features
    min_samples=5,     # minimum points to form a cluster
    metric='euclidean'
).fit(X_scaled)

# Add cluster labels to DataFrame
df['cluster'] = dbscan.labels_

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

# Split clusters with large time gaps
df = split_clusters_by_time_gap(df, time_gap_threshold=2.0, max_duration=30.0)

# Apply our splitting steps in sequence
df = split_clusters_by_direction(df)
df = split_clusters_by_velocity_bands(df, velocity_gap=10.0)
df = split_clusters_by_velocity_jump(df, jump_threshold=5.0, time_window=0.8)
df = split_clusters_by_time_gap(df, time_gap_threshold=2.0, max_duration=30.0)

# Filter out clusters with too few points (moved after all splitting)
min_points = 25
valid_clusters = []
for label in sorted(set(df['cluster'].unique()) - {-1}):
    cluster_df = df[df['cluster'] == label]
    if len(cluster_df) >= min_points:
        valid_clusters.append(label)
    else:
        df.loc[df['cluster'] == label, 'cluster'] = -1  # mark as noise

print(f"\nDiscarded {len(set(df['cluster'].unique()) - {-1} - set(valid_clusters))} vehicles with < {min_points} points")
print(f"Final vehicle count after filtering: {len(valid_clusters)}")

if doPlot:
    # --- Step 4: Plot events grouped by color ---
    plt.figure(figsize=(15, 8))

    # Plot noise points first so they're in the background
    noise = df[df['cluster'] == -1]
    plt.scatter(noise['time'], noise['velocity'], 
            color='lightgray', alpha=0.5, label='Noise', s=10)

    # Plot clusters with lines connecting points
    for label in sorted(set(df['cluster'].unique()) - {-1}):  # exclude noise
        cluster_df = df[df['cluster'] == label].sort_values('time')  # Sort by time
        plt.scatter(cluster_df['time'], cluster_df['velocity'],
                label=f'Vehicle {label}', s=30, alpha=0.6)
        # Connect points within each cluster
        plt.plot(cluster_df['time'], cluster_df['velocity'],
                alpha=0.4)

    plt.xlabel("Time (s since first record)")
    plt.ylabel("Speed (km/h)")
    plt.title("Detected Vehicle Events (DBSCAN)")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Print cluster statistics
final_vehicles = len(valid_clusters)
print(f"Final vehicle count after splitting: {final_vehicles}")
print("\nVehicle details:")
# Create mapping of old labels to sequential numbers starting at 1
label_map = {old: i+1 for i, old in enumerate(valid_clusters)}

ped = 0 # pedestrian count
shortPed = 0 # short-duration pedestrian count

for slabel in valid_clusters:
    isVehicle = 1
    cluster_df = df[df['cluster'] == slabel]
    duration = cluster_df['epoch'].max() - cluster_df['epoch'].min()
    speeds = cluster_df['kmh'].to_numpy()
    dir = int(np.sign(speeds.mean()))
    accel = np.abs(np.diff(speeds)) # should divide by time, but less accurate
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
    print(f"{label_map[slabel]}, {len(cluster_df)}, {duration:.1f}s, {dir}, {avg_speed:.1f}, {max_speed:.1f}, {aAvg:.1f}, {isVehicle}")
          
print(f"\nPedestrians: {ped}  <5s Ped: {shortPed}  Vehicles: {final_vehicles - (ped + shortPed)}")          