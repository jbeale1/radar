#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
import glob
import re

import numpy as np
import pandas as pd
from scipy.signal import medfilt, savgol_filter

def hampel_filter(x, window_size=11, n_sigmas=3):
    """
    A Hampel filter to replace outliers in x with the local median.
    """
    x = pd.Series(x)
    k = 1.4826  # scale factor for Gaussian
    rolling_median = x.rolling(window_size, center=True, min_periods=1).median()
    mad = x.rolling(window_size, center=True, min_periods=1) \
           .apply(lambda w: np.median(np.abs(w - np.median(w))), raw=True)
    threshold = n_sigmas * k * mad
    outliers = np.abs(x - rolling_median) > threshold
    x[outliers] = rolling_median[outliers]
    return x.values

def get_robust_max_speed(seg):
    """
    Adaptive, outlier-resistant max-speed estimator.
    """
    times  = seg['epoch'].values
    speeds = np.abs(seg['kmh'].values)
    n = len(times)
    if n < 70:
        return np.median(speeds)

    # 1) isolate central 60% of the pass
    peak_idx    = np.argmax(np.abs(speeds))
    center_time = times[peak_idx]
    tspan       = times[-1] - times[0]
    mid_mask    = np.abs(times - center_time) < 0.3 * tspan
    cs_raw      = speeds[mid_mask]
    if len(cs_raw) < 5:
        return np.median(speeds)

    # 2) choose pedestrian vs. vehicle parameters
    raw_max = np.max(cs_raw)
    if raw_max < 20.0:
        # pedestrian
        med_w    = 41    # median window
        sg_w     = 41    # SavGol window (odd)
        roll_w   = 15    # rolling quantile window
        perc     = 85    # percentile for rolling quantile
        sigma_k  = 2     # Hampel sensitivity
    else:
        # vehicle
        med_w    = 7
        sg_w     = 11
        roll_w   = 9
        perc     = 90
        sigma_k  = 3

    # 3) knock down spikes
    cs_h = hampel_filter(cs_raw, window_size=med_w, n_sigmas=sigma_k)

    # 4) smooth
    
    # ensure med_w is odd and no larger than len(cs_h)
    n     = len(cs_h)
    med_w = min(med_w, n if n%2==1 else n-1)
    if med_w < 3:
        # if your signal is very short, just skip the filter altogether:
        cs_m = cs_h.copy()
    else:
        cs_m = medfilt(cs_h, kernel_size=med_w)
    
    if len(cs_m) >= sg_w:
        cs_s = savgol_filter(cs_m, window_length=sg_w, polyorder=2)
    else:
        cs_s = cs_m

    # 5) rolling-quantile to create a very smooth “upper envelope”
    s = pd.Series(cs_s)
    rq = s.rolling(window=roll_w, center=True, min_periods=1) \
          .quantile(perc/100.0)

    # 6) plateau average: take the mean of the top pct of that envelope
    thresh = np.percentile(rq.dropna(), perc)
    top_vals = rq[rq >= thresh]
    robust_max = top_vals.mean() if len(top_vals)>0 else rq.max()

    # —— optional debug plot —— 
    """
    plt.plot(times, speeds,       'r.', alpha=0.3, label='all')
    plt.plot(times[mid_mask], cs_h,'ko', alpha=0.4, label='hampel')
    plt.plot(times[mid_mask], cs_s,'b-', lw=2,    label='sav-gol')
    plt.plot(times[mid_mask], rq,  'k--',         label=f'roll {perc}th')
    plt.axhline(robust_max, color='g', ls='-', label=f'plateau avg: {robust_max:.1f}')
    plt.legend(); plt.show()
    """
    # ————————————————

    return float(robust_max)



def detect_motion_events_seg(df,
                              gap_threshold: float = 1.5,
                              dv_threshold: float = 16.0,
                              min_rate: float = 5.0,
                              min_duration: float = 2.0):
    """
    Segment readings purely by time gaps and large velocity jumps:
    - Break on dt > gap_threshold or |dv| > dv_threshold.
    - Then filter segments by minimum duration and sampling rate.
    """
    # ensure sorted
    df = df.sort_values('epoch').reset_index(drop=True)
    segments = []
    current = [df.iloc[0]]

    # build segments
    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]
        dt = curr['epoch'] - prev['epoch']
        dv = abs(curr['kmh'] - prev['kmh'])
        # break if time gap or velocity jump too big
        if dt > gap_threshold or dv > dv_threshold:
            segments.append(pd.DataFrame(current))
            current = [curr]
        else:
            current.append(curr)
    # last segment
    segments.append(pd.DataFrame(current))

    # summarize each segment
    events = []
    for seg in segments:
        t0 = seg['epoch'].iloc[0]
        t1 = seg['epoch'].iloc[-1]
        duration = t1 - t0
        n_pts = len(seg)
        if duration >= min_duration and (n_pts / duration) >= min_rate:
            speeds = seg['kmh'].abs()
            avg_vel = seg['kmh'].mean()
            
            # Calculate robust maximum speed
            robust_max = get_robust_max_speed(seg)
            
            # Format time string
            dt = datetime.fromtimestamp(t0)
            time_str = dt.strftime('%H:%M:%S.%f')[:-4]  # Keep 2 decimal places
            
            # Modify type assignment to include short pedestrians
            if speeds.mean() > 15.0:
                event_type = 1  # vehicle
            elif duration < 5.0:
                event_type = -1  # short pedestrian (noise)
            else:
                event_type = 0  # normal pedestrian
                
            events.append({
                'epoch':      t0,         # start time in epoch seconds
                'time':       time_str,   # time in HH:MM:SS.ss format
                'duration':   duration,    # duration in seconds
                'samples':    n_pts,      # number of samples in segment
                'avg_speed':  speeds.mean(), # average speed in km/h
                'max_speed':  robust_max,    # robust max speed in km/h
                'direction':  int(np.sign(avg_vel)), # direction: 1 for forward, -1 for backward
                'type':       event_type  # event type: 1=vehicle, 0=pedestrian, -1=noise
            })

    events_df = pd.DataFrame(events)
    
    return events_df  # Remove event_id addition

def process_radar_file(fname: str, indir: str, show_plot: bool = True):
    """
    Process radar data file(s) and detect motion events.
    Handles wildcards in filenames - e.g., '20250529_0000_DpCh1.csv' will process
    all files matching '20250529_????_DpCh1.csv'
    
    Args:
        fname: Name pattern of input CSV file(s)
        indir: Directory containing the input file(s)
        show_plot: Whether to display the plot (default: True)
    
    Returns:
        pd.DataFrame: DataFrame containing detected events
    """
    # Convert single filename to pattern
    pattern = re.sub(r'_\d{4}_', '_????_', fname)
    file_pattern = os.path.join(indir, pattern)
    
    # Get list of matching files
    files = sorted(glob.glob(file_pattern))
    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return None
    
    # print(f"Processing {len(files)} files matching {pattern}")
    
    # Load and concatenate all matching files
    df_list = []
    for f in files:
        print(f"Reading {os.path.basename(f)}")
        df = pd.read_csv(f)
        df_list.append(df)
    
    df = pd.concat(df_list, ignore_index=True)
    df = df.sort_values('epoch').reset_index(drop=True)
    df = df[df['kmh'] != 0.0].copy()

    all_events = []
    # split by direction
    for sign in [+1, -1]:
        df_sign = df[df['kmh'] * sign > 0].copy()
        if df_sign.empty:
            continue
        ev_df = detect_motion_events_seg(
            df_sign,
            gap_threshold=1.5,
            dv_threshold=10.0,
            min_rate=5.0,
            min_duration=2.0
        )
        ev_df['direction'] = sign
        all_events.append(ev_df)

    if not all_events:
        print("No events detected.")
        return None

    events_df = pd.concat(all_events, ignore_index=True)
    events_df = events_df.sort_values('epoch').reset_index(drop=True)
    
    outpath = os.path.join(indir, fname.replace('.csv', '_summary.csv'))
    events_df.to_csv(outpath, index=False)
    print(events_df)
    print(f"Detected {len(events_df)} motion events → {outpath}")

    # Print summary statistics
    n_pedestrians = len(events_df[events_df['type'] == 0])
    n_vehicles = len(events_df[events_df['type'] == 1])
    n_short_peds = len(events_df[events_df['type'] == -1])
    
    # Calculate samples per second for vehicles and pedestrians
    vehicle_events = events_df[events_df['type'] == 1]
    ped_events = events_df[events_df['type'] == 0]
    
    if not vehicle_events.empty:
        vehicle_rate = vehicle_events['samples'].sum() / vehicle_events['duration'].sum()
        print(f"Vehicle events average {vehicle_rate:.1f} samples/second")
    
    if not ped_events.empty:
        ped_rate = ped_events['samples'].sum() / ped_events['duration'].sum()
        print(f"Pedestrian events average {ped_rate:.1f} samples/second")
    
    print(f"\nSummary:")
    print(f"Vehicles: {n_vehicles}")
    print(f"Pedestrians: {n_pedestrians}")
    print(f"Low-V noise: {n_short_peds}")
    print(f"Total events: {len(events_df)-n_short_peds}\n")

    if show_plot:
        plot_events(df, events_df, fname)
        
    return events_df

def plot_events(df: pd.DataFrame, events_df: pd.DataFrame, title: str):
    """
    Plot detected motion events.
    
    Args:
        df: Original DataFrame with raw data
        events_df: DataFrame containing detected events
        title: Title for the plot
    """
    plt.figure(figsize=(15, 8))
    utc = pytz.UTC
    local_tz = pytz.timezone('America/Los_Angeles')

    # Define color maps for different event types
    import matplotlib.cm as cm
    n_events = len(events_df)
    
    # Create dark, saturated colors for vehicles using tab20
    vehicle_colors = np.array([
        [0.00, 0.00, 0.60, 0.6],  # dark blue
        [0.60, 0.00, 0.00, 0.6],  # dark red
        [0.2, 0.30, 0.00, 0.6],   # dark greenish
        [0.50, 0.00, 0.50, 0.6],  # dark purple
        [0.60, 0.30, 0.00, 0.6],  # dark orange
        [0.00, 0.40, 0.40, 0.6],  # dark cyan
        [0.40, 0.20, 0.00, 0.6],  # dark brown
        [0.30, 0.00, 0.30, 0.6],  # dark magenta
        [0.00, 0.30, 0.50, 0.6],  # dark teal
        [0.50, 0.25, 0.25, 0.6],  # dark maroon
    ])
    
    # Create dark green colors for pedestrians
    ped_colors = np.array([
        [0.00, 0.50, 0.00, 0.7],  # dark green
        [0.00, 0.40, 0.20, 0.7],  # forest green
        [0.20, 0.50, 0.20, 0.7],  # medium green
        [0.00, 0.30, 0.00, 0.7],  # very dark green
        [0.15, 0.40, 0.15, 0.7],  # sage green
    ])
    
    noise_colors = plt.cm.Greys(np.linspace(0.6, 0.8, 10))      # light grays
    
    # No need for saturation enhancement since colors are already dark green
    
    # Randomize the order of colors
    np.random.seed(42)  # for reproducibility
    np.random.shuffle(vehicle_colors)
    np.random.shuffle(ped_colors)
    np.random.shuffle(noise_colors)
    
    for idx, ev in events_df.iterrows():
        mask = (
            (df['epoch'] >= ev['epoch']) &
            (df['epoch'] <= (ev['epoch'] + ev['duration'])) &  # Reconstruct end time
            (np.sign(df['kmh']) == ev['direction'])
        )
        seg = df[mask]
        times = [datetime.fromtimestamp(ts, tz=utc).astimezone(local_tz)
                 for ts in seg['epoch']]
        
        # Set color and style based on event type
        if ev['type'] == 1:  # Vehicle
            color = vehicle_colors[idx % len(vehicle_colors)]
            alpha = 0.6  # increased opacity
        elif ev['type'] == 0:  # Normal pedestrian
            color = ped_colors[idx % len(ped_colors)]
            alpha = 0.7  # increased opacity
        else:  # type == -1, low-V noise
            color = noise_colors[idx % len(noise_colors)]
            alpha = 0.2  # keep noise transparent
            
        direction_arrow = '→' if ev['direction'] > 0 else '←'
        if ev['type'] == 1:
            label = f"Vehicle {idx+1} ({direction_arrow})"
        elif ev['type'] == 0:
            label = f"Pedestrian {idx+1} ({direction_arrow})"
        else:
            label = f"Short Ped {idx+1} ({direction_arrow})"
            
        # Plot both lines and dots
        plt.plot(times, seg['kmh'], '-', color=color, alpha=alpha*0.7,  # line slightly more transparent
                 label=label, linewidth=1)
        plt.plot(times, seg['kmh'], 'o', color=color, alpha=alpha,  # dots more opaque
                 markersize=4, label='_nolegend_')  # _nolegend_ prevents duplicate labels

    plt.gcf().autofmt_xdate()
    
    # Calculate time range to determine appropriate format
    time_range = max(times) - min(times)
    if time_range.total_seconds() < 3600:  # less than 1 hour
        time_format = '%H:%M:%S'
    else:
        time_format = '%H:%M'
        
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter(time_format, tz=local_tz))
    
    plt.xlabel('Time')
    plt.ylabel('Speed (km/h)')
    plt.title(f'Road Traffic - {title}')
    plt.grid(True, alpha=0.3)
    if len(events_df) < 10:
        plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # adjust paths as needed
    fname1 = '20250529_0000_DpCh1.csv'
    fname2 = '20250529_0000_DpCh2.csv'
    # indir = r"/home/jbeale/Documents/doppler"
    indir = r"C:\Users\beale\Documents\doppler"
    
    doPlot = True  # set to False to skip plotting
    process_radar_file(fname1, indir, show_plot=doPlot)
    process_radar_file(fname2, indir, show_plot=doPlot)