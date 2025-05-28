#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
import pytz


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
            # Modify type assignment to include short pedestrians
            if speeds.mean() > 15.0:
                event_type = 1  # vehicle
            elif duration < 5.0:
                event_type = -1  # short pedestrian (noise)
            else:
                event_type = 0  # normal pedestrian
                
            events.append({
                'start_time': t0,
                'end_time':   t1,
                'duration':   duration,
                'avg_speed':  speeds.mean(),
                'max_speed':  speeds.max(),
                'direction':  int(np.sign(avg_vel)),
                'type':       event_type
            })

    return pd.DataFrame(events)


if __name__ == "__main__":
    # adjust paths as needed
    #fname = '20250528_0000_DpCh1.csv'
    fname = '20250528_0000_DpCh2.csv'
    # indir = r"C:\Users\beale\Documents\doppler"
    indir = r"/home/jbeale/Documents/doppler"
    inpath = os.path.join(indir, fname)
    outpath = os.path.join(indir, fname.replace('.csv', '_seg_events.csv'))

    # load and filter zeros
    df = pd.read_csv(inpath)
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

    if all_events:
        events_df = pd.concat(all_events, ignore_index=True)
        events_df = events_df.sort_values('start_time').reset_index(drop=True)
        events_df.to_csv(outpath, index=False)
        print(events_df)
        print(f"Detected {len(events_df)} motion events → {outpath}")

        # Print summary statistics
        n_pedestrians = len(events_df[events_df['type'] == 0])
        n_vehicles = len(events_df[events_df['type'] == 1])
        n_short_peds = len(events_df[events_df['type'] == -1])
        print(f"\nSummary:")
        print(f"Vehicles: {n_vehicles}")
        print(f"Pedestrians: {n_pedestrians}")
        print(f"Low-V noise: {n_short_peds}")
        print(f"Total events: {len(events_df)-n_short_peds}\n")

        # plot each event using only its segment points
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
                (df['epoch'] >= ev['start_time']) &
                (df['epoch'] <= ev['end_time']) &
                (np.sign(df['kmh']) == ev['direction']) &
                True
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
        plt.title(f'Motion Events (segmentation) - {fname}')
        plt.grid(True, alpha=0.3)
        if len(events_df) < 10:
            plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("No events detected.")
