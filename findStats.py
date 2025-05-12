# Analyze data from speed radar
# Display vehicle & pedestrian stats
# Attempt to classify events (cars, people, rain)
# Python 3.9.7 (later versions do timezones differently)
# J.Beale 5/11/2025

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro, probplot
from scipy.ndimage import binary_dilation, label
# from filterpy.kalman import KalmanFilter
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from dataclasses import dataclass
from typing import List, NamedTuple
import logging
from radar_plots import RadarPlotter, PlotConfig
from plot_utils import get_spikes, moving_avg, clean_spikes, kalman_filter
# ------------------------------------------------------------

PDT = ZoneInfo("America/Los_Angeles")


# ------------------------------------------------------------
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)

class SpeedEvent(NamedTuple):
    start_time: float
    start_index: int
    end_index: int
    dir: int
    duration: float
    max: float
    avg: float
    amax: float
    amin: float

# ==============================================================

# ==============================================================

def count_people(times, speeds, speed_threshold=1.5, 
                max_abs_speed=20, min_group_size=150, time_gap=10, 
                plot=True):
    """
    Count people and rain events from radar data.
    
    Parameters:
    - times: np.ndarray of shape (N,) with Unix timestamps
    - speeds: np.ndarray of shape (N,) with Doppler speed measurements
    - speed_threshold: minimum absolute speed to consider valid movement (km/h)
    - max_abs_speed: maximum absolute speed for valid movement (km/h)
    - min_group_size: minimum number of points for a group to be valid
    - time_gap: max gap (in seconds) to treat as same event    
    - plot: whether to generate a plot of the data

    Returns:
    - slow_cars: list of dicts with car events
    - people: list of dicts with pedestrian events
    - rain: list of dicts with rain events
    """
        # Valid points
    #valid_mask = (np.abs(speeds) > speed_threshold) & (np.abs(speeds) < max_abs_speed)
    valid_mask = (np.abs(speeds) < max_abs_speed)
    valid_times = times[valid_mask]
    valid_speeds = speeds[valid_mask]

    # Grouping by time gaps
    time_diffs = np.diff(valid_times)
    group_breaks = np.where(time_diffs > time_gap)[0] + 1
    group_indices = np.split(np.arange(len(valid_times)), group_breaks)


    people = []
    slow_cars = []
    rain = []
    
    for idx_group in group_indices:
        if len(idx_group) < min_group_size:
            continue

        group_times = valid_times[idx_group]
        group_speeds = valid_speeds[idx_group]

        # Calculate zero ratio for this group
        zero_mask = np.abs(group_speeds) < 0.1  # Consider speeds near 0
        zero_ratio = np.sum(zero_mask) / len(group_speeds)
        # print(f"Zero ratio: {zero_ratio:.2f} for group of size {len(group_speeds)}")

        # Determine dominant direction
        num_positive = np.sum(group_speeds > 0)
        num_negative = np.sum(group_speeds < 0)
        if num_positive >= num_negative:
            dominant_mask = group_speeds >= 0
            direction = "towards"
        else:
            dominant_mask = group_speeds <= 0
            direction = "away"

        dominant_times = group_times[dominant_mask]
        dominant_speeds = group_speeds[dominant_mask]
        accel = np.diff(dominant_speeds)
        jerk = np.diff(accel)
        jerkstd = np.std(jerk)

        avg_velocity = np.mean(dominant_speeds)  
        max_speed_s = np.max(moving_avg(abs(dominant_speeds), 25))
        as_round = round(abs(avg_velocity), 2)       

        # Determine category including rain
        category = "unknown"
        # rain zero_ratio threshold should be lower, but that gives more false detects of rain
        if  ((zero_ratio >= 0.153)
            and (abs(avg_velocity) < 3.1)             
            and (jerkstd < 3.0)):
            category = "rain"
        elif (max_speed_s > 17) or (jerkstd < 0.36):    
            category = "car"
        elif (len(dominant_times) >= min_group_size) and (jerkstd >= 0.36):
            category = "person"

        event_data = {
            "start_time": dominant_times[0],
            "end_time": dominant_times[-1],
            "direction": direction,
            "avg_speed_kmh": as_round,
            "jerk_std": round(jerkstd, 2),
            "zero_ratio": round(zero_ratio, 3)
        }

        if category == "person":
            people.append(event_data)
        elif category == "car":
            slow_cars.append(event_data)
        elif category == "rain":
            rain.append(event_data)
            # print(f"Rain event detected: {event_data}")

    if plot:        
        # Convert timestamps to datetime objects using system timezone
        # Note: Python 3.9 fromtimestamp() uses local system timezone by default
        dt_times = [datetime.fromtimestamp(t) for t in times]
        
        # Plot all data
        plt.figure(figsize=(16, 6))
        plt.scatter(dt_times, speeds, color='green', s=10, label='Radar speed')
        plt.ylim(-20, 20)
        plt.xlabel("Time (PDT)")
        plt.ylabel("Speed (km/h)")
        
        # Format x-axis for better time display
        plt.gcf().autofmt_xdate()
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
        
        # Get the date for the title
        date_str = dt_times[0].strftime("%a %m/%d/%Y")
        plt.title(f"Radar Speeds with Detected Events - {date_str}")

        # Convert event timestamps with correct timezone
        for person in people:
            color = 'blue' if person['direction'] == 'towards' else 'red'
            start_dt = datetime.fromtimestamp(person['start_time'])
            end_dt = datetime.fromtimestamp(person['end_time'])
            plt.axvspan(start_dt, end_dt, color=color, alpha=0.2,
                      label=f"{person['direction'].capitalize()} (avg {person['avg_speed_kmh']} km/h)")

        # Also fix car and rain event timestamps
        for car in slow_cars:
            start_dt = datetime.fromtimestamp(car['start_time'])
            end_dt = datetime.fromtimestamp(car['end_time'])
            plt.axvspan(start_dt, end_dt, color='orange', alpha=0.2,
                        label=f"Car (avg {car['avg_speed_kmh']} km/h)")            

        for rain_event in rain:
            start_dt = datetime.fromtimestamp(rain_event['start_time'])
            end_dt = datetime.fromtimestamp(rain_event['end_time'])
            plt.axvspan(start_dt, end_dt, 
                    color='yellow', alpha=0.2,
                    label=f"Rain (ratio={rain_event['zero_ratio']:.2f})")
            
        # Remove duplicate legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.grid(True)
        plt.show()

    return slow_cars, people, rain

def count_vehicles(times, speeds, speed_threshold=20, 
                max_abs_speed=100, min_group_size=50, time_gap=1.5):
    """
    Count vehicles from doppler radar data
    
    Parameters:
    - times: np.ndarray of shape (N,) with Unix timestamps
    - speeds: np.ndarray of shape (N,) with Doppler speed measurements
    - speed_threshold: minimum absolute speed to consider valid movement (km/h)
    - max_abs_speed: maximum absolute speed for valid movement (km/h)
    - min_group_size: minimum number of points for a group to be valid
    - time_gap: max gap (in seconds) to treat as same event    

    Returns:
    - vehicles: list of dicts with vehicle events
    """
    # Valid points
    valid_mask = (np.abs(speeds) > speed_threshold) & (np.abs(speeds) < max_abs_speed)
    #valid_mask = (np.abs(speeds) < max_abs_speed)
    valid_times = times[valid_mask]
    valid_speeds = speeds[valid_mask]

    # Grouping by time gaps
    time_diffs = np.diff(valid_times)
    group_breaks = np.where(time_diffs > time_gap)[0] + 1
    group_indices = np.split(np.arange(len(valid_times)), group_breaks)

    vehicles = []  # list of vehicle events
    
    for idx_group in group_indices:
        if len(idx_group) < min_group_size:
            continue

        group_times = valid_times[idx_group]
        group_speeds = valid_speeds[idx_group]

        # Calculate zero ratio for this group
        zero_mask = np.abs(group_speeds) < 0.1  # Consider speeds near 0
        zero_ratio = np.sum(zero_mask) / len(group_speeds)
        # print(f"Zero ratio: {zero_ratio:.2f} for group of size {len(group_speeds)}")

        # Determine dominant direction
        num_positive = np.sum(group_speeds > 0)
        num_negative = np.sum(group_speeds < 0)
        if num_positive >= num_negative:
            dominant_mask = group_speeds >= 0
            direction = "towards"
        else:
            dominant_mask = group_speeds <= 0
            direction = "away"

        dominant_times = group_times[dominant_mask]
        dominant_speeds = group_speeds[dominant_mask]
        accel = np.diff(dominant_speeds)
        jerk = np.diff(accel)
        jerkstd = np.std(jerk)

        avg_velocity = np.mean(dominant_speeds)  
        max_speed_s = np.max(moving_avg(abs(dominant_speeds), 21))
        as_round = round(abs(avg_velocity), 2)       

        # Determine category including rain
        category = "unknown"

        if  ((zero_ratio < 0.2)
            and (abs(avg_velocity) > 20)             
            and (jerkstd < 3.0)):
            category = "vehicle"

        event_data = {
            "start_time": dominant_times[0],
            "end_time": dominant_times[-1],
            "direction": direction,
            "avg_speed_kmh": as_round,
            "jerk_std": round(jerkstd, 2),
            "zero_ratio": round(zero_ratio, 3)
        }

        if category == "vehicle":
            vehicles.append(event_data)

    return vehicles


def plot_hours(hour_counts, s, label):
    # Plot using matplotlib
    plt.figure(figsize=(10, 5))
    plt.bar(range(24), hour_counts, color='skyblue', edgecolor='black')
    plt.xlabel("hour of day (PDT)")
    plt.ylabel("%s count" % label)
    plt.title("%s per Hour  %s" % (label, s))
    plt.xticks(range(24))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    total_events = hour_counts.sum()
    plt.gca().text(
        0.98, 0.95, f"Total: {total_events}", 
        transform=plt.gca().transAxes, 
        ha='right', va='top', 
        fontsize=12, fontweight='normal',
        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5')
    )

    plt.tight_layout()
    plt.show()

# Extract hour-of-day (0–23) directly from datetime
def hour_count(event_times, label):
    # Convert to datetime and extract hours
    hours = np.array([datetime.fromtimestamp(ts, tz=PDT).hour for ts in event_times])
    # Count how many events fall into each hour
    hour_counts = np.bincount(hours, minlength=24)

    dt = datetime.fromtimestamp(event_times[0], tz=PDT)    
    date_string = dt.strftime("%a %#m/%#d/%y")

    plot_hours(hour_counts, date_string, label)
    peak_hour = np.argmax(hour_counts)
    peak_count = hour_counts[peak_hour]
    total = hour_counts.sum()
    print("Peak activity at hour %02d with %d %s (%.1f%% of total)" % 
          (peak_hour, peak_count, label, 100.0 * peak_count/total ))

    #print("Hours, traffic for %s" % date_string)
    #for i in range(24):
    #    print(i, hour_counts[i])


#  return indices where discrete difference is above threshold, or empty array
def get_spikes(arr, threshold):
    diffs = np.diff(arr)
    indices = np.where(np.abs(diffs) > threshold)[0]
    max = np.max(diffs)
    return indices, max

# replace element with average of N elements
def moving_avg(arr, N):
    if N % 2 == 0:
        raise ValueError("N should be an odd number for symmetric averaging")

    pad_width = N // 2
    padded = np.pad(arr, pad_width, mode='edge')
    kernel = np.ones(N) / N

    # Convolve with 'valid' to get the same size as input
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed

# remove spikes faster than a_max in m/s^2
def clean_spikes(speed_kmh, a_max = 5):
    speed_ms = speed_kmh / 3.6   # convert to m/s
    dt = 0.09  # 90 ms
    acceleration = np.diff(speed_ms) / dt
    acceleration = np.insert(acceleration, 0, 0)
    # print(acceleration)
    valid_mask = np.abs(acceleration) <= a_max
    expanded_mask = binary_dilation(valid_mask, structure=np.ones(3))
    cleaned_speed_ms = speed_ms.copy()
    # Use linear interpolation for invalid segments
    invalid_indices = np.where(~expanded_mask)[0]
    valid_indices = np.where(expanded_mask)[0]
    cleaned_speed_ms[invalid_indices] = np.interp(invalid_indices, valid_indices, cleaned_speed_ms[valid_indices])
    cleaned_speed_kmh = cleaned_speed_ms * 3.6
    return cleaned_speed_kmh


def find_stable_segment(speeds: np.ndarray, threshold: float) -> tuple[int, int]:
    """Find the longest segment without speed jumps larger than threshold."""

    speeds = np.asarray(speeds)
    if len(speeds) < 2:
        return (0, len(speeds))  # Edge case

    diffs = np.abs(np.diff(speeds))
    breakpoints = np.where(diffs > threshold)[0]
    segment_ends = np.concatenate(([ -1 ], breakpoints, [ len(speeds) - 1 ]))
    starts = segment_ends[:-1] + 1
    ends = segment_ends[1:] + 1
    lengths = ends - starts
    max_idx = np.argmax(lengths)
    return starts[max_idx], ends[max_idx]


# plot speed of fastest vehicle in dataset    
def doPlotOne(times, speeds):    
    plt.title("Vehicle event")    
    plt.scatter(times, speeds, s=4)
    plt.ylabel('speed, km/h')
    plt.xlabel('sample number')
    plt.grid('both')
    plt.show(block=False)


def find_groups_df(
        raw_data: pd.DataFrame,
        speed_threshold: float = 20.0,
        min_duration: float = 2.5
        ) -> pd.DataFrame:
    """
    Find and analyze groups of speed measurements that represent vehicle passes.

    Args:
        raw_data: DataFrame with 'kmh' and 'epoch' columns
        speed_threshold: Minimum speed to consider (km/h)
        min_duration: Minimum duration of event (seconds)

    Returns:
        DataFrame containing analyzed vehicle events with columns:
        - start_time: Event start time (epoch seconds)
        - start_index/end_index: Data indices
        - direction: Movement direction (-1 or 1)
        - duration: Event duration (seconds)
        - max/avg: Maximum and average speeds
        - amax/amin: Maximum and minimum accelerations
    """
    try:
        speeds = raw_data['kmh'].to_numpy()
        times = raw_data['epoch'].to_numpy()
    except KeyError as e:
        logger.error(f"Missing required column: {e}")
        raise ValueError("Input DataFrame must have 'kmh' and 'epoch' columns")

    # Find continuous segments above threshold
    speed_mask = np.abs(speeds) > speed_threshold
    padded = np.pad(speed_mask.astype(int), (1, 1), constant_values=0)
    transitions = np.diff(padded)
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]

    logger.info(f"Found {len(starts)} potential vehicle events")
    events: List[SpeedEvent] = []

    for start, end in zip(starts, ends):
        start_time = times[start]
        end_time = times[end-1]
        duration = end_time - start_time

        if duration < min_duration:
            continue

        # Find stable segment within group
        group = speeds[start:end]
        stable_start, stable_end = find_stable_segment(group, 5)
        stable_start += start
        stable_end += start

        # Recompute timing for stable segment
        start_time = times[stable_start]
        end_time = times[stable_end-1]
        duration = end_time - start_time

        if duration < min_duration:
            continue

        # Analyze stable segment
        stable_speeds = speeds[stable_start:stable_end]
        direction = int(np.sign(np.mean(stable_speeds)))
        
        # Apply filters and smoothing
        filtered_speeds = kalman_filter(stable_speeds, direction)
        smoothed = moving_avg(filtered_speeds, 7)
        abs_speeds = np.abs(smoothed)

        # Calculate acceleration
        accel = np.diff(abs_speeds)/(3.6 * 0.09)  # Convert to m/s^2
        accel_smooth = moving_avg(accel, 21)

        # Trim acceleration data based on direction
        size = len(abs_speeds)
        if direction > 0:
            accel_trim = accel_smooth[0:int(size*0.7)]
        else:
            accel_trim = accel_smooth[int(size*0.3):]

        events.append(SpeedEvent(
            start_time=start_time,
            start_index=stable_start,
            end_index=stable_end,
            dir=direction,
            duration=duration,
            max=abs_speeds.max(),
            avg=abs_speeds.mean(),
            amax=accel_trim.max(),
            amin=accel_trim.min()
        ))

    if not events:
        logger.warning("No valid vehicle events found")
        return pd.DataFrame()

    return pd.DataFrame([event._asdict() for event in events])

# display Q-Q plot
def showQQ(dfg, dfg1, dfRaw):
    speeds = dfg1['max'].to_numpy()
    stat, p = shapiro(speeds)
    print("Shapiro-Wilk statistic = %.4f, p-value = %0.3e" % (stat, p))

    count = len(dfg)
    firstDate = dfg['datetime'].iloc[0][:-6] # only hh:mm
    epoch0 = dfRaw['epoch'].iloc[0]
    epoch1 = dfRaw['epoch'].iloc[-1]
    dur = (epoch1 - epoch0)/(60*60.0) # duration in hours
    probplot(speeds, dist="norm", plot=plt)
    title = ("Probability Plot   [%d in %.1f h] %s" % (count, dur, firstDate))
    plt.title(title)
    plt.ylabel('speed, km/h')
    plt.grid('both')
    plt.show()


# plot speed of fastest vehicle in dataset    
def showFast(speeds, dir, label):    
    plt.title("Fastest vehicle  %s" % label)    
    plt.plot(speeds, 'x')
    cleaner = kalman_filter(speeds, dir)    
    smooth = moving_avg(cleaner, 7)
    plt.plot(smooth, linewidth = 1, color='#40B000')
    # plt.plot(cleaner, linewidth = 1, color='#B04000')
    plt.ylabel('speed, km/h')
    plt.xlabel('sample number')
    plt.grid('both')
    plt.show()

def showStats(note, dfg):
    print("%s events: %d " % (note,len(dfg)),end="")
    going_left = (dfg['dir'] > 0).sum()
    going_right = (dfg['dir'] < 0).sum()
    print("Left: %d  Right: %d" % (going_left, going_right))
    print("Max speed Avg: %.2f std: %.2f" % ( dfg['max'].mean(), dfg['max'].std() ) )
    print("Avg speed Avg: %.2f std: %.2f" % ( dfg['avg'].mean(), dfg['avg'].std() ) )
    print("Duration Avg: %.2f std: %.2f" % ( dfg['duration'].mean(), dfg['duration'].std() ) )
    index_max = dfg['max'].idxmax()
    kmh_max = dfg.at[index_max, 'max']
    duration = dfg.at[index_max, 'duration']
    avg = dfg.at[index_max, 'avg']
    kmh_max = dfg['max'].max()
    mph_max = 0.621371 * kmh_max
    print("Max: %.2f km/h (%.2f mph) %.1f avg %.1f sec" % 
          (kmh_max, mph_max, avg, duration), end='')

    dtime = dfg.at[index_max,'datetime']
    print("  at %s PDT  dir: %d" % (dtime, dfg.at[index_max,'dir']))
    

# Plot histogram ===================
def doHistPlot(dfg1):
    plt.hist(dfg1['max'], bins=12, range=(20, 80), edgecolor='black')
    plt.xlabel('km/h')
    plt.ylabel('events')
    plt.title('Speeds in km/h  '+hr_string+lastDate )
    plt.grid('both')
    plt.show()

def summarize_slow_events(events, label):
    if not events:
        print("No %s detected." % label)
        return

    speeds = [p["avg_speed_kmh"] for p in events]
    times =  [p["start_time"] for p in events]
    num_events = len(events)
    avg_speed = np.mean(speeds)
    min_speed = np.min(speeds)
    max_speed = np.max(speeds)

    print(f"%s: %d  Avg: %.2f km/h  Min/Max: %.2f / %.2f" % 
          (label, num_events, avg_speed, min_speed, max_speed))
    
    plt.hist(speeds) # display plots of speeds
    plt.title("Histogram of speeds for %s" % label)
    plt.show()

    hour_count(times, label) # display plot by hour

# ===============================================================
def main():

    # input data file from radar
    in_dir = r"C:\Users\beale\Documents\doppler"
    #in_dir = "/home/john/Documents/doppler"

    #fname = r"20250504_202105_SerialLog.csv"
    fname = r"20250505_212916_SerialLog.csv"
    #fname = r"20250506_220329_SerialLog.csv"
    #fname = r"20250507_222129_SerialLog.csv" # 9 total
    #fname = r"20250508_000003_SerialLog.csv"
    #fname = r"20250509_000003_SerialLog.csv"
    #fname = r"20250510_000003_SerialLog.csv"
    #fname = r"20250511_000003_SerialLog.csv"

    # Configure which plots to show
    plot_config = PlotConfig(        
        show_speed_hist= True,
        show_hourly_vehicles= True,
        show_hourly_slow_cars= True,
        show_hourly_people= True,
        show_hourly_rain= True,
        show_qq= True,
        show_fastest= True,
        show_median= True
       )    


    in_path = os.path.join(in_dir, fname)
    if not os.path.exists(in_path):
        print("File not found: %s" % in_path)
        return
    
    dfRaw = pd.read_csv(in_path)    # read CSV file

    speed_kmh = dfRaw['kmh'].to_numpy()  # get just the speeds
    epoch = dfRaw['epoch'].to_numpy()  # get just the seconds timestamp

    dur = (epoch[-1] - epoch[0])/(60*60.0) # duration in hours
    hr_string = ("%s   %.1f hours " % (fname,dur))
    print(hr_string)

    speed_min_threshold = 20.0 # threshold in km/h for interesting event
    duration_min_threshold = 2.5 # duration in seconds
    # result = get_groups(kmh_speed, T, N)

    fraction = np.mean(np.abs(speed_kmh) < speed_min_threshold)

    #print("File: %s" % fname)
    print("Readings: %d  frac below %.1f: %.3f" % (len(dfRaw),speed_min_threshold,fraction))

    #slow_cars, people = count_people(epoch, speed_kmh, plot = False)
    #slow_cars, people = count_people(epoch[4000:10000], speed_kmh[4000:10000], plot = True)
    slow_cars, people, rain = count_people(epoch, speed_kmh,                                      
                                        plot=False)

    vehicles = count_vehicles(epoch, speed_kmh)

    #summarize_slow_events(people, "People")
    #summarize_slow_events(slow_cars, "Slow cars")
    #summarize_slow_events(rain, "Rain events")

    # dfg1 contains valid vehicle events
    dfg1 = find_groups_df(dfRaw, speed_min_threshold, duration_min_threshold)
    pd.set_option('display.max_columns', None)

    dfg1['datetime'] = pd.to_datetime(dfg1['start_time'], unit='s', utc=True
            ).dt.tz_convert('US/Pacific').dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    dfg1['datetime'] = dfg1['datetime'].str[:-4] # remove excess digits

    lastDate = dfg1['datetime'].iloc[-1]
    firstDate = dfg1['datetime'].iloc[0]
    firstDateStr = "Start: " +str(firstDate)[0:-3]
    print("%s" % firstDateStr)
    print("Last event: %s" % lastDate)
    # print(dfg1)
    pd.set_option('display.float_format', '{:.2f}'.format)
    #print(dfg1[['datetime', 'dir', 'duration', 'max', 'avg']])

    showStats("Summary", dfg1)

    # Find max speed event
    index_max = dfg1['max'].idxmax()
    start = dfg1.at[index_max,'start_index']
    end = dfg1.at[index_max,'end_index']
    fast_data = np.abs(speed_kmh[start:end])

    # Find median speed event
    median_speed = dfg1['max'].median()
    index_median = (dfg1['max'] - median_speed).abs().idxmin()
    start_median = dfg1.at[index_median,'start_index']
    end_median = dfg1.at[index_median,'end_index']
    median_data = np.abs(speed_kmh[start_median:end_median])

    parts = fname.split("_")
    outname = parts[0]+"_CarSpeeds.csv"
    outPath = os.path.join(in_dir, outname)
    dfg1.to_csv(outPath, index=False, float_format='%.2f')  # save vehicle events to CSV

  
    # Generate all enabled plots
    plotter = RadarPlotter(plot_config)
    plotter.setup_plots()

    plotter.plot_speed_histogram(dfg1, firstDate, lastDate, hr_string) # display histogram of speeds
    plotter.plot_qq(dfg1, dfRaw) # display Probability (~ Quantile-Quantile) plot

    plotter.plot_hourly(dfg1, "vehicles")
    plotter.plot_hourly(slow_cars, "slow_cars")
    plotter.plot_hourly(people, "people")
    plotter.plot_hourly(rain, "rain")

    dir = dfg1.at[index_max,'dir']
    dtime = dfg1.at[index_max,'datetime']
    label = "[%d] %s PDT" % (dir, dtime)
    plotter.plot_fastest(fast_data, dir, label)

    dir = dfg1.at[index_median,'dir']
    dtime = dfg1.at[index_median,'datetime']
    label = "[%d] %s PDT" % (dir, dtime)
    plotter.plot_median(median_data, dir, label)
    
    # Show all plots together
    plotter.show()

    print("Peak accel: %.2f  %.2f m/s^2" % (dfg1['amax'].max(), dfg1['amin'].min()))
    print("# ==================================================")


# ==================================================
if __name__ == "__main__":
    main()
