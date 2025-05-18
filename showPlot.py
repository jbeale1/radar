#!/home/john/anaconda3/envs/cv/bin/python
# python v3.9.7

# Display speeds, count pedestrians

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import matplotlib.dates as mdates
from scipy.stats import skew
from scipy.ndimage import binary_dilation

utc_tz = ZoneInfo("UTC")
local_tz = ZoneInfo("America/Los_Angeles")  # use local timezone. PST/PDT is automatic by date 

def format_time_with_tenths(x, pos=None):
    dt = mdates.num2date(x, tz=local_tz)
    return dt.strftime('%H:%M:%S.%f')[:-5]  # slice to keep only tenths

def filter_constant_groups(speeds, threshold=10.0, max_group_size=8):
    result = speeds.copy()
    n = len(speeds)
    i = 0
    
    while i < n:
        # Find run of identical values
        j = i + 1
        while j < n and speeds[j] == speeds[i]:
            j += 1
            
        group_size = j - i
        
        # Check if group meets criteria
        if group_size <= max_group_size and group_size > 1:
            # Check differences with neighbors
            prev_diff = abs(speeds[i-1] - speeds[i]) if i > 0 else 0
            next_diff = abs(speeds[j-1] - speeds[j]) if j < n else 0
            
            if prev_diff > threshold and next_diff > threshold:
                result[i:j] = 0
                
        i = j
        
    return result

def calculate_slow_rate(epochs, speeds, window_size=110, speed_threshold=5):
    """Calculate rate of slow readings per hour in sliding window.
    
    Args:
        epochs: array of Unix timestamps
        speeds: array of speeds in mph
        window_size: number of readings to look back
        speed_threshold: speed below which to count as 'slow' (mph)
    
    Returns:
        rates: array of hourly rates for each point
    """
    rates1 = np.zeros(len(speeds))
    rates2 = np.zeros(len(speeds))
    #skewness = rates.copy()

    for i in range(window_size, len(speeds), 10):
        window_start = i - window_size
        window_speeds = speeds[window_start:i]
        window_epochs = epochs[window_start:i]
        
        # Count slow readings in window (excluding zeros)
        #slow_count = np.sum((np.abs(window_speeds) < speed_threshold) & (window_speeds != 0))
        slow_count1 = np.sum((window_speeds < speed_threshold) & (window_speeds > 0)) # pos velocity
        slow_count2 = np.sum((-window_speeds < speed_threshold) & (-window_speeds > 0)) # neg velocity
        
        # Calculate time span in seconds
        time_span = (window_epochs[-1] - window_epochs[0])
        
        # Calculate hourly rate
        if time_span > 0:
            rates1[i] = slow_count1 / time_span
            rates2[i] = slow_count2 / time_span

            # skewness[i] = skew(np.abs(window_speeds) > 0)
            
    return rates1, rates2

def filter_close_events(events, datetimes, min_gap=12.0):
    """Filter out events that occur too close together in time.
    
    Args:
        events: array of 0s and 1s marking event locations
        datetimes: array of datetime objects corresponding to each event
        min_gap: minimum time in seconds required between events
    
    Returns:
        filtered events array with close events removed
    """
    filtered_events = events.copy()
    last_event_time = None
    
    for i in range(len(filtered_events)):
        if filtered_events[i] == 1:
            current_time = datetimes[i].timestamp()
            if last_event_time is not None:
                time_diff = current_time - last_event_time
                if time_diff < min_gap:  # too close to previous event
                    filtered_events[i] = 0
                    continue
            last_event_time = current_time
            
    return filtered_events

def doPlot(filename):
    # Load CSV file assuming it has a header with "epoch" and "kmh"
    data = np.genfromtxt(filename, delimiter=',', names=True)

    # Extract columns
    epochs = data['epoch']
    speeds = data['kmh']    
    speeds = speeds * 0.621371      # Convert km/h to mph

    # Filter out suspicious constant groups
    speeds = filter_constant_groups(speeds, threshold=6, max_group_size=8)
    
    # Determine absolute speed values
    abs_speeds = np.abs(speeds)

    # Identify where original speeds were negative
    is_negative = speeds < 0

    # Convert epochs (UTC) to local timezone datetime objects
    datetimes = [datetime.fromtimestamp(ts, tz=utc_tz).astimezone(local_tz) for ts in epochs]

    # Get date of first data point
    if datetimes:
        annotation_text = datetimes[0].strftime('%A, %Y-%m-%d')  # e.g., "Monday, 2025-05-12"
    else:
        annotation_text = 'No data'

    # Calculate slow reading rates using specified window sizes
    slow1, slow2 = calculate_slow_rate(epochs, speeds, window_size=150)    
    
    mask1 = (slow1 > 10).astype(int)    # select those areas likely to be oncoming pedestrians
    mask2 = (slow2 > 10).astype(int)    # select those areas likely to be departing pedestrians

    
    # Expand mask2 using binary dilation
    structure = np.ones(51)  #  points on each side plus center point
    mask1 = binary_dilation(mask1, structure=structure)
    mask2 = binary_dilation(mask2, structure=structure)
    
       # Find transitions from 0 to 1
    events1 = np.zeros_like(mask1)
    events1[1:] = (mask1[1:] > mask1[:-1]).astype(int)  # 1 where value increases
    events2 = np.zeros_like(mask2)
    events2[1:] = (mask2[1:] > mask2[:-1]).astype(int)  # 1 where value increases

    # Filter out events that are too close together
    events1 = filter_close_events(events1, datetimes, min_gap=12.0)
    events2 = filter_close_events(events2, datetimes, min_gap=12.0)

    # Create single plot
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot speeds
    ax1.scatter([dt for i, dt in enumerate(datetimes) if not is_negative[i]],
            abs_speeds[~is_negative], color='#4088D0', label='Westbound', s=10)
    ax1.scatter([dt for i, dt in enumerate(datetimes) if is_negative[i]],
            abs_speeds[is_negative], color='#60A040', label='Eastbound', s=10)
    
    # Add mask plots to same axis
    ax1.plot(datetimes, events1, color='blue', linestyle='-', alpha=0.5, label='Ped Coming')
    ax1.scatter(datetimes, events1, color='blue', s=10, alpha=0.8)
    ax1.plot(datetimes, events2, color='green', linestyle='-', alpha=0.5, label='Ped Going')
    ax1.scatter(datetimes, events2, color='green', s=10, alpha=0.8)
    
    ax1.set_xlabel('Local Time (PDT)', fontsize=12)
    ax1.set_ylabel('Speed (mph)', fontsize=12)
    ax1.set_title('Vehicle Speed vs Time')
    ax1.grid(True)
    ax1.legend()

    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=local_tz))
    ax1.format_xdata = format_time_with_tenths
    ax1.tick_params(axis='x', labelrotation=0, labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    for label in ax1.yaxis.get_ticklabels():
        label.set_fontweight('bold')
    
    # Add annotation text above plot
    ax1.text(0.02, 1.05, annotation_text,
            transform=ax1.transAxes,
            verticalalignment='bottom',
            horizontalalignment='left',
            fontsize=10)


    print("Pedestrians: coming %d, going %d" % (np.sum(events1), np.sum(events2)))

    plt.tight_layout()
    plt.show()

    
    #plt.plot(datetimes,mask2)
    #plt.show()

# ========== Main Function ==========
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python showPlot.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    doPlot(filename)