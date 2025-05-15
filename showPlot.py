#!/home/john/anaconda3/envs/cv/bin/python
# python v3.9.7

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import matplotlib.dates as mdates

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

    # Plot
    plt.figure(figsize=(12, 6))
    plt.scatter([dt for i, dt in enumerate(datetimes) if not is_negative[i]],
            abs_speeds[~is_negative], color='#4088D0', label='Westbound', s=10)
    plt.scatter([dt for i, dt in enumerate(datetimes) if is_negative[i]],
            abs_speeds[is_negative], color='#60A040', label='Eastbound', s=10)

    plt.xlabel('Local Time (PDT)', fontsize=12)
    plt.ylabel('Speed (mph)', fontsize = 12)  # Changed from km/h to mph
    plt.title('Vehicle Speed vs Time')
    plt.legend()
    plt.grid(True)

    ax = plt.gca()

        # Create locator and formatter for dynamic tick spacing
    locator = mdates.AutoDateLocator(tz=local_tz)
    formatter = mdates.ConciseDateFormatter(locator, tz=local_tz)    
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.format_xdata = format_time_with_tenths
    
    # Get actual data time range instead of plot limits
    start_dt = min(datetimes)
    end_dt = max(datetimes)
    span_hours = (end_dt - start_dt).total_seconds() / 3600

    ax.tick_params(axis='x', labelrotation=0, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # Make y-axis numbers bold
    for label in ax.yaxis.get_ticklabels():
        label.set_fontweight('bold')
    
    # Add annotation text above plot area
    plt.text(0.02, 1.005, annotation_text,  # Increased y position from 1.02 to 1.05
         transform=plt.gca().transAxes,
         verticalalignment='bottom',  # Changed from 'top' to 'bottom'
         horizontalalignment='left',
         fontsize=10
         )

    # Adjust margins to make room for annotation
    plt.subplots_adjust(top=0.92)  # Add this line before tight_layout
    plt.tight_layout()
    plt.show()

# ========== Main Function ==========
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python showPlot.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    doPlot(filename)