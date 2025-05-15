import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
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
    
    # Filter out suspicious constant groups
    speeds = filter_constant_groups(speeds, threshold=10.0, max_group_size=8)
    
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
            abs_speeds[~is_negative], color='green', label='Westbound', s=10)
    plt.scatter([dt for i, dt in enumerate(datetimes) if is_negative[i]],
            abs_speeds[is_negative], color='grey', label='Eastbound', s=10)

    plt.xlabel('Local Time (PDT)')
    plt.ylabel('Speed (km/h)')
    plt.title('Vehicle Speed vs Time')
    plt.legend()
    plt.grid(True)


    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=local_tz))    
    ax.format_xdata = format_time_with_tenths
    ax.tick_params(axis='x', labelrotation=0)

    plt.text(0.02, 1.02, annotation_text,
         transform=plt.gca().transAxes,
         verticalalignment='top',
         horizontalalignment='left',
         fontsize=10,
         bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1'))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python showPlot.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    doPlot(filename)