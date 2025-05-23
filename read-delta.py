# read delta-encoded CSV file and reconstruct the original values
# The first column is a timestamp, the second column is the first absolute reading,
# and the subsequent columns are delta-encoded values.
# Display the data in a plot with interactive legend.

# J.Beale 2025-05-21

import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from zoneinfo import ZoneInfo  # Python 3.9+ timezone handling
import matplotlib.dates as mdates
from scipy import signal

def decodeFile(filepath):
    # Read the CSV, skipping comments and header
    df = pd.read_csv(filepath, header=None, skiprows=1, comment='#')

    data = df.to_numpy()
    readings = []
    timestamps = []

    for i in range(len(data)):
        row = data[i]
        current_time = float(row[0])
        abs_reading = int(row[1])
        deltas = row[2:]
        num_readings = 1 + len(deltas)  # 1 absolute + len(deltas)

        # Reconstruct readings for this line
        current_readings = [abs_reading]
        last_value = abs_reading
        for delta in deltas:
            last_value += int(delta)
            current_readings.append(last_value)

        # Compute time step for this line
        if i < len(data) - 1:
            next_time = float(data[i + 1][0])
            time_step = (next_time - current_time) / num_readings
        else:
            # For last line, reuse previous interval if possible
            if i > 0:
                prev_time = float(data[i - 1][0])
                prev_num_readings = 1 + len(data[i - 1][2:])
                time_step = (current_time - prev_time) / prev_num_readings
            else:
                time_step = 1.0  # Fallback if only one line

        # Interpolated timestamps
        interpolated_times = [current_time + j * time_step for j in range(num_readings)]

        readings.extend(current_readings)
        timestamps.extend(interpolated_times)

    return np.array(timestamps), np.array(readings, dtype=int)

def apply_lowpass_filter(data, sample_rate=60.0, cutoff_freq=1.0, poles=1):
    """Apply n-pole Butterworth low-pass filter to data.
    
    Args:
        data: numpy array of readings
        sample_rate: sampling frequency in Hz
        cutoff_freq: filter cutoff frequency in Hz
    
    Returns:
        filtered_data: numpy array of filtered readings
    """
    nyquist = sample_rate / 2
    normalized_cutoff_freq = cutoff_freq / nyquist
    b, a = signal.butter(poles, normalized_cutoff_freq, btype='low')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def smooth3(data):
    """Apply 3-point smoothing with reflective boundaries using vector operations.
    
    Args:
        data: numpy array of readings
    
    Returns:
        smoothed: numpy array of same length as input, with 3-point weighted average
    """
    kernel = np.array([0.25, 0.5, 0.25])
    padded = np.pad(data, (1, 1), mode='reflect')
    return np.convolve(padded, kernel, mode='valid')

def integrate_with_drift_control(data, sample_rate=60.0, drift_time=120.0):
    """Integrate data with drift control using 1-pole highpass filter.
    
    Args:
        data: numpy array of readings
        sample_rate: sampling frequency in Hz
        drift_time: highpass filter time constant in seconds
    
    Returns:
        integrated_data: numpy array of integrated readings with drift control
    """
    # First do the integration (cumulative sum / sample rate)
    integrated = np.cumsum(data) / sample_rate
    
    # Apply 1-pole highpass to remove drift
    nyquist = sample_rate / 2
    highpass_freq = 1.0 / drift_time  # Convert time constant to frequency
    normalized_freq = highpass_freq / nyquist
    b, a = signal.butter(1, normalized_freq, btype='high')
    filtered_data = signal.filtfilt(b, a, integrated)
    
    return filtered_data

# =============================================
# Main script
# =============================================

#indir=r"/home/john/Documents/source"
indir = r"C:\Users\beale\Documents\Tiltmeter"

fname = r"20250523-0846_adc1256-log2.csv"

fpath = os.path.join(indir, fname)

timestamps, readings = decodeFile(fpath)

# Convert Unix timestamps to datetime objects with PDT timezone
pdt = ZoneInfo("America/Los_Angeles")
datetimes = [datetime.fromtimestamp(ts, tz=pdt) for ts in timestamps]

# Apply filters before plotting
filtered_readings_1hz = apply_lowpass_filter(readings, cutoff_freq=1.0)
filtered_readings_01hz = apply_lowpass_filter(readings, cutoff_freq=0.1)
integ = smooth3(integrate_with_drift_control(readings))
integ = apply_lowpass_filter(integ, cutoff_freq=10.0)

plt.figure(figsize=(12, 6))

# Plot raw and filtered data
raw_line = plt.plot(datetimes, readings/100.0, 'b-', linewidth=1, alpha=0.3, label='Raw')[0]
#filtered_line_1hz = plt.plot(datetimes, filtered_readings_1hz, 'y-', linewidth=1, alpha=0.7, label='1 Hz LP')[0]
#filtered_line_01hz = plt.plot(datetimes, filtered_readings_01hz, 'g-', linewidth=1, alpha=0.7, label='0.1 Hz LP')[0]
integrated_line = plt.plot(datetimes, integ, 'g-', linewidth=1, alpha=0.7, label='Integrated')[0]

# Create interactive legend
leg = plt.legend(loc='upper right', framealpha=0.8)

# Define legend click handler
def toggle_lines(event):
    if event.artist in leg.get_lines():
        line = event.artist
        if line.get_label() == 'Raw':
            origline = raw_line
        elif line.get_label() == '1 Hz LP':
            origline = filtered_line_1hz
        elif line.get_label() == '0.1 Hz LP':
            origline = filtered_line_01hz
        else:
            origline = integrated_line
        visible = not origline.get_visible()
        origline.set_visible(visible)
        # Change alpha of legend line to indicate visibility
        line.set_alpha(0.7 if visible else 0.2)
        plt.draw()

# Make legend interactive
plt.gcf().canvas.mpl_connect('pick_event', toggle_lines)
leg.set_draggable(True)  # Allow legend to be moved
for legline in leg.get_lines():
    legline.set_picker(True)  # Enable picking on legend lines
    legline.set_pickradius(5)  # Make it easier to click

# Set up dynamic locator that adjusts with zoom level
locator = mdates.AutoDateLocator(interval_multiples=True)
formatter = mdates.ConciseDateFormatter(locator, tz=pdt)

# Custom formatter to show appropriate time precision based on zoom level
@plt.FuncFormatter
def custom_formatter(x, p):
    dt = mdates.num2date(x, tz=pdt)
    ax = plt.gca()
    time_range = np.diff(ax.get_xlim())[0]  # Get current x-axis span in days
    seconds_range = time_range * 24 * 3600
    
    if seconds_range > 600:  # More than 10 minutes shown
        return dt.strftime('%H:%M')
    elif seconds_range > 60:  # Between 60 seconds and 10 minutes
        return dt.strftime('%H:%M:%S')
    else:  # Less than 60 seconds shown
        return dt.strftime('%H:%M:%S.%f')[:-4]  # Show 2 decimal places

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(custom_formatter)
plt.gcf().autofmt_xdate()

# Enable mouse zoom with dynamic tick adjustment
ax.callbacks.connect('xlim_changed', lambda ax: plt.draw())

# Labels and title
plt.xlabel('Time (PDT)')
plt.ylabel('Reading')
plt.title('Sensor Readings vs Time')
plt.grid(True, alpha=0.3)

# Show the plot
plt.tight_layout()
plt.show()