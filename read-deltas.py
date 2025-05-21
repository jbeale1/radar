# read delta-encoded CSV file and reconstruct the original values
# The first column is a timestamp, the second column is the first absolute reading,
# and the subsequent columns are delta-encoded values.

# J.Beale 2025-05-20    

import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from zoneinfo import ZoneInfo  # Python 3.9+ timezone handling
import matplotlib.dates as mdates

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

# Example usage:

indir=r"/home/john/Documents/source"
#fname = r"20250520-2207_adc1256-log2.csv"
#fname = r"20250520-2240_adc1256-log2.csv"
fname = r"20250520-2253_adc1256-log2.csv"
fpath = os.path.join(indir, fname)

timestamps, readings = decodeFile(fpath)
#print(list(zip(timestamps, readings)))

# Convert Unix timestamps to datetime objects with PDT timezone
pdt = ZoneInfo("America/Los_Angeles")
datetimes = [datetime.fromtimestamp(ts, tz=pdt) for ts in timestamps]

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(datetimes, readings, 'b-', linewidth=1, alpha=0.7)

# Format x-axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=pdt))
plt.gcf().autofmt_xdate()  # Angle and align the tick labels

# Labels and title
plt.xlabel('Time (PDT)')
plt.ylabel('Reading')
plt.title('Sensor Readings vs Time')
plt.grid(True, alpha=0.3)

# Show the plot
plt.tight_layout()
plt.show()