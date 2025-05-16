import pandas as pd
from datetime import datetime, timezone
import pytz

# Parameters
SPEED_THRESHOLD = 20.0  # km/h qualifying speed
IGNORE_THRESHOLD = 8.0  # km/h ignore speeds below this
MIN_EVENT_LENGTH = 10   # samples
MAX_EVENT_LENGTH = 250  # samples
TIME_GAP = 0.9          # seconds between events

def findEvents(infile):
    # Read CSV with headers
    df = pd.read_csv(infile)

    # Convert speed to numeric and get absolute values
    df['abs_kmh'] = df['kmh'].abs()

    # Initialize state
    events = []
    in_event = False
    start_idx = None
    last_qualifying_time = None

    # Timezone setup
    utc = pytz.utc
    pdt = pytz.timezone('US/Pacific')

    for i, row in df.iterrows():
        current_time = row['epoch']
        
        #if row['abs_kmh'] == 0:
        #    continue                    
        

        if row['abs_kmh'] > SPEED_THRESHOLD:
            if not in_event or last_qualifying_time is None:
                in_event = True
                start_idx = i
                last_qualifying_time = current_time
                last_qualifying_index = i
            
            else: # we have been in an event
                # Check if we are still in the same event            
                if (current_time - last_qualifying_time) > TIME_GAP:
                    # End event due to time gap
                    end_idx = last_qualifying_index   # End at last qualifying measurement
                    length = end_idx - start_idx + 1
                    if MIN_EVENT_LENGTH <= length <= MAX_EVENT_LENGTH:
                        events.append((start_idx, end_idx))

                    # and immediately start new event
                    start_idx = i   
                    last_qualifying_time = current_time
                    last_qualifying_index = i
                else:
                    # Update last qualifying time
                    last_qualifying_time = current_time
                    last_qualifying_index = i

        else: # Speed below threshold
            if in_event:
                # Check if we are still in the same event
                if (current_time - last_qualifying_time) > TIME_GAP:
                    # End event due to time gap
                    in_event = False
                    end_idx = last_qualifying_index
                    length = end_idx - start_idx + 1
                    if MIN_EVENT_LENGTH <= length <= MAX_EVENT_LENGTH:
                        events.append((start_idx, end_idx))

    # Handle case where event goes till the end
    if in_event:
        end_idx = len(df) - 1
        length = end_idx - start_idx + 1
        if MIN_EVENT_LENGTH <= length <= MAX_EVENT_LENGTH:
            events.append((start_idx, end_idx))

    # Output events with duration, direction and PDT start time
    print("Detected vehicle events:")
    for start, end in events:
        start_time_utc = datetime.fromtimestamp(df.loc[start, 'epoch'], tz=utc)
        start_time_pdt = start_time_utc.astimezone(pdt)
        duration_sec = df.loc[end, 'epoch'] - df.loc[start, 'epoch']
        
        # Determine direction based on majority of speed signs in event
        event_speeds = df.loc[start:end, 'kmh']
        direction = "East" if event_speeds.mean() < 0 else "West"
        
        print(f"Start: {start_time_pdt.strftime('%Y-%m-%d %H:%M:%S %Z')}, "
              f"idx: {start}, "
              f"Dir: {direction}, "
              f"Dur: {duration_sec:.2f} sec, "
              f"Samples: {end - start + 1}")
    
    print("Total events detected:", len(events))        

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python vevents.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    
    # input_file = r"C:\Users\beale\Documents\doppler\20250515_clip.csv"
    findEvents(input_file)
