# This script processes two CSV files containing event data from two channels,
# finds matches based on directional criteria, and classifies matched and unmatched events.

import pandas as pd
import os

def load_data(ch1_file, ch2_file):
    """Load and preprocess data from two CSV files."""
    try:
        df_ch1 = pd.read_csv(ch1_file)
        df_ch2 = pd.read_csv(ch2_file)
        
        print(f"Channel 1: {len(df_ch1)} total events")
        print(f"Channel 2: {len(df_ch2)} total events")
        
        # Add event_id if not present
        for df in [df_ch1, df_ch2]:
            if 'event_id' not in df.columns:
                df.insert(0, 'event_id', range(1, len(df) + 1))
        
        # Filter to only event type 1
        df_ch1 = df_ch1[df_ch1['type'] == 1].copy()
        df_ch2 = df_ch2[df_ch2['type'] == 1].copy()
        
        print(f"Channel 1: {len(df_ch1)} vehicle events")
        print(f"Channel 2: {len(df_ch2)} vehicle events")
        
        # Compute end times
        df_ch1['end_epoch'] = df_ch1['epoch'] + df_ch1['duration']
        df_ch2['end_epoch'] = df_ch2['epoch'] + df_ch2['duration']
        
        # Renumber event_ids after filtering
        df_ch1['event_id'] = range(1, len(df_ch1) + 1)
        df_ch2['event_id'] = range(1, len(df_ch2) + 1)
        
        return df_ch1.reset_index(drop=True), df_ch2.reset_index(drop=True)
        
    except pd.errors.EmptyDataError:
        print(f"Error: One or both input files are empty")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def find_directional_matches(df_ch1, df_ch2, time_tolerance=4.5):
    matches = []

    for i, ch1_event in df_ch1.iterrows():
        for j, ch2_event in df_ch2.iterrows():
            if ch1_event['direction'] == 1 and ch2_event['direction'] == -1: # OK for some
                # Ch1 → Ch2: match ch1 end to ch2 start
                ch1_end = ch1_event['end_epoch']
                diff = (ch2_event['epoch'] - ch1_end)
                if diff <= time_tolerance and diff >= -2.0:
                    matches.append((i, j, diff))
            elif ch1_event['direction'] == -1 and ch2_event['direction'] == 1:
                # Ch2 → Ch1: match ch2 end to ch1 start
                ch2_end = ch2_event['end_epoch']
                diff = (ch1_event['epoch'] - ch2_end)
                if diff <= time_tolerance and diff >= -2.0:
                    matches.append((i, j, diff))

    # Sort by smallest time difference
    # matches.sort(key=lambda x: x[2])

    matched_ch1 = set()
    matched_ch2 = set()
    final_matches = []

    for i, j, diff in matches:
        if i not in matched_ch1 and j not in matched_ch2:
            final_matches.append((i, j, diff))
            matched_ch1.add(i)
            matched_ch2.add(j)

    return final_matches


def classify_matches(df_ch1, df_ch2, matches):
    matched_ch1_idx = {i for i, _, _ in matches}
    matched_ch2_idx = {j for _, j, _ in matches}

    # Create copies of the filtered DataFrames
    unmatched_ch1 = df_ch1.loc[~df_ch1.index.isin(matched_ch1_idx)].copy()
    unmatched_ch2 = df_ch2.loc[~df_ch2.index.isin(matched_ch2_idx)].copy()

    matched_rows = []
    for i, j, diff in matches:
        matched_rows.append({
            'ch1_event_id': df_ch1.loc[i, 'event_id'],
            'ch2_event_id': df_ch2.loc[j, 'event_id'],
            'time_diff_sec': diff,
            'ch1_speed_kmh': df_ch1.loc[i, 'max_speed'],    # Changed from smooth_max
            'ch2_speed_kmh': df_ch2.loc[j, 'max_speed'],    # Changed from smooth_max
            'ch1_time': df_ch1.loc[i, 'time']          
        })

    matched_df = pd.DataFrame(matched_rows)
    
    # Sort the DataFrames
    unmatched_ch1.sort_values(by='epoch', inplace=True)
    unmatched_ch2.sort_values(by='epoch', inplace=True)    
    matched_df.sort_values(by='ch1_time', inplace=True)
    
    return matched_df, unmatched_ch1, unmatched_ch2

def main():
    #directory = r"C:\Users\beale\Documents\doppler" 
    directory = r"/home/jbeale/Documents/doppler"
    ch1_file = "20250529_0000_DpCh1_summary.csv"
    ch2_file = "20250529_0000_DpCh2_summary.csv"

    path1 = os.path.join(directory, ch1_file)
    path2 = os.path.join(directory, ch2_file)

    df_ch1, df_ch2 = load_data(path1, path2)
    if df_ch1.empty or df_ch2.empty:
        print("Error loading data files. Exiting.")
        return
        
    matches = find_directional_matches(df_ch1, df_ch2)
    matched_df, unmatched_ch1, unmatched_ch2 = classify_matches(df_ch1, df_ch2, matches)

    # Save matched events to CSV
    outf = ch1_file[:8] + "_matched_events.csv"
    outPath = os.path.join(directory, outf)
    matched_df.to_csv(outPath, index=False)
    print(f"Matched events saved to {outPath}")
    
    # Print results
    print("Matched Event Pairs with Speeds (km/h):\n", matched_df)
    print("\nUnmatched Events from Ch1:\n", 
          unmatched_ch1[['event_id', 'time', 'duration', 'direction', 'max_speed']])  # Changed
    print("\nUnmatched Events from Ch2:\n", 
          unmatched_ch2[['event_id', 'time', 'duration', 'direction', 'max_speed']])  # Changed
    print(f"\nFound {len(matches)} matching event pairs")
    print(f"Unmatched in Ch1: {len(unmatched_ch1)}")
    print(f"Unmatched in Ch2: {len(unmatched_ch2)}")


if __name__ == '__main__':
    main()

