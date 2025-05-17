# combine csv files together 

import os
import pandas as pd

# Set the directory path containing your CSV files
directory_path = r"C:\Users\beale\Documents\doppler"

# Get list of .csv files in alphabetical order
csv_files = sorted([f for f in os.listdir(directory_path) 
                    if f.endswith('_SerialLog.csv')
                    and f.startswith('202505')])
                    
print("CSV files found:")
for filename in csv_files:
    print(filename)

# Read and concatenate CSVs
dataframes = []
for i, filename in enumerate(csv_files):
    file_path = os.path.join(directory_path, filename)
    if i == 0:
        # Read with header
        df = pd.read_csv(file_path)
    else:
        # Skip header row
        df = pd.read_csv(file_path, skiprows=1, header=None)
        df.columns = dataframes[0].columns  # assign same columns as first file
    dataframes.append(df)

# Concatenate all DataFrames
combined_df = pd.concat(dataframes, ignore_index=True)

# Write to output CSV
outpath = os.path.join(directory_path, 'combined_output.csv')

combined_df.to_csv(outpath, index=False)

print(f"Combined {len(csv_files)} files into 'combined_output.csv'")
