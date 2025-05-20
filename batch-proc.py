# batch process radar data files

import os
import glob
import subprocess

# Directory to search
directory = r"C:\Users\beale\Documents\doppler"

# Pattern to match files
pattern = os.path.join(directory, '202505*_0000_DpCh1.csv')

# Find matching files
matching_files = glob.glob(pattern)

script_to_call = 'c:/Users/beale/Documents/Code/radar/getVehicles.py'
pycmd = r"C:/ProgramData/Anaconda3/envs/ds/python.exe"

# Call the script for each matching file
for file_path in matching_files:
    file = os.path.basename(file_path)
    # print(f"Processing file: {file}")
    subprocess.run([pycmd, script_to_call, '-n', file])
