import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


csv_files = glob.glob(os.path.join('./data', '*.csv'))

# Initialize an empty list to store DataFrames
dfs = []

# Iterate over the list of CSV files
for file in csv_files:
    # Read each CSV file into a DataFrame
    df = pd.read_csv(file)
    # Append the DataFrame to the list
    dfs.append(df)

# Concatenate all DataFrames in the list into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Display the combined DataFrame
print(combined_df)