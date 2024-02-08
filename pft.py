#!/usr/bin/env python3
import xarray as xr
import pandas as pd
import numpy as np
import multiprocessing


# Function to mine data
def PFT_matchup(df, pft, pft_list, grid_size=3, time_bound=12):
    for index, row in df.iterrows():
        try:
            # Get lon lat index of nearest pixel to target coordinate
            lon_idx = abs(pft.lon - row['Lon']).argmin().values
            lat_idx = abs(pft.lat - row['Lat']).argmin().values
            # Define the coordinate indices
            lat_start = lat_idx - (grid_size // 2)
            lat_end = lat_idx + (grid_size // 2) + 1
            lon_start = lon_idx - (grid_size // 2)
            lon_end = lon_idx + (grid_size // 2) + 1
            # Calculate time range
            date_range = slice(row['Date']-np.timedelta64(time_bound,'h'),row['Date']+np.timedelta64(time_bound,'h'))
            # Get 3x3 grid surround target coordianate, then get rolling n-day 
            pft_subset = pft.isel(lat=slice(lat_start,lat_end),lon=slice(lon_start,lon_end)).sel(time=date_range)
            # Median
            median = pft_subset.median(skipna=True)
            # Update to output table
            for i in pft_list:
                df.at[index,i] = median[i].values
        except Exception:
            print('{} error occurred'.format(row['Float_cycle']))
            pass
    return df

def parallelize_pft(func, df):
    num_cores = multiprocessing.cpu_count()-1  # leave one free to not freeze machine
    df_split = np.array_split(df, num_cores) # split dataframe into chunks
    pool = multiprocessing.Pool(num_cores) # number of pool
    df = pd.concat(pool.map(func, df_split)) # concatenate result
    pool.close()
    pool.join()
    return df