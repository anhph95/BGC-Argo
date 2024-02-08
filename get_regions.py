#!/usr/bin/env python3
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import regionmask
import geopandas as gp
import cartopy.crs as ccrs

# Ocean basins
def set_up_mask(regions, lon_start=-180, lon_end=180, lat_start=-90, lat_end=90, plot=True):
    region_ids = regions.region_ids
    lonrange = np.arange(lon_start,lon_end,0.25)
    latrange = np.arange(lat_start,lat_end,0.25)
    region_mask = regions.mask(lonrange,latrange)
    
    # Region code to full name dict
    region_code = defaultdict(list)
    for x,y in region_ids.items():
        region_code[y].append(x)    

    # Plot
    if plot:
        f, ax = plt.subplots(figsize=(15,15),subplot_kw=dict(projection=ccrs.PlateCarree()))
        region_mask.plot(cmap='tab20c',ax=ax,add_colorbar=False)
        regions.plot(label="abbrev",add_coastlines=True)
        lon_step = 60
        lon_ticks = np.arange(lon_start, lon_end + lon_step, lon_step)
        ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
        ax.set_xticklabels(lon_ticks)
        ax.set_xlabel('Longitude')

        # Add latitude tick labels
        lat_step = 30
        lat_ticks = np.arange(lat_start, lat_end + lat_step, lat_step)
        ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
        ax.set_yticklabels(lat_ticks)
        ax.set_ylabel('Latitude')
        ax.set_title(regions.name)

    return region_ids, region_mask, region_code

        
def get_regions(lon,lat,region_mask,region_code):
    rname = []
    rid = []
    rabbrev = []
    for x, y in zip(lon, lat):
        z = region_mask.sel(lon=x, lat=y, method='nearest').item()
        temp = region_code.get(z)
        if np.size(temp)==1:
            rid.append(None)
            rabbrev.append(None)
            rname.append(None)
        elif np.size(temp)==2:
            rid.append(temp[0])
            rabbrev.append(None)
            rname.append(temp[1])
        else:
            rid.append(temp[0])
            rabbrev.append(temp[1])
            rname.append(temp[2])
    return(rid, rabbrev, rname)

