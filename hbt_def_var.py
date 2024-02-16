#!/usr/bin/env python3

import multiprocessing
import os
import gsw
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import  find_peaks
from scipy.integrate import cumtrapz



# Function to standardize vector
def standardize(ds):
    mean = np.nanmean(ds)
    std = np.nanstd(ds)
    ds_std = (ds - mean) / std
    #ds_std = ds_std - np.nanmin(ds_std) 
    return ds_std

def rolling_average(x, y, interval,by_mean=False):
    x=np.array(x)
    y=np.array(y) 
    rolling_avg = np.empty_like(y, dtype=np.float64)
    for idx, depth in enumerate(y):
        mask = (y >= (depth - interval / 2)) & (y <= (depth + interval / 2))
        if np.isnan(x[mask]).all(): rolling_avg[idx] = np.nan
        else:
            if by_mean: rolling_avg[idx] = np.nanmean(x[mask])
            else: rolling_avg[idx] = np.nanmedian(x[mask])
    return rolling_avg

def interpolate(x, x1, y1, x2, y2):
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)

# def euphotic_depth(chl, depth):
#     chl_tot = 0
#     for i in range(1, np.size(depth)): 
#         # Update chl_tot using the trapezoid rule
#         chl_tot += 0.5 * (chl[i] + chl[i-1]) * (depth[i] - depth[i-1])
#         # Compute zeu based on current chl_tot
#         zeu = 912.5 * chl_tot ** -0.839
#         # Stop if zeu lower than integrated depth, use other equation if zeu higher than 102
#         if zeu < depth[i]:
#             if zeu < 102:
#                 return zeu
#             else:
#                 return 426.3 * chl_tot ** -0.547

def euphotic_depth(chl_a, depths):
    chl_a += 1e-10
    # Integrate Chl-a profile
    chl_tot = cumtrapz(chl_a, depths, initial=0)
    # Calculate Zeu using vectorized operations
    zeu_est_a1 = 912.5 * (chl_tot ** -0.839)
    zeu_est_a2 = 426.3 * (chl_tot ** -0.547)
    zeu_est = np.where(zeu_est_a1 < 102, zeu_est_a1, zeu_est_a2)
    # Find the crossover point
    crossover_index = np.where(zeu_est <= depths)[0][0]
    # Interpolate to find exact Zeu
    zeu_exact = np.interp(depths[crossover_index], [depths[crossover_index], depths[crossover_index+1]], [zeu_est[crossover_index], zeu_est[crossover_index+1]])
    return zeu_exact

# Function to calculate habitat-defining variable
def def_var(mydata, data_mode=['D','A'], flag=[1,2,5,8], var_adj=True, depth_top=10, depth_bottom=200,
            N2_local_max=1, d_sigma=0.03, d_theta=0.8,
            smooth=True,smooth_window=10, by_mean=False, peak_height=1,
            CHLA_threshold=0.01,
            dNITRATE_ref=1, NAI_boundary=1,
            plot=False, plot_extra=False, zmax=500, verbose=False):
    '''
    Function to calculate habitat defining variable of a single float cycle from BGC-ARGO
    Input:
    - mydata: input BGC-ARGO data download by argo_download.py
    - data_mode: list of data mode to be included in calculation
    - flag: list of quality control flag to be included in calculation
    - var_adj: whether adjusted data should be used
    - dec: number of decimal digits for result rounding
    - N2.smooth: option to apply smoothing function on buoyancy frequency (N2)
    - N2_local_max: n-th local maximum from surface, set to FALSE or 0 for global maximum
    - CHLA_smooth: option to apply smoothing function on chl a
    - CHLA_local_max: n-th local maximum from surface, set to FALSE or 0 for global maximum
    - dNITRATE_smooth: option to apply smoothing function on nitrate derivative
    - dNITRATE_local_max: select local maximum from surface, set to FALSE or 0 for global maximum
    - smooth_window: window size for savgol_filter, default is 10
    - smooth_order: order for savgol_filter, default is 2
    - peak_promince: prominence for find_peaks, default is 1
    - NAI.bound: boundary for NAI calculation, default is 2 uM as in Weber et al. 2019
    - plot: option to plot water column profiles, default is FALSE
    - zmax: depth limit for plotting, default is 500m
    '''
    # ============ Float basic info ============
    result = {}
    result['Float'] = mydata.PLATFORM_NUMBER.astype(str).str.replace(r'\s', '').item()
    result['N_PROF'] = mydata.N_PROF.item()
    result['Cycle'] = mydata.CYCLE_NUMBER.astype(int).item()
    result['Float_Cycle'] = '_'.join((str(result['Float']), str(result['Cycle'])))
    result['Date'] = mydata.JULD.dt.strftime('%m/%d/%Y %H:%M:%S').item()
    result['Year'] = mydata.JULD.dt.year.item()
    result['Month'] = mydata.JULD.dt.month.item()
    result['Hour'] = mydata.JULD.dt.hour.item()
    result['Season'] = mydata.JULD.dt.season.item()
    result['Daynight'] = 'Night' if result['Hour'] < 6 or result['Hour'] > 18 else 'Day'
    result['Lat'] = mydata.LATITUDE.item()
    result['Lon'] = mydata.LONGITUDE.item()
    result['MLD'] = np.nan # Mixed layer depth based on delta sigma-theta = 0.03
    result['DCM'] = np.nan # Depth of chlorophyll maximum
    result['SST'] = np.nan # Sea surface temperature
    result['SSS'] = np.nan # Sea surface salinity
    result['DNC'] = np.nan # Depth of nitracline based on delta nitrate of 1 umol L-1
    result['NAI'] = np.nan # Nitrate availability index
    result['Zeu'] = np.nan # Euphotic depth
    result['Zpd'] = np.nan # First optical depth
    result['Zlow'] = np.nan # Chlorophyll penentration depth
    result['CHL_surface'] = np.nan # Surface chlorophyll
    result['CHL_max'] = np.nan # Maximum chlorophyll 
    result['CHL_peak'] = np.nan # Depth-integrated chlorophyll for the peak of chlorophyll maxima 
    result['CHL_Zeu'] = np.nan # Depth-integrated chlorophyll for euphotic depth
    result['CHL_Zpd'] = np.nan # Depth-integrated chlorophyll for first optical depth
    result['CHL_Zlow'] = np.nan # Depth-integrated chlorophyll for chlorophyll penentration depth
    result['CHL_sat'] = np.nan # Weighted mean chlorophyll for first optical depth
    result['CHL_profile'] = np.nan # Chlorophyll profile classification 
    result['BBP_ratio'] = np.nan
    result['KPAR'] = np.nan # Light attenuation coefficient 
    result['MLD_2'] = np.nan # Mixed layer depth based on buoyancy frequency
    result['MLD_3'] = np.nan
    result['DNC_2'] = np.nan # Nitracline depth based on nitrate rate of change
    result['NAI_2'] = np.nan # Nitrate availability index
    result['Data_mode'] = str(data_mode).replace('[','').replace(']','')
    result['QC'] = str(flag).replace('[','').replace(']','')
    result['Note'] = ''

    # Data mode filter
    para_list = ['PRES','TEMP','PSAL','CHLA','NITRATE']
    Parameter = mydata.PARAMETER.astype(str).str.replace(r'\s', '').squeeze().values
    Parameter_data_mode = mydata.PARAMETER_DATA_MODE.astype(str).values
    if ~np.in1d(Parameter_data_mode[np.in1d(Parameter,para_list)],data_mode).all():
        result['Note'] += 'Not all data has required data mode.'
        return(pd.DataFrame.from_dict(result,orient='index').T)

    # Adjusted value filter
    if var_adj:
        adj = '_ADJUSTED'
    else:
        adj = ''

    if ~np.in1d([f'PRES{adj}',f'TEMP{adj}',f'PSAL{adj}',f'CHLA{adj}',f'NITRATE{adj}'],list(mydata.variables)).all():
        result['Note'] += 'Not all data has adjusted value.'
        return(pd.DataFrame.from_dict(result,orient='index').T)

    table_full = {
        'PRES' : mydata[f'PRES{adj}'].values,
        'PRES_QC' : mydata[f'PRES{adj}_QC'].astype(float).values,
        'PSAL' : mydata[f'PSAL{adj}'].values,
        'PSAL_QC' : mydata[f'PSAL{adj}_QC'].astype(float).values,
        'TEMP' : mydata[f'TEMP{adj}'].values,
        'TEMP_QC' : mydata[f'TEMP{adj}_QC'].astype(float).values,
        'CHLA' : mydata[f'CHLA{adj}'].values,
        'CHLA_QC' : mydata[f'CHLA{adj}_QC'].astype(float).values,
        'CHLA_RT' : mydata[f'CHLA'].values,
        'NITRATE' : mydata[f'NITRATE{adj}'].values,
        'NITRATE_QC' : mydata[f'NITRATE{adj}_QC'].astype(float).values,
        'BBP700' : mydata[f'BBP700{adj}'].values if f'BBP700{adj}' in mydata.data_vars else np.nan,
        'BBP700_QC' : mydata[f'BBP700{adj}_QC'].astype(float).values if f'BBP700{adj}_QC' in mydata.data_vars else np.nan,
    }
    
    # Calculate depth from pressure
    table_full['DEPTH'] = -np.asarray(gsw.conversions.z_from_p(table_full['PRES'], result['Lat'], geo_strf_dyn_height=0, sea_surface_geopotential=0))
    
    # Calculate potential density from temperature, salinity
    table_full['SIGMA'] = gsw.density.sigma0(table_full['PSAL'], table_full['TEMP'])

    table_full = pd.DataFrame(table_full).sort_values(by='DEPTH').dropna(subset=para_list).drop_duplicates(subset=['DEPTH']).reset_index(drop=True)
        
    # ========== CALCULATE SST, SSS ============
    table_ctd = table_full[np.in1d(table_full.TEMP_QC,flag)&np.in1d(table_full.PSAL_QC,flag)&(table_full.DEPTH>=0)].reset_index(drop=True)
    if (np.shape(table_ctd)[0] == 0) or (np.nanmin(table_ctd.DEPTH) > depth_top) or (np.nanmax(table_ctd.DEPTH) < depth_bottom):
        result['Note'] += f'Insufficient QC flagged CTD data, '
        return(pd.DataFrame.from_dict(result,orient='index').T)
    else:
        # Sea surface temperature
        try:
            result['SST'] = table_ctd.TEMP[table_ctd.DEPTH.idxmin()]
        except Exception as e:
            result['Note'] += f'SST error, '
            if verbose: print(e)
            pass
        
        # Sea surface salinity
        try:
            result['SSS'] = table_ctd.PSAL[table_ctd.DEPTH.idxmin()]
        except Exception as e:
            result['Note'] += f'SSS error, '
            if verbose: print(e)
            pass

    # ========== CALCULATE MLD ============ 
    # Mixed layer depth
    # Calculate squared buoyancy frequency from potential density, apply smoothing function if prompted
    try:
        table_ctd['N2']=np.append(np.nan,gsw.Nsquared(table_ctd.PSAL, table_ctd.TEMP, table_ctd.PRES, result['Lat'])[0])
        table_ctd.N2.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Apply smoothing
        if smooth:
            table_ctd['SIGMA'] = rolling_average(table_ctd.SIGMA,table_ctd.DEPTH,smooth_window,by_mean=by_mean)
            table_ctd['N2'] = rolling_average(table_ctd.N2,table_ctd.DEPTH,smooth_window,by_mean=by_mean)        
        # MLD base on theshold d(sigma-theta) = 0.03
        if ~np.isnan(table_ctd['SIGMA']).all():
            # Find potential density at 10m through intepolation
            #sigma_ref = table_ctd.SIGMA[np.abs(table_ctd.DEPTH - 10).idxmin()] # Sigma-theta at 10m
            sigma_ref = np.interp(10,table_ctd.DEPTH,table_ctd.SIGMA)
            # Find depth where sigma0 increase by 0.03
            result['MLD'] = np.interp(sigma_ref+d_sigma,table_ctd.SIGMA,table_ctd.DEPTH)
            # MLD by change in density with respect to change in temperature
            sal10 = np.interp(10,table_ctd.DEPTH,table_ctd.PSAL)
            temp10 = np.interp(10,table_ctd.DEPTH,table_ctd.TEMP)
            result['MLD_2'] = np.interp(sigma_ref+(gsw.density.sigma0(sal10,temp10-d_theta)-gsw.density.sigma0(sal10,temp10)),table_ctd.SIGMA[table_ctd.DEPTH>10],table_ctd.DEPTH[table_ctd.DEPTH>10])
        # Find all depth of local maxima for buoyancy
        if ~np.isnan(table_ctd.N2).all():
            N2_peak,_ = find_peaks(standardize(table_ctd.N2[table_ctd.DEPTH>=10]),height=peak_height)
            # Extract local/global maximum for buoyancy
            if (N2_local_max) and (np.size(N2_peak)!=0):
                result['MLD_3'] = table_ctd.DEPTH[table_ctd.DEPTH>=10][N2_peak[N2_local_max-1]]
                result['Note'] += 'N2 local maximum, '
            else:
                result['MLD_3'] = table_ctd.DEPTH[table_ctd.DEPTH>=10][table_ctd.N2[table_ctd.DEPTH>=10].idxmax()]
                result['Note'] += 'N2 global maximum, '                        
        else:
            result['Note'] += f'Unable to compute N2, '
        
    except Exception as e:
        result['Note'] += f'MLD error, '
        if verbose: print(e)
        pass

    # ========== CALCULATE DCM & SCHL ==========
    # Depth or chlorophyll maximum and surface chlorophyll
    # Filter flag
    table_chla = table_full[~np.isnan(table_full.CHLA)&np.in1d(table_full.CHLA_QC,flag)&~np.isnan(table_full.DEPTH)&(table_full.DEPTH>=0)].reset_index(drop=True)
    # Calculation
    if (np.shape(table_chla)[0] == 0) or (np.nanmin(table_chla.DEPTH) > depth_top) or (np.nanmax(table_chla.DEPTH) < depth_bottom):
        result['Note'] += 'Insufficient QC flagged CHLA data, '
        return(pd.DataFrame.from_dict(result,orient='index').T)
    elif np.nanmax(table_chla.CHLA) <= 0.05: # Account for Chl a below dectection point
        result['Note'] += 'No CHLA observation, '
        return(pd.DataFrame.from_dict(result,orient='index').T)
    else:
        try:
            table_chla['CHLA_unsmooth'] = table_chla.CHLA
            table_chla['BBP700_unsmooth'] = table_chla.BBP700
            # Extract Chl a
            if smooth:
                table_chla['CHLA'] = rolling_average(table_chla.CHLA,table_chla.DEPTH,smooth_window,by_mean=by_mean)
                
            # Surface Chla
            result['CHL_surface'] = table_chla.CHLA[table_chla.DEPTH.idxmin()]

            # Find all depth of local maxima for chl a
            DCM_idx = table_chla.CHLA.idxmax()
            CHLA_surface = np.nanmedian(table_chla.CHLA_unsmooth[(table_chla.DEPTH <= 15)&(table_chla.CHLA_unsmooth>0)])
            CHLA_max = table_chla.CHLA_unsmooth[DCM_idx]

            # Get depth of chlorophyll maxima
            result['DCM'] = table_chla.DEPTH[DCM_idx]
            result['CHL_max'] = CHLA_max

            # Backscattering
            if ~np.isnan(table_chla.BBP700).all():
                if smooth:
                    table_chla['BBP700'] = rolling_average(table_chla.BBP700, table_chla.DEPTH,smooth_window)
                BBP_idx = table_chla.BBP700[table_chla.DEPTH.between(result['DCM']-10,result['DCM']+10)].idxmax()
                BBP_surface = np.nanmin(table_chla.BBP700_unsmooth[(table_chla.DEPTH <= 15)&(table_chla.BBP700_unsmooth>0)])
                BBP_max = table_chla.BBP700_unsmooth[BBP_idx]
                result['BBP_ratio'] = BBP_max/BBP_surface

                # Classify chlorophyll profiles
                if (CHLA_max < 2*CHLA_surface):
                    result['CHL_profile'] = 'NO'
                else:
                    if (BBP_max < 1.3*BBP_surface):
                        result['CHL_profile'] = 'DAM'
                    else:
                        result['CHL_profile'] = 'DBM'
            
            # Estimate euphotic depth
            result['Zeu'] = euphotic_depth(table_chla.CHLA,table_chla.DEPTH)
            # Estimate first optical depth
            result['Zpd'] = result['Zeu']/-np.log(0.01)
            # Estimate chlorophyll penentration depth
            zlow_idx = table_chla.DEPTH[(table_chla.DEPTH>result['DCM'])&(table_chla.CHLA<CHLA_threshold)].idxmin()
            result['Zlow'] = table_chla.DEPTH[zlow_idx]

            ### Depth-integrated CHLA ###
            # Turn negatives to 0 as CHLA is often underestimated
            CHLA = np.array(table_chla.CHLA)
            CHLA[CHLA<0] = 0

            # Calculate depth-integrated CHL for CHL_peak
            if DCM_idx!=0:
                # result['CHL_peak'] = np.sum(0.5*(np.array(CHLA[:DCM_idx+1][:-1])+np.array(CHLA[:DCM_idx+1][1:]))*np.diff(table_chla.DEPTH[:DCM_idx+1]))
                result['CHL_peak'] = np.trapz(CHLA[:DCM_idx+1],table_chla.DEPTH[:DCM_idx+1])
            else:
                result['CHL_peak'] = CHLA[DCM_idx]*table_chla.DEPTH[DCM_idx]

            # Depth-integrated chlorophyll for euphotic depth
            zeu_idx = table_chla.DEPTH[table_chla.DEPTH > result['Zeu']].idxmin()
            if zeu_idx!=0:
                result['CHL_Zeu'] = np.trapz(CHLA[:zeu_idx+1],table_chla.DEPTH[:zeu_idx+1])
            else:
                result['CHL_Zeu'] = CHLA[zeu_idx]*table_chla.DEPTH[zeu_idx]

            # Depth-integrated chlorophyll for first optical depth:
            zpd_idx = table_chla.DEPTH[table_chla.DEPTH > result['Zpd']].idxmin()
            if zpd_idx!=0:
                result['CHL_Zpd'] = np.trapz(CHLA[:zpd_idx+1],table_chla.DEPTH[:zpd_idx+1])
            else:
                result['CHL_Zpd'] = CHLA[zpd_idx]*table_chla.DEPTH[zpd_idx]

            # Depth-integrated CHL to chlrophyll penentration depth
            if zlow_idx!=0:
                result['CHL_Zlow'] = np.trapz(CHLA[:zlow_idx+1],table_chla.DEPTH[:zlow_idx+1])
            else:
                result['CHL_Zlow'] = CHLA[zlow_idx]*table_chla.DEPTH[zlow_idx]

            # Calculate KPAR base on Zeu
            result['KPAR'] = (-np.log(0.01)/result['Zeu'])
            
            # Calculate CHL_sat
            result['CHL_sat'] = np.trapz(CHLA[:zpd_idx+1]*np.exp(-2*result['KPAR']*table_chla.DEPTH[:zpd_idx+1]),table_chla.DEPTH[:zpd_idx+1])/np.trapz(np.exp(-2*result['KPAR']*table_chla.DEPTH[:zpd_idx+1]),table_chla.DEPTH[:zpd_idx+1])
            
        except Exception as e:
            result['Note'] += f'DCM error 2, '
            if verbose: print(e)
            pass

    # ========== CALCULATE NAI/DNC ===========
    # Depth of nitracline or nitrate availability index
    # Filter flag
    table_filter = table_full[~np.isnan(table_full.NITRATE)&np.in1d(table_full.NITRATE_QC,flag)&~np.isnan(table_full.DEPTH)&(table_full.DEPTH>=0)].reset_index(drop=True)
    # DNC
    if( np.shape(table_filter)[0] == 0) or (np.nanmin(table_filter.DEPTH) > depth_top) or (np.nanmax(table_filter.DEPTH) < depth_bottom):
        result['Note'] += 'Insufficient QC flagged NITRATE data, '
        return(pd.DataFrame.from_dict(result,orient='index').T)
    elif (table_filter.NITRATE < -1).any():
        result['Note'] += 'NITRATE contains negative, '
        return(pd.DataFrame.from_dict(result,orient='index').T)
    else:
        try:
            nitrate_surface = np.nanmedian(table_filter.NITRATE[table_filter.DEPTH <= 10])
            # Apply smoothing function if prompted
            if smooth:
                #table_filter['NITRATE'] = savgol_filter(table_filter.NITRATE,smooth_window,smooth_order)
                table_filter['NITRATE'] = rolling_average(table_filter.NITRATE,table_filter.DEPTH,smooth_window,by_mean=by_mean)
            #nitrate_surface = table_filter.NITRATE[np.nanargmin(table_filter.DEPTH)]

            # Calculate discrete different among measurement intervals
            dNITRATE = np.diff(table_filter.NITRATE)/np.diff(table_filter.DEPTH)
            dNITRATE[(dNITRATE==np.inf)|(dNITRATE==-np.inf)]=np.nan
            dNITRATE = rolling_average(dNITRATE,table_filter.DEPTH[1:],smooth_window,by_mean=by_mean)
            dNITRATE_peak,_  = find_peaks(standardize(dNITRATE),height=peak_height)
            # Extract local/global maximum for buoyancy
            if (N2_local_max) and (np.size(dNITRATE_peak)!=0):
                result['DNC_2'] = table_filter.DEPTH[1:][dNITRATE_peak[N2_local_max-1]]
                result['Note'] += 'dNITRATE local maximum, '
            else:
                result['DNC_2'] = table_filter.DEPTH[1:][np.nanargmax(dNITRATE)]
                result['Note'] += 'dNITRATE global maximum, '         
            
            # DNC based on delta Nitrate
            #result['DNC'] = np.nanmin(table_filter.DEPTH[table_filter.NITRATE >= (nitrate_surface+dNITRATE_ref)])
            result['DNC'] = np.interp(nitrate_surface+dNITRATE_ref,table_filter.NITRATE,table_filter.DEPTH)

        except Exception as e:
            result['Note'] += f'DNC error, '
            if verbose: print(e)
            pass
        
        # NAI   
        try:
            # If surface [NO3] >= boundary then NAI = surface [NO3]
            # If surface [NO3] <= boundary and bottom [NO3] >= target then integrate depth for [NO3]=target
            # If bottom [NO3] <= target then NAI = minus bottom depth
            NAI_target = nitrate_surface+dNITRATE_ref
            if nitrate_surface >= NAI_boundary:
                result['NAI'] = nitrate_surface
                result['NAI_2'] = nitrate_surface
            elif (nitrate_surface < NAI_boundary) and (np.nanmax(table_filter.NITRATE) > NAI_target): # interpolate around NAI_boundary
                #idx = table_filter.NITRATE[table_filter.NITRATE >= NAI_target].index[0]
                #result['NAI'] = -(interpolate(NAI_target,table_filter.NITRATE[idx],table_filter.DEPTH[idx],table_filter.NITRATE[idx-1],table_filter.DEPTH[idx-1]))
                result['NAI'] = -result['DNC']
                result['NAI_2'] = -result['DNC_2']
            else:
                result['NAI'] = -table_filter.DEPTH[table_filter.DEPTH.idxmax()]
                result['NAI_2'] = -table_filter.DEPTH[table_filter.DEPTH.idxmax()]
        except Exception as e:
            result['Note'] += f'NAI error, '
            if verbose: print(e)
            pass
    
    # ========== Plot ===========
    if plot:
        try:
            plt.rc('font', size=10) 
            fig1, [ax1,ax4] = plt.subplots(1,2,figsize=(10,7))
            ax1.plot(table_ctd.SIGMA, table_ctd.DEPTH,'k.-',label='Density')
            ax2 = ax1.twiny()
            ax2.plot(table_ctd.PSAL, table_ctd.DEPTH,'y.-',label='Salinity')
            ax3 = ax1.twiny()
            ax3.plot(table_ctd.TEMP, table_ctd.DEPTH,'r.-',label='Temperature')
            ax3.spines['top'].set_position(('outward', 60))
            ax4.plot(table_chla.CHLA, table_chla.DEPTH,'g.-',label='Chla')
            ax5 = ax4.twiny()
            ax5.plot(table_filter.NITRATE, table_filter.DEPTH,'b.-',label='Chla')

            ax1.axhline(y=result['MLD_2'],linestyle='--',color='k')
            ax4.axhline(y=result['DNC'],linestyle='--',color='b')
            ax5.axhline(y=result['DCM'],linestyle='--',color='g')

            ax1.set_ylim(zmax,0)
            ax4.set_ylim(zmax,0)

            ax1.set_ylabel("Depth [$m$]")
            ax1.set_xlabel("Density [$kg$ $m^{-3}$]")
            ax2.set_xlabel("Salinity [$psu$]")
            ax3.set_xlabel("Temperature [$°C$]")
            ax4.set_xlabel("Chlorophyll  a [$mg$ $m^{-3}$]")
            ax5.set_xlabel("Nitrate [$µmol$ $kg^{-1}$]")

            ax1.xaxis.label.set_color('k')
            ax2.xaxis.label.set_color('olive')
            ax3.xaxis.label.set_color('r')
            ax4.xaxis.label.set_color('g')
            ax5.xaxis.label.set_color('b')
            ax2.tick_params(axis='x', colors='olive')
            ax3.tick_params(axis='x', colors='r')
            ax4.tick_params(axis='x', colors='g')
            ax5.tick_params(axis='x', colors='b')

            ax2.spines['top'].set_color('olive')
            ax3.spines['top'].set_color('r')
            ax5.spines['bottom'].set_color('g')
            ax5.spines['top'].set_color('b')
            plot_name = result['Float_Cycle']
            plt.suptitle(f'Float_cycle: {plot_name}',y=1)
            plt.tight_layout()

        except Exception as e:
            if verbose: print(e)
            pass
    
    if plot_extra:
        try:
            fig2, axs = plt.subplots(2,2,figsize=(10,10))
            axs = axs.flatten()
            
            axs[0].plot(table_ctd.SIGMA,table_ctd.DEPTH,'k.-')
            ax6 = axs[0].twiny()
            ax6.plot(table_ctd.N2*1000,table_ctd.DEPTH,'r-')
            axs[0].axhline(y=result['MLD'],linestyle='--',color='b')
            axs[0].axhline(y=result['MLD_2'],linestyle='--',color='g')
            axs[0].axhline(y=result['MLD_3'],linestyle='--',color='y')
            axs[0].plot(sigma_ref,10,'bo')

            axs[1].plot(table_filter.NITRATE,table_filter.DEPTH,'k.-')
            ax7 = axs[1].twiny()
            ax7.plot(dNITRATE,table_filter.DEPTH[1:],'r-')
            axs[1].axhline(y=result['DNC'],linestyle='--',color='b')
            axs[1].axhline(y=result['DNC_2'],linestyle='--',color='g')

            axs[2].plot(table_chla.CHLA,table_chla.DEPTH,'g.-')
            ax8 = axs[2].twiny()
            ax8.plot(table_chla.BBP700*10**4,table_chla.DEPTH,'k.-')
            lines = []
            lines.append(axs[2].axhline(y=result['Zpd'],linestyle='--',color='y',label='Zpd'))
            lines.append(axs[2].axhline(y=result['Zeu'],linestyle='--',color='r',label='Zeu'))
            lines.append(axs[2].axhline(y=result['Zlow'],linestyle='--',color='b',label='Zlow'))
            axs[2].legend(handles=lines,loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=axs[2].transAxes)
            
            axs[3].axis("off")

            axs[0].set_ylim(zmax,0)
            axs[1].set_ylim(zmax,0)
            axs[2].set_ylim(zmax,0)

            axs[0].set_ylabel("Depth [$m$]")
            axs[0].set_xlabel("Density [$m^3$ $kg^{-1}$]")
            axs[2].set_ylabel("Depth [$m$]")
            axs[2].set_xlabel("Chlorophyll  a [$mg$ $m^{-3}$]")
            axs[1].set_xlabel("Nitrate [$µmol$ $kg^{-1}$]")
            ax6.set_xlabel("N2 [$10^{-3}$ $s^{-2}$]")
            ax7.set_xlabel("dNITRATE")
            ax8.set_xlabel("BBP700 [$10^{-4}$ $m^{-1}$]")
            plt.tight_layout()
        except Exception as e:
            if verbose: print(e)
            pass
    # ========== EXPORT RESULT ==========

    return(pd.DataFrame.from_dict(result,orient='index').T)

def parallelize_hbt(func, df):
    num_cores = multiprocessing.cpu_count()-1  # leave one free to not freeze machine
    df_split = np.array_split(df, num_cores) # split dataframe into chunks
    pool = multiprocessing.Pool(num_cores) # number of pool
    pool.imap(func, df_split) # split work
    pool.close()
    pool.join()

def model_run(filenames, file_dir=None, save_dir=None, **kwargs):
    if file_dir is None:
        file_dir = os.getcwd()

    if save_dir is None:
        save_dir = os.getcwd()
    
    for file in filenames:
        try:
            # Load .nc file
            data = xr.open_dataset(file_dir + '/' + file)
            data['N_PROF'] = data['N_PROF'] # Copy dimension to coordinate for info extraction
            ## Automized calculations
            # Create blank dataframe
            cols = ['Float' , 'N_PROF', 'Cycle', 'Float_cycle', 'Date', 'Year', 'Month', 'Hour', 'Season', 'Daynight',
                    'Lat', 'Lon', 'MLD', 'DCM', 'SST', 'SSS', 'DNC', 'NAI', 'Zeu', 'Zpd', 'Zlow',
                    'CHL_surface', 'CHL_max', 'CHL_peak', 'CHL_Zeu', 'CHL_Zpd', 'CHL_Zlow', 'CHL_sat','CHL_profile','BBP_ratio','KPAR',
                    'MLD_2', 'MLD_3', 'DNC_2', 'NAI_2', 'Data_mode', 'QC', 'Note']
            result_table = pd.DataFrame(columns=cols)
            # Calculate all float cycle
            for i in np.unique(data.N_PROF):
                try:
                    mydata = data.sel(N_PROF=i)
                    mycal = def_var(mydata, **kwargs)
                    row = pd.DataFrame([mycal.values[0]], columns=cols)
                    result_table = pd.concat([result_table, row])
                except Exception as e:
                    print(f'{file} error occurred during cycle {i}')
                    pass

            #  Edit data frame
            df = pd.DataFrame(result_table, columns=cols)
            df['Date'] =  pd.to_datetime(df['Date'])

            # Save final calculations
            savename = save_dir + '/' + str(df['Float'].iat[0]) + '_hdv.csv'
            df.to_csv(savename, index=False)
            print(f'{file} processed.')

        except Exception as e:
            print(f'{file} error occurred')
            pass
