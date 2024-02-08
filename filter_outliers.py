#!/usr/bin/env python3
# filter by 1 group

def filter_outliers_1(df, group, columns, lower_quantile=0.25, upper_quantile=0.75): 
    grouped_df = df.groupby(group)
    # Calculate upper and lower quantiles for each column
    quantiles = {}
    for col in columns:
        quantiles[col] = {
            'q1': grouped_df[col].quantile(lower_quantile),
            'q3': grouped_df[col].quantile(upper_quantile)
        }
        
    # Calculate interquartile range (IQR) for each column
    iqr = {}
    for col in columns:
        iqr[col] = quantiles[col]['q3'] - quantiles[col]['q1']

    # Define cutoff for identifying outliers based on upper and lower quantile for each column
    upper_cutoff = {}
    lower_cutoff = {}
    for col in columns:
        upper_cutoff[col] = quantiles[col]['q3'] + 1.5 * iqr[col]
        lower_cutoff[col] = quantiles[col]['q1'] - 1.5 * iqr[col]

    # Filter data to exclude outliers for each column
    filtered_df = df.copy()
    for col in columns:
        filtered_df = filtered_df[~filtered_df.apply(lambda x: (x[col] > upper_cutoff[col][(x[group])]) | (x[col] < lower_cutoff[col][(x[group])]), axis=1)]
    return filtered_df.reset_index(drop=True)

### For 2 group at the same time
def filter_outliers_2(df, group1, group2, columns, lower_quantile=0.25, upper_quantile=0.75):
    grouped_df = df.groupby([group1, group2])
    # Calculate upper and lower quantiles for each column
    quantiles = {}
    for col in columns:
        quantiles[col] = {
            'q1': grouped_df[col].quantile(lower_quantile),
            'q3': grouped_df[col].quantile(upper_quantile)
        }
    
    # Calculate interquartile range (IQR) for each column
    iqr = {}
    for col in columns:
        iqr[col] = quantiles[col]['q3'] - quantiles[col]['q1']
    
    # Define cutoff for identifying outliers based on upper and lower quantile for each column
    upper_cutoff = {}
    lower_cutoff = {}
    for col in columns:
        upper_cutoff[col] = quantiles[col]['q3'] + 1.5 * iqr[col]
        lower_cutoff[col] = quantiles[col]['q1'] - 1.5 * iqr[col]
    
    # Filter data to exclude outliers for each column
    filtered_df = df.copy()
    for col in columns:
        filtered_df = filtered_df[~filtered_df.apply(lambda x: (x[col] > upper_cutoff[col][(x[group1], x[group2])]) | (x[col] < lower_cutoff[col][(x[group1], x[group2])]), axis=1)]
    
    return filtered_df.reset_index(drop=True)

## multiple groups
# def filter_outliers_n(df, groups, columns, lower_quantile=0.25, upper_quantile=0.75):
#     grouped_df = df.groupby(groups)
#     # Calculate upper and lower quantiles for each column
#     quantiles = {}
#     for col in columns:
#         quantiles[col] = {
#             'q1': grouped_df[col].quantile(lower_quantile),
#             'q3': grouped_df[col].quantile(upper_quantile)
#         }
    
#     # Calculate interquartile range (IQR) for each column
#     iqr = {}
#     for col in columns:
#         iqr[col] = quantiles[col]['q3'] - quantiles[col]['q1']
    
#     # Define cutoff for identifying outliers based on upper and lower quantile for each column
#     upper_cutoff = {}
#     lower_cutoff = {}
#     for col in columns:
#         upper_cutoff[col] = quantiles[col]['q3'] + 1.5 * iqr[col]
#         lower_cutoff[col] = quantiles[col]['q1'] - 1.5 * iqr[col]
    
#     # Filter data to exclude outliers for each column
#     filtered_df = df.copy()
#     for col in columns:
#         #filtered_df = filtered_df[~filtered_df.apply(lambda x: tuple(x[group] for group in groups) in upper_cutoff[col] and (x[col] > upper_cutoff[col][tuple(x[group] for group in groups)]) or tuple(x[group] for group in groups) in lower_cutoff[col] and (x[col] < lower_cutoff[col][tuple(x[group] for group in groups)]), axis=1)]
#         filtered_df = filtered_df[~filtered_df.apply(lambda x: tuple([x[group] for group in groups]) in upper_cutoff[col] and (x[col] > upper_cutoff[col][tuple([x[group] for group in groups])]) or tuple([x[group] for group in groups]) in lower_cutoff[col] and (x[col] < lower_cutoff[col][tuple([x[group] for group in groups])]), axis=1)]

#     return filtered_df.reset_index(drop=True) 


def filter_outliers_n(df, groups, columns, lower_quantile=0.25, upper_quantile=0.75):
    if isinstance(groups, str):
        groups = [groups]
    
    grouped_df = df.groupby(groups)
    
    # Calculate upper and lower quantiles for each column
    quantiles = {}
    for col in columns:
        quantiles[col] = {
            'q1': grouped_df[col].quantile(lower_quantile),
            'q3': grouped_df[col].quantile(upper_quantile)
        }
    
    # Calculate interquartile range (IQR) for each column
    iqr = {}
    for col in columns:
        iqr[col] = quantiles[col]['q3'] - quantiles[col]['q1']
    
    # Define cutoff for identifying outliers based on upper and lower quantile for each column
    upper_cutoff = {}
    lower_cutoff = {}
    for col in columns:
        upper_cutoff[col] = quantiles[col]['q3'] + 1.5 * iqr[col]
        lower_cutoff[col] = quantiles[col]['q1'] - 1.5 * iqr[col]
    
    # Filter data to exclude outliers for each column
    filtered_df = df.copy()
    for col in columns:
        if len(groups) > 1:
            filtered_df = filtered_df[~filtered_df.apply(lambda x: tuple(x[group] for group in groups) in upper_cutoff[col] and (x[col] > upper_cutoff[col][tuple(x[group] for group in groups)]) or tuple(x[group] for group in groups) in lower_cutoff[col] and (x[col] < lower_cutoff[col][tuple(x[group] for group in groups)]), axis=1)]
        else:
            filtered_df = filtered_df[~filtered_df.apply(lambda x: (x[col] > upper_cutoff[col][(x[groups[0]])]) | (x[col] < lower_cutoff[col][(x[groups[0]])]), axis=1)]
    
    return filtered_df.reset_index(drop=True) 
