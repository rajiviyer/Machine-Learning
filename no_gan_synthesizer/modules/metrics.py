import pandas as pd
import numpy as np
from typing import List, Tuple

def compute_ecdf(data:pd.DataFrame, n_nodes:int = 1000, 
                 adjusted:bool = False)->List:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data should be a Pandas DataFrame!!") 
    
    eps = 0.0000000001
    ecdf = []
    features = data.columns
    n_features = len(features)
    for point in range(n_nodes):
        if point % 100 == 0:
            print(f"Sampling ecdf, location = {point}, adjusted = {adjusted}")
        
        # Get random percentiles
        percentiles = np.random.uniform(0, 1, n_features)
        if adjusted:
            percentiles = percentiles**(1/n_features)
        
        # Get the percentile values from the dataset for each column
        perc_vals = [eps + np.quantile(data.iloc[:,k],perc) for k, perc in enumerate(percentiles)]
        #print(perc_vals)
        
        # Create the query string combined for each column
        query_str = " and ".join([ f"{features[k]} <= {perc_val}"  for k, perc_val in enumerate(perc_vals)])
        
        # Get the counts of rows which satisfy the conditions in the query string
        filter_count = len(data.query(query_str))
        #print(filter_count)
        
        # For counts > 0, create key: str of the list of perc_vals
        # Append key, query_str & the normalized filter count
        if filter_count > 0:
            key = ', '.join(map(str, perc_vals))
            #print(key)
            ecdf.append((key, query_str, filter_count/ len(data)))
            
    # Sort the list based on the items (third element of each tuple)
    ecdf.sort(key=lambda item: item[2])

    return ecdf

def ks_delta(synthetic_data:pd.DataFrame, ecdf_validation:List)->Tuple:
    ks_max = 0
    ecdf_real = []
    ecdf_synth = []
    # for each entry in the ecdf_validation list
    # Retrieve the query_str and run on synthetic data to get the filter counts
    # Normalize the filter count
    # calculate ks distance between the normalized validation filter count & the normalized synthetic filter count and get the max ks distance value
    
    for i, e_val in enumerate(ecdf_validation):
        query_str = e_val[1]
        filter_count = len(synthetic_data.query(query_str))
        synth_value = filter_count / len(synthetic_data)
        real_value = e_val[2]
        ks = abs(real_value - synth_value)
        ecdf_real.append(real_value)
        ecdf_synth.append(synth_value)
        if ks > ks_max:
            ks_max = ks

    return (ks_max, ecdf_real, ecdf_synth)
