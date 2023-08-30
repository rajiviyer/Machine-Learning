import numpy as np
import pandas as pd
from typing import List
from modules import metrics

class NoGANSynth:
    def __init__(self, data:np.array, features:List = None, 
                 bins:List = None
                 ):
        self.data = np.array(data, np.float64)
        self.nobs = len(data)
        if isinstance(data, pd.DataFrame):
            self.features = data.columns
        elif features is None:
            raise TypeError("features should not be null if data is not a Pandas Dataframe")
        else:
            self.features = features
        self.n_features = len(self.features)
        self.eps = 0.0000000001
        if bins is None:
            self.bins_per_feature = [ np.random.randint(3,5) for i in range(self.n_features)]
        else:
            self.bins_per_feature = bins

        # self.bin_edges = np.array([(np.histogram(npdata[:,k], 
        #                                          bins=self.bins_per_feature[k]) \
        #                     )[1] for k in range(self.n_features)], dtype='object')
        self.bin_edges = [np.quantile(self.data[:,k], 
                                      np.arange(0, 1 + self.eps, 1/self.bins_per_feature[k]), 
                                      axis=0
                                      ) for k in range(self.n_features)]
        
    def create_bin_keys(self):
        # Get bin indices for each row in the data
        bin_indices = np.array([np.clip(np.searchsorted(self.bin_edges[col], 
                                                        self.data[:, col], side='right')-1,
                                        0,
                                        len(self.bin_edges[col])-2
                                        ) for col in range(self.data.shape[1])])
        
        bin_indices = bin_indices.T

        # Find the counts of all unique list entries
        unique_entries, counts = np.unique(bin_indices, axis=0, return_counts=True)

        # Create a dictionary having each entry as key and corresponging counts and actual lists as values
        bin_keys = {}
        for entry, count in zip(unique_entries, counts):
            key_str = ', '.join(map(str, entry))
            lower_val = [self.bin_edges[k][entry[k]] for k in range(len(entry))]
            upper_val = [self.bin_edges[k][1 + entry[k]] for k in range(len(entry))]
            bin_keys[key_str] = {'frequency': count, 'value': entry, 
                                    'lower_val': lower_val, 'upper_val': upper_val
                                    }

        self.bin_keys = bin_keys
    
    def random_bin_counts(self,no_of_rows:int)->np.array:
        # generate multinomial bin counts with same expectation as real counts
        pvals = []
        for key in self.bin_keys:
            #print(f"bin_count[{skey}] = {bin_count[skey]}, nobs: {nobs}, bin_count[{skey}]/nobs = {bin_count[skey]/nobs}")
            pvals.append(self.bin_keys[key]["frequency"]/self.nobs)
        return(np.random.multinomial(no_of_rows, pvals))

    def generate_synthetic_data(self, no_of_rows:int)->pd.DataFrame:
        #print("*"*40 + "Generating Synthetic Data" + "*"*40)
        bin_count_random = self.random_bin_counts(no_of_rows)
        data_synth = []
        for i, key in enumerate(self.bin_keys):
            lower_val = self.bin_keys[key]["lower_val"]
            upper_val = self.bin_keys[key]["upper_val"]
            count = bin_count_random[i]
            #print(key, count)
            for j in range(count):
                new_obs = np.empty(self.n_features) # synthesized obs
                for k in range(self.n_features):
                    new_obs[k] = np.random.uniform(lower_val[k],
                                                   upper_val[k])   
                    #print("adding new_obs")
                data_synth.append(new_obs)               
        data_synth = pd.DataFrame(data_synth, columns = self.features)
        return data_synth
    
    def evaluate(self, training_data:pd.DataFrame, 
                 validation_data:pd.DataFrame, synthetic_data:pd.DataFrame
                 ):
            
        ecdf_validation = metrics.compute_ecdf(validation_data, 1000, False)
        ks_max, ecdf_val1, ecdf_synth = metrics.ks_delta(synthetic_data, ecdf_validation)  
        print(f"ECDF Kolmogorof-Smirnov dist. (synth. vs valid.): {ks_max:6.4f}")
        
        base_ks_max, ecdf_val2, ecdf_train = metrics.ks_delta(training_data, ecdf_validation)  
        print(f"Base ECDF Kolmogorof-Smirnov dist. (train. vs valid.): {base_ks_max:6.4f}")          
        
            
        
        