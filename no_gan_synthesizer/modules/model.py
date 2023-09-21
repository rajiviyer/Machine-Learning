import numpy as np
import pandas as pd
from typing import List
from modules import metrics
from sklearn.decomposition import PCA
from genai_evaluation import multivariate_ecdf, ks_statistic

class NoGANSynth:
    """
    The main NoGAN Synthesizer Class
    """
    def __init__(self, data:np.array, features:List = None
                 )->None:
        """
        Initialize Training Data, no of objects, features, no of features and epsilon

        Args:
            data (np.array): Training Pandas DataFrame
            features (List, optional): Features List. Defaults to None.

        Raises:
            TypeError: If Pandas DataFrame is passed as Training Data, the features can be retrieved from that and hence features list is not required. But if a numpy array is passed, then the features is required. Typerror will be raised if numpy array is passed and features list is None.
        Returns:
            Tuple[pd.DataFrame, dict]: Tuple of Pandas DataFrame & results dictionary            
            
        """
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
        
    def create_bin_keys(self, bins:List = None)->None:
        """
        Function to create bins for each training data column using the list send as arguments.
        
        Args:
            bins (List, optional): Bins List. Defaults to None. If it is None, then random bins between 3 to 5 will be assigned. Recommended to pass a tuned hyperparameter bins list
        """
        
        # Get bin indices for each row in the data
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
        """
        Function to generate multinomial bin counts with same expectation as real counts

        Args:
            no_of_rows (int): Row Count

        Returns:
            np.array: Random Bin Count Array
        """
        pvals = []
        for key in self.bin_keys:
            #print(f"bin_count[{skey}] = {bin_count[skey]}, nobs: {nobs}, bin_count[{skey}]/nobs = {bin_count[skey]/nobs}")
            pvals.append(self.bin_keys[key]["frequency"]/self.nobs)
        return(np.random.multinomial(no_of_rows, pvals))

    def generate_synthetic_data(self, no_of_rows:int)->pd.DataFrame:
        """
        The main function which Generates the Synthetic Data.
        It calls random bin to create the multinomial bin counts.
        Then for each key, gets the lower and upper bound and generates an observation (random uniform value) between those bounds
        Once the new observations list is generated, convert into a pandas synthetic dataframe and return.

        Args:
            no_of_rows (int): Row Count

        Returns:
            pd.DataFrame: Generate Synthetic Pandas DataFrame
        """
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
                 validation_data:pd.DataFrame, 
                 synthetic_data:pd.DataFrame,
                 n_nodes:int = 1000,
                 random_seed:int = 42,
                 verbose:bool = True
                 )->dict:
        """
        Function to evaluate Synthetic vs Validation Data & Training vs Validation Data

        Args:
            training_data (pd.DataFrame): Training Pandas DataFrame
            validation_data (pd.DataFrame): Validation Pandas DataFrame
            synthetic_data (pd.DataFrame): Synthetic Pandas DataFrame
            n_nodes (int, optional): Number of ECDF Nodes. Defaults to 1000.
            verbose (bool, optional): Defaults to True.

        Returns:
            dict: Results Dictionary having the evaluation metrics of train vs validation and synth vs validation
        """
        #np.random.seed(random_seed)
        # ecdf_validation = metrics.compute_ecdf(validation_data, n_nodes, True, verbose)
        # ks_max, ecdf_val1, ecdf_synth = metrics.ks_delta(synthetic_data, ecdf_validation)  
        # print(f"ECDF Kolmogorof-Smirnov dist. (synth. vs valid.): {ks_max**(1/len(training_data.columns)):6.4f}")
        
        # base_ks_max, ecdf_val2, ecdf_train = metrics.ks_delta(training_data, ecdf_validation)  
        # print(f"Base ECDF Kolmogorof-Smirnov dist. (train. vs valid.): {base_ks_max**(1/len(training_data.columns)):6.4f}")          
        
        _, ecdf_val1, ecdf_synth = \
                        multivariate_ecdf(validation_data, 
                                     synthetic_data, 
                                     n_nodes = n_nodes,
                                     verbose = verbose,
                                     random_seed=random_seed)

        _, ecdf_val2, ecdf_train = \
                        multivariate_ecdf(validation_data, 
                                     training_data, 
                                     n_nodes = n_nodes,
                                     verbose = verbose,
                                     random_seed=random_seed)
        
        ks_max = ks_statistic(ecdf_synth, ecdf_val1)
        base_ks_max = ks_statistic(ecdf_train, ecdf_val2)
                        
        results = {}
        results["synth_comparison"] = {"ks_stat": ks_max, 
                                       "ecdf_data": (ecdf_val1, ecdf_synth)}
        
        results["train_comparison"] = {"ks_stat": base_ks_max, 
                                       "ecdf_data": (ecdf_val2, ecdf_train)}
        
        return results
    
    def evaluate_with_pca(self, training_data:pd.DataFrame, 
                          validation_data:pd.DataFrame, 
                          synthetic_data:pd.DataFrame,
                          n_nodes:int = 1000,
                          verbose:bool = True):
        
        pca_t = PCA(n_components = 2)
        pca_v = PCA(n_components = 2)
        pca_s = PCA(n_components = 2)
        
        pca_training = pd.DataFrame(pca_t.fit_transform(training_data), 
                                    columns = ["x0","x1"])
        pca_validation = pd.DataFrame(pca_v.fit_transform(validation_data), 
                                    columns = ["x0","x1"])
        pca_synthetic = pd.DataFrame(pca_s.fit_transform(synthetic_data), 
                                    columns = ["x0","x1"])

            
        ecdf_validation = metrics.compute_ecdf(pca_validation, n_nodes, False, verbose)
        ks_max, ecdf_val1, ecdf_synth = metrics.ks_delta(pca_synthetic, ecdf_validation)  
        
        # print(f"ECDF Kolmogorof-Smirnov dist. (synth. vs valid.): {ks_max**(1/len(training_data.columns)):6.4f}")
        
        base_ks_max, ecdf_val2, ecdf_train = metrics.ks_delta(pca_training, ecdf_validation)  
        # print(f"Base ECDF Kolmogorof-Smirnov dist. (train. vs valid.): {base_ks_max**(1/len(training_data.columns)):6.4f}")    
              
        results = {}
        results["synth_comparison"] = {"ks_stat": ks_max, 
                                       "ecdf_data": (ecdf_val1, ecdf_synth)}
        
        results["train_comparison"] = {"ks_stat": base_ks_max, 
                                       "ecdf_data": (ecdf_val2, ecdf_train)}
        
        return results          