import numpy as np
import pandas as pd
from modules import model
from typing import List, Tuple

def nogan_synth(nogan:model.NoGANSynth, training_data:pd.DataFrame,
                validation_data:pd.DataFrame, bins:List,
                n_nodes:int = 1000, random_seed:int = 42,
                verbose:bool = True)-> Tuple[pd.DataFrame, dict]:
    """
    Function to train NoGan on a training set, create synthesis data and compare with validation set using ECDF & KS Distance Measure

    Args:
        nogan (model.NoGANSynth): NoGANSynth Object Instance
        training_data (pd.DataFrame): Training Pandas DataFrame
        validation_data (pd.DataFrame): Validation Pandas DataFrame
        bins (List): List of bins per column - Hyperparameter
        n_nodes (int, optional): The number of ECDF Nodes to be generated during evaluation. Defaults to 1000.
        random_seed (int, optional): Random Seed. Defaults to 42.
        verbose (bool, optional): Defaults to True.

    Raises:
        ValueError: If any of training_data, validation_data, NoGANSynth Object Instance or bins hyperparameter list is empty
        ValueError: If Bin List Length and count of columns don't match

    Returns:
        Tuple[pd.DataFrame, dict]: Tuple of Pandas DataFrame & results dictionary
    """
    if training_data.empty or validation_data.empty or not nogan or not bins:
        raise ValueError("One or more arguments are empty")
    
    if len(bins) != len(training_data.columns):
        raise ValueError("Bin length and columns count should be the same")        
    
    np.random.seed(random_seed) 

    nogan.create_bin_keys(bins = bins)
    
    # Synthesize Data
    n_synth_rows = len(validation_data)
    synth_data = nogan.generate_synthetic_data(n_synth_rows)

    # Convert columns to integer data type
    int_columns = training_data.select_dtypes(include=['int']).columns
    for col in int_columns:
        synth_data[col] = synth_data[col].astype(int) 
        
    # Evaluate the generated synthetic with validation data
    # results = nogan.evaluate_with_pca(training_data, validation_data,
    #                          synth_data, n_nodes, verbose = False) 

    results = nogan.evaluate(training_data, validation_data,
                             synth_data, n_nodes, random_seed, 
                             verbose = verbose) 

    
    return synth_data, results  