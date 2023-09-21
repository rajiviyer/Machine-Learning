from typing import Union, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import ipywidgets as widgets
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from modules.models import MCA
from flatten_dict import flatten
import re

def get_cleaned_students_data()->pd.DataFrame:
    """
    Get Students Dataset, clean the column names and Label Encode the Category Columns

    Returns:
        pd.DataFrame: Pandas DataFrame
    """
    data = pd.read_csv("data/student_success.csv",sep = ";")
    data.columns = data.columns.str.lower().str.replace("[/,\t,(,),']","", regex = True).str.replace(" ", "_")
    
    # cat_cols = [
    #     'marital_status','application_mode','course',
    #     'previous_qualification','nacionality','mothers_qualification', 'fathers_qualification','mothers_occupation', 'fathers_occupation','target'
    #     ]
    
    # #for col in data.select_dtypes(include=[object]).columns:
    # for col in cat_cols:
    #     data[col] = label_encode(data[col])
    
    # data["target"] = label_encode(data["target"])
    
    return data
    
def label_encode(categorical_data: Union[list, pd.Series])->List:
    """
    Function to perform Label Encoding for Categorical Columns

    Args:
        categorical_data (Union[list, pd.Series]): Accepts a list or pandas Series as input

    Returns:
        List: Returns a list of Label Encoded values
    """
    unique_values = list(set(categorical_data))
    label_mapping = {value: index for index, value in enumerate(unique_values)}

    # Encode the categorical data using the label mapping
    encoded_data = [label_mapping[value] for value in categorical_data]
    
    # Reverse transformation to get back original labels
    # decoded_data = [value for index in encoded_data for value, idx 
    #                 in label_mapping.items() if idx == index]
    return encoded_data

def stratified_train_test_split(df:pd.DataFrame, target_column:str, 
                                train_size:float=0.8, 
                                random_state:int=None)->Tuple:
    """
    Function to perform Stratified Sampling

    Args:
        df (pd.DataFrame): Pandas DataFrame
        target_column (str): Target Column Name
        train_size (float, optional): Percent Split Value for Train. Defaults to 0.8.
        random_state (int, optional): Random Seed. Defaults to None.

    Returns:
        Tuple: Returns a tuple of Train and Validation Data
    """
    unique_targets = df[target_column].unique()
    train_indices = []
    test_indices = []
    
    for target_value in unique_targets:
        subset = df[df[target_column] == target_value]
        subset_size = len(subset)
        
        train_subset_size = int(subset_size * train_size)
        
        # Shuffle indices
        np.random.seed(random_state)
        shuffled_indices = np.random.permutation(subset.index)
        
        train_indices.extend(shuffled_indices[:train_subset_size])
        test_indices.extend(shuffled_indices[train_subset_size:])
        
    train_set = df.loc[train_indices]
    test_set = df.loc[test_indices]
    
    train_set.reset_index(drop=True, inplace=True)
    test_set.reset_index(drop=True, inplace=True)
    
    return train_set, test_set

def rebinnable_interactive_histogram(series, initial_bin_width=10):
    """
    Ref: https://towardsdatascience.com/how-to-quickly-find-the-best-bin-width-for-your-histogram-4d8532f053b0
    """
    figure_widget = go.FigureWidget(
        data=[go.Histogram(x=series, xbins={"size": initial_bin_width})]
    )

    bin_slider = widgets.FloatSlider(
        value=initial_bin_width,
        min=1,
        max=30,
        step=1,
        description="Bin width:",
        readout_format=".0f",  # display as integer
    )

    histogram_object = figure_widget.data[0]

    def set_bin_size(change):
        histogram_object.xbins = {"size": change["new"]}

    bin_slider.observe(set_bin_size, names="value")

    output_widget = widgets.VBox([figure_widget, bin_slider])
    return output_widget

def plot_histogram_comparison(dfs:pd.DataFrame, dfv:pd.DataFrame,
                   features:List, title:str)->None:
    num_graphs = len(features) * 2
    num_cols = 2
    num_rows = num_graphs // num_cols
    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        column_name = features[i//2]
        if i%2 == 0:
            sns.histplot(data = dfs, x = column_name, 
                         ax = ax)
        else:
            sns.histplot(data = dfv, x = column_name, 
                         ax = ax)
    plt.suptitle(title)

def plot_scatter_comparison(dfs:pd.DataFrame, dfv:pd.DataFrame, 
                 features:List, title:str)-> None:
    """
    Function to display scatter plots between columns of two DataFrames

    Args:
        dfs (pd.DataFrame): Pandas DataFrame A
        dfv (pd.DataFrame): Pandas DataFrame B
        features (List): Features List
        title (str): Plot Title
        
    Returns: None
    """
    
    feature_combinations = list(itertools.combinations(features, 2))

    num_graphs = len(feature_combinations) * 2
    num_cols = 2
    num_rows = num_graphs // num_cols

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        x_col = feature_combinations[i // 2][0]
        y_col = feature_combinations[i // 2][1]
        if i%2 == 0:
            sns.scatterplot(data = dfs, x = x_col, y = y_col, 
                            ax = ax)
        else:
            sns.scatterplot(data = dfv, x = x_col, y = y_col, 
                            ax = ax)
        
    plt.suptitle(title)

def plot_density_comparison(df_a:pd.DataFrame, df_b:pd.DataFrame,
                            features:List, title:str,
                            df_a_name:str = "A", 
                            df_b_name:str = "B")->None:
    """
    Function to display density plots between columns of two DataFrames

    Args:
        dfs (pd.DataFrame): Pandas DataFrame A
        dfv (pd.DataFrame): Pandas DataFrame B
        features (List): Features List
        title (str): Plot Title
        
    Returns: None
    """
    
    if len(df_a.columns) != len(df_b.columns):
        raise Exception("Both datasets should have same number of columns")
    
    
    num_cols = min(4, len(features))
    num_rows = int(np.ceil(len(features) / num_cols))
    #print(num_rows)
    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i <= len(features)-1:
            column_name = features[i]
            labels = [df_a_name] * len(df_a) + [df_b_name] * len(df_b)
            #print(dfs[column_name])
            combined_data = pd.concat([df_a[column_name], df_b[column_name]])
            combined_df = pd.DataFrame({f"{column_name}": combined_data, 'Data': labels})
            #print(combined_df.columns)
            sns.kdeplot(data=combined_df, x=column_name, 
                        hue = labels, common_norm=False ,ax = ax)
    plt.suptitle(title) 

def plot_count_comparison(df_a:pd.DataFrame, df_b:pd.DataFrame,
                            features:List, title:str,
                            df_a_name:str = "A", 
                            df_b_name:str = "B")->None:
    """
    Function to display count plots between columns of two DataFrames

    Args:
        dfs (pd.DataFrame): Pandas DataFrame A
        dfv (pd.DataFrame): Pandas DataFrame B
        features (List): Features List
        title (str): Plot Title
        
    Returns: None
    """
    
    if len(df_a.columns) != len(df_b.columns):
        raise Exception("Both datasets should have same number of columns")
    
    
    num_cols = min(4, len(features))
    num_rows = int(np.ceil(len(features) / num_cols))
    #print(num_rows)
    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i <= len(features)-1:
            column_name = features[i]
            labels = [df_a_name] * len(df_a) + [df_b_name] * len(df_b)
            #print(dfs[column_name])
            combined_data = pd.concat([df_a[column_name], df_b[column_name]])
            combined_df = pd.DataFrame({f"{column_name}": combined_data, 'Data': labels})
            #print(combined_df.columns)
            combined_df.groupby([column_name,"Data"]).size().reset_index().pivot(columns="Data", index=column_name, values=0).plot.barh(ax = ax)
    plt.tight_layout()
    plt.suptitle(title) 


def plot_pca_comparison(df_a:pd.DataFrame, df_b:pd.DataFrame,
                        title:str,
                        df_a_name:str = "A", 
                        df_b_name:str = "B")->None:
    """
    Function to reduce the dimensions of two dataframes using PCA with 2 components and plot them.
    Args:
        df_a (pd.DataFrame): Pandas DataFrame A
        df_b (pd.DataFrame): Pandas DataFrame B
        title (str): Plot Title
        df_a_name (str): DataFrame A column Label. Defaults to "A".
        df_b_name (str): DataFrame B column Label. Defaults to "B".
    """
        # Initialize the label encoder
    label_encoder = LabelEncoder()
    
    df_a = df_a.copy()
    df_b = df_b.copy()
    
    # Encode non-numeric columns for data set a
    non_numeric_cols_a = df_a.select_dtypes(exclude=['int', 'float', 'bool']).columns
    if np.size(non_numeric_cols_a) > 0:
        for col in non_numeric_cols_a:
            df_a[col] = label_encoder.fit_transform(df_a[col])
    
    # Encode non-numeric columns for data set b
    non_numeric_cols_b = df_b.select_dtypes(exclude=['int', 'float', 'bool']).columns
    if np.size(non_numeric_cols_b) > 0:
        for col in non_numeric_cols_b:
            df_b[col] = label_encoder.fit_transform(df_b[col])

    pca_a = PCA(n_components = 2)
    pca_b = PCA(n_components = 2)
    
    pca_a_t = pca_a.fit_transform(df_a)
    pca_b_t = pca_b.fit_transform(df_b)
    
    _, ax = plt.subplots(1,2)
    
    sns.scatterplot(x = pca_a_t[:,0], y = pca_a_t[:,1], 
                    ax = ax[0], label = df_a_name)
    sns.scatterplot(x = pca_b_t[:,0], y = pca_b_t[:,1], 
                    ax = ax[1], label = df_b_name)
    plt.suptitle(title)
    plt.show()    

def plot_mca_comparison(df_a:pd.DataFrame, df_b:pd.DataFrame,
                        title:str,
                        df_a_name:str = "A", 
                        df_b_name:str = "B")->None:
    """
    Function to reduce the dimensions of two dataframes using PCA with 2 components and plot them.
    Args:
        df_a (pd.DataFrame): Pandas DataFrame A
        df_b (pd.DataFrame): Pandas DataFrame B
        title (str): Plot Title
        df_a_name (str): DataFrame A column Label. Defaults to "A".
        df_b_name (str): DataFrame B column Label. Defaults to "B".
    """
    # Initialize the label encoder
    label_encoder = LabelEncoder()
    
    df_a = df_a.copy()
    df_b = df_b.copy()
    
    # Encode non-numeric columns for data set a
    non_numeric_cols_a = df_a.select_dtypes(exclude=['int', 'float', 'bool']).columns
    if np.size(non_numeric_cols_a) > 0:
        for col in non_numeric_cols_a:
            df_a[col] = label_encoder.fit_transform(df_a[col])
    
    # Encode non-numeric columns for data set b
    non_numeric_cols_b = df_b.select_dtypes(exclude=['int', 'float', 'bool']).columns
    if np.size(non_numeric_cols_b) > 0:
        for col in non_numeric_cols_b:
            df_b[col] = label_encoder.fit_transform(df_b[col])
            
    # sc = StandardScaler()
    
    # df_a_scaled = sc.fit_transform(df_a)
    # df_b_scaled = sc.fit_transform(df_b)

    mca_a = MCA(n_components = 2)
    mca_b = MCA(n_components = 2)
    
    mca_a_t = mca_a.fit_transform(df_a)
    mca_b_t = mca_b.fit_transform(df_b)
    
    _, ax = plt.subplots(1,2)
    
    sns.scatterplot(x = mca_a_t[:,0], y = mca_a_t[:,1], 
                    ax = ax[0], label = df_a_name)
    sns.scatterplot(x = mca_b_t[:,0], y = mca_b_t[:,1], 
                    ax = ax[1], label = df_b_name)
    plt.suptitle(title)
    plt.show()    

def get_unique_keys(d, parent_key=''):
    unique_keys = []

    for key, value in d.items():
        current_key = f"{parent_key}_{key}" if parent_key else key
        if isinstance(value, dict):
            unique_keys.extend(get_unique_keys(value, current_key))
        else:
            unique_keys.append(current_key)

    return unique_keys

def composite_metrics(df_a: pd.DataFrame, df_b: pd.DataFrame) -> float:
    """Function to calculate a composite metric based on different metrics for similarity

    Args:
        df_a (pd.DataFrame): Pandas DataFrame
        df_b (pd.DataFrame): Pandas DataFrame

    Raises:
        TypeError: Throws error if any of the Input Datasets is not a Pandas DataFRame
        ValueError: Throws error if any of the Input Datasets is empty
        ValueError: Throws error if there are special characters or spaces in the columns of 
                    the Input Datasets
        ValueError: Throws error if any of the column names in the Input Datasets do not 
                    match

    Returns:
        float: Composite Metric value
    """
    if not isinstance(df_a, pd.DataFrame) or not isinstance(df_b, pd.DataFrame):
        raise TypeError("Both input datasets should be Pandas DataFrames")
    
    if df_a.empty or df_b.empty:
        raise ValueError("The Input Datasets should not be empty")
    
    if re.search(r'[^a-zA-Z0-9_]', "".join(df_a.columns)) or re.search(r'[^a-zA-Z0-9_]', "".join(df_b.columns)):
        raise ValueError("There are special characters or space in the Column Names of Input Datasets. Please clean them before processing.")
    
    if set(df_a.columns) != set(df_b.columns):
        raise ValueError("There are differences between column names in the Input datasets")
    
    #Calculate Multiple Metrics
    # Define and calculate individual metrics
    metrics = {}
    categorical_features = df_a.select_dtypes(exclude = np.number).columns
    numerical_features = df_a.select_dtypes(include = np.number).columns

    # Metric 1: Mean Absolute Error for Numerical Columns
    mae = mean_absolute_error(df_a[numerical_features], df_b[numerical_features])
    metrics['MAE'] = mae

    # Metric 2: KS Statistic for Categorical Columns
    ks_statistics = {}
    for col in categorical_features:
        ks_statistic, _ = ks_2samp(df_a[col], df_b[col])
        ks_statistics[col] = ks_statistic
    metrics['KS_Statistic'] = ks_statistics

    # Metric 3: Accuracy for Categorical Columns (classification)
    accuracy_scores = {}
    for col in categorical_features:
        accuracy = accuracy_score(df_a[col], df_b[col])
        accuracy_scores[col] = accuracy
    metrics['Accuracy'] = accuracy_scores

    # Metric 4: Wasserstein Distance for Numerical Columns
    wasserstein_distances = {}
    for col in numerical_features:
        w_distance = wasserstein_distance(df_a[col], df_b[col])
        wasserstein_distances[col] = w_distance
    metrics['Wasserstein_Distance'] = wasserstein_distances

    # Metric 5: Jensen-Shannon Divergence for Categorical Columns
    js_divergences = {}
    for col in categorical_features:
        data_onehot_a = pd.get_dummies(df_a[col])
        data_onehot_b = pd.get_dummies(df_b[col])
        js_distance = jensenshannon(data_onehot_a.mean(axis=0), data_onehot_b.mean(axis=0))
        js_divergences[col] = js_distance
    metrics['JS_Divergence'] = js_divergences
    
    # # Calculate the composite metric (weighted sum)
    # composite_metric = sum(weights[key] * metrics[key] for key in weights if key in metrics)

    flatten_metrics = flatten(metrics, reducer="underscore")
    metrics_lst = np.array([v for k, v in flatten_metrics.items()])
    composite_metric = np.mean(metrics_lst/np.sum(metrics_lst))
    
    return composite_metric
    # # Display the composite metric
    # print("Composite Metric:", composite_metric)


def plot_histogram(data:pd.DataFrame, col:str, bins:int= 10)->None:
    sns.histplot(data=data, x = col, bins = bins)