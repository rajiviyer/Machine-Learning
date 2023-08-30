from typing import Union, List
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import ipywidgets as widgets
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

def get_cleaned_students_data():
    data = pd.read_csv("data/student_success.csv",sep = ";")
    data.columns = data.columns.str.lower().str.replace("[/,\t,(,),']","", regex = True).str.replace(" ", "_")
    for col in data.select_dtypes(include=[object]).columns:
        data[col] = label_encode(data[col])
    return data
    
def label_encode(categorical_data: Union[list, pd.Series])->List:
    unique_values = list(set(categorical_data))
    label_mapping = {value: index for index, value in enumerate(unique_values)}

    # Encode the categorical data using the label mapping
    encoded_data = [label_mapping[value] for value in categorical_data]
    
    # Reverse transformation to get back original labels
    # decoded_data = [value for index in encoded_data for value, idx 
    #                 in label_mapping.items() if idx == index]
    return encoded_data

def stratified_train_test_split(df, target_column, train_size=0.8, 
                                random_state=None):
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
                 features:List, title:str):
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
    if len(df_a.columns) != len(df_b.columns):
        raise Exception("Both datasets should have same number of columns")
    
    
    num_cols = 4
    num_rows = len(features) // num_cols
    #print(num_rows)
    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        column_name = features[i]
        labels = [df_a_name] * len(df_a) + [df_b_name] * len(df_b)
        #print(dfs[column_name])
        combined_data = pd.concat([df_a[column_name], df_b[column_name]])
        combined_df = pd.DataFrame({f"{column_name}": combined_data, 'Data': labels})
        #print(combined_df.columns)
        sns.kdeplot(data=combined_df, x=column_name, 
                    hue = labels, common_norm=False ,ax = ax)
    plt.suptitle(title) 

def plot_histogram(data:pd.DataFrame, col:str, bins:int= 10)->None:
    sns.histplot(data=data, x = col, bins = bins)
    
# def plot_histogram(data:pd.DataFrame):
#     # Calculate the number of rows and columns for subplots
#     num_rows = (data.shape[1] + 3) // 4  # Number of rows required
#     num_cols = min(data.shape[1], 4)  # Number of columns in each row

#     # Create subplots
#     fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
#     axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

#     # Plot histograms for each column in subplots
#     for i, col in enumerate(data.columns):
#         ax = axes[i]
#         data[col].plot(kind='hist', ax=ax, bins=10, edgecolor='black')
#         ax.set_title(col)
#         ax.set_xlabel('Value')
#         ax.set_ylabel('Frequency')

#     # Remove any unused subplots
#     for i in range(len(data.columns), len(axes)):
#         axes[i].remove()

#     plt.tight_layout()
#     plt.show()    