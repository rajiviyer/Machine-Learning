from typing import Union, List
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import ipywidgets as widgets
import matplotlib.pyplot as plt

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

def plot_histogram(data:pd.DataFrame):
    # Calculate the number of rows and columns for subplots
    num_rows = (data.shape[1] + 3) // 4  # Number of rows required
    num_cols = min(data.shape[1], 4)  # Number of columns in each row

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

    # Plot histograms for each column in subplots
    for i, col in enumerate(data.columns):
        ax = axes[i]
        data[col].plot(kind='hist', ax=ax, bins=10, edgecolor='black')
        ax.set_title(col)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

    # Remove any unused subplots
    for i in range(len(data.columns), len(axes)):
        axes[i].remove()

    plt.tight_layout()
    plt.show()    