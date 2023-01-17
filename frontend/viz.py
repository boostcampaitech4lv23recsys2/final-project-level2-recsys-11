import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go

def histogram(df, option=None):
    fig, ax = plt.subplots()
    if option is None:
        ax.set_title('Set column')
    else:
        ax.set_title(option)
    ax.hist(df)
    
    return fig
    
def histogram(df, option=None):
    fig, ax = plt.subplots()
    ax.hist(df)
    
    return fig

def plot_multi_area(plot_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.item, y=plot_df.all_count,
                            fill = "tozeroy", mode = "none"))
    fig.add_trace(go.Scatter(x=plot_df.item, y=plot_df.rec_count,
                            fill = "tonexty", mode = "none"))
    fig.update_layout(
        xaxis_type = 'category'
    )
    return fig