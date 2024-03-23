import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go

def hex_to_rgba(h, alpha):
    '''
    converts color value in hex format to rgba format with alpha transparency
    '''
    return tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha])

def plot_group_regression_lines(df, y_col='DISTS', x_col='MOS', group_col='reference_folder'):
    colors = [
        '#1f77b4',  # Mutated blue
        '#ff7f0e',  # Safety orange
        '#2ca02c',  # Cooked asparagus green
        '#d62728',  # Brick red
        '#9467bd',  # Muted purple
        '#8c564b',  # Chestnut brown
        '#e377c2',  # Raspberry yogurt pink
        '#7f7f7f',  # Middle gray
        '#bcbd22',  # Curry yellow-green
        '#17becf'   # Blue-teal
    ]

    def linear_func(x, a, b):
        return a + b * x

    fig = go.Figure()

    unique_groups = df[group_col].unique()
    for i, group in enumerate(unique_groups):
        group_df = df[df[group_col] == group]
        group_x = group_df[x_col]
        group_y = group_df[y_col]
        
        params, params_covariance = curve_fit(linear_func, group_x, group_y)
        
        x_range = np.linspace(min(group_x), max(group_x), 400)
        y_pred = linear_func(x_range, *params)
        
        color = colors[i % len(colors)]
        rgba_color = hex_to_rgba(color, 0.5)  # Adjust the alpha value to 0.5 for transparency
        
        fig.add_trace(go.Scatter(x=group_x, y=group_y, mode='markers', name=f'Data: {group}', marker_color=color))
        fig.add_trace(go.Scatter(x=group_x, y=group_y, mode='lines', line=dict(color=rgba_color, width=2)))
        
        fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name=f'Regression: {group}', line=dict(color=color)))

    fig.update_layout(title=f'Linear Regression per Group between {y_col} and {x_col}',
                      xaxis_title=x_col,
                      yaxis_title=y_col)
    return fig


# %%
