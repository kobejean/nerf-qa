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
    return "rgba" +str(tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha]))


def plot_group_regression_lines(df, y_col='MOS', x_col='DISTS', group_col='reference_folder'):
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
    # Define the logistic function with parameters β1 to β4
    def logistic(x, beta1, beta2, beta3, beta4):
        return (beta1 - beta2) / (1 + np.exp(-(x - beta3) / abs(beta4))) + beta2

    x = df[x_col]
    y = df[y_col]
    # Initial parameter guesses
    beta1_init = np.max(y)
    beta2_init = np.min(y)
    beta3_init = np.mean(x)
    beta4_init = np.std(x) / 4

    fig = go.Figure()

    unique_groups = df[group_col].unique()
    for i, group in enumerate(unique_groups):
        group_df = df[df[group_col] == group]
        group_x = group_df[x_col]
        group_y = group_df[y_col]
        
        params, params_covariance = curve_fit(logistic, group_x, group_y, p0=[beta1_init, beta2_init, beta3_init, beta4_init])
        
        x_range = np.linspace(min(group_x), max(group_x), 400)
        y_pred = logistic(x_range, *params)
        
        color = colors[i % len(colors)]
        rgba_color = hex_to_rgba(color, 0.2)  # Adjust the alpha value to 0.5 for transparency
        
        fig.add_trace(go.Scatter(x=group_x, y=group_y, mode='markers', name=f'Data: {group}', marker_color=color))
        fig.add_trace(go.Scatter(x=group_x, y=group_y, mode='lines', line=dict(color=rgba_color, width=1)))
        
        fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name=f'Regression: {group}', line=dict(color=color)))

    fig.update_layout(title=f'Linear Regression per Group between {y_col} and {x_col}',
                      xaxis_title=x_col,
                      yaxis_title=y_col)
    return fig


# %%
