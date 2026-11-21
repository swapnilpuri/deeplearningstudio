"""Reusable figures for learning modules (CNN/Transformer can reuse tensor and gradient views)."""
import plotly.graph_objects as go

def tensor_heatmap(values, title='Tensor values'):
    import numpy as np
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError('Expected 1D or 2D tensor')
    fig = go.Figure(go.Heatmap(z=matrix, colorscale='RdBu', zmid=0))
    fig.update_layout(title=title, xaxis_title='Feature index', yaxis_title='Sample index', height=260)
    return fig

def gradient_bar(updates):
    fig = go.Figure(go.Bar(x=[u['parameter'] for u in updates], y=[u['gradient_norm'] for u in updates]))
    fig.update_layout(title='Gradient norm per parameter (after backward)', xaxis_title='Parameter',
                      yaxis_title='L2 norm', height=330)
    return fig
