"""Plotly graphs for the training laboratory."""
import numpy as np
import plotly.graph_objects as go
import torch


def dataset_figure(exp):
    x = exp.x.numpy()
    y = exp.y.numpy()
    fig = go.Figure()
    if exp.classification:
        for label in (0, 1):
            mask = y == label
            fig.add_trace(go.Scatter(x=x[mask, 0], y=x[mask, 1], mode='markers', name=f'Class {label}', marker={'size': 7, 'opacity': 0.75}))
        fig.update_layout(xaxis_title='Feature 1', yaxis_title='Feature 2')
    else:
        fig.add_trace(go.Scatter(x=x[:, 0], y=y[:, 0], mode='markers', name='Samples', marker={'opacity': 0.65, 'size': 7}))
        xx = torch.linspace(float(x.min()), float(x.max()), 250).reshape(-1, 1)
        predictions = exp.predict(xx).detach().numpy().flatten()
        fig.add_trace(go.Scatter(x=xx.numpy().flatten(), y=predictions, mode='lines', name='Model prediction', line={'width': 3}))
        fig.update_layout(xaxis_title='Input', yaxis_title='Target / prediction')
    fig.update_layout(height=400, margin={'l': 15, 'r': 15, 't': 15, 'b': 25}, legend={'orientation': 'h'})
    return fig


def decision_figure(exp, resolution=85):
    if not exp.classification:
        raise ValueError('Decision boundary applies to classification only')
    x = exp.x.numpy()
    y = exp.y.numpy()
    x0 = np.linspace(float(x[:, 0].min()) - .4, float(x[:, 0].max()) + .4, resolution)
    x1 = np.linspace(float(x[:, 1].min()) - .4, float(x[:, 1].max()) + .4, resolution)
    xx, yy = np.meshgrid(x0, x1)
    with torch.no_grad():
        grid = torch.tensor(np.column_stack((xx.ravel(), yy.ravel())), dtype=torch.float32)
        probabilities = exp.predict(grid).softmax(dim=1)[:, 1].numpy().reshape(xx.shape)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=x0, y=x1, z=probabilities, zmin=0, zmax=1, colorscale='RdBu', reversescale=True, opacity=0.68, colorbar={'title': 'P(class 1)'}))
    for label in (0, 1):
        mask = y == label
        fig.add_trace(go.Scatter(x=x[mask, 0], y=x[mask, 1], mode='markers', name=f'Class {label}', marker={'size': 6, 'color': '#1d4ed8' if label == 0 else '#b91c1c', 'line': {'color': 'white', 'width': .5}}))
    fig.update_layout(height=440, margin={'l': 15, 'r': 15, 't': 15, 'b': 25}, xaxis_title='Feature 1', yaxis_title='Feature 2')
    return fig


def history_figure(exp, key='loss'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r['epoch'] for r in exp.history], y=[r[key] for r in exp.history], mode='lines+markers', name=key, line={'width': 3}, marker={'size': 4}))
    fig.update_layout(height=320, margin={'l': 15, 'r': 15, 't': 12, 'b': 25}, xaxis_title='Epoch', yaxis_title={'loss': 'Loss', 'score': 'Accuracy' if exp.classification else 'RMSE', 'grad_norm': 'Gradient norm', 'val_loss': 'Validation loss', 'val_score': 'Validation accuracy' if exp.classification else 'Validation RMSE'}[key])
    return fig
