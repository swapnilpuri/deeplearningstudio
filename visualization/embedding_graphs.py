"""Reusable Plotly views; numeric projection/ranking is supplied by computation."""
import numpy as np
import plotly.graph_objects as go


def matrix_figure(values, title, labels=None):
    values = np.asarray(values)
    if values.ndim == 1: values = values[None,:]
    return go.Figure(go.Heatmap(z=values, y=labels, colorscale='RdBu', zmid=0)).update_layout(
        title=title, xaxis_title='Coordinate / column', yaxis_title='Token / row')


def space_figure(coordinates, labels, selected=None, neighbors=(), title='Embedding space'):
    coordinates = np.asarray(coordinates)
    colors = ['#ef553b' if i==selected else '#00a878' if i in neighbors else '#636efa' for i in range(len(labels))]
    sizes = [14 if i==selected else 9 for i in range(len(labels))]
    is3d = coordinates.shape[1] == 3
    kwargs = dict(x=coordinates[:,0], y=coordinates[:,1], text=labels, mode='markers+text',
                  marker=dict(color=colors,size=sizes), textposition='top center')
    if is3d: kwargs['z'] = coordinates[:,2]
    figure = go.Figure(go.Scatter3d(**kwargs) if is3d else go.Scatter(**kwargs))
    if selected is not None:
        for neighbor in neighbors:
            pair = coordinates[[selected,neighbor]]
            line = dict(x=pair[:,0],y=pair[:,1],mode='lines',line=dict(color='#aaaaaa'),showlegend=False,hoverinfo='skip')
            if is3d: line['z'] = pair[:,2]
            figure.add_trace(go.Scatter3d(**line) if is3d else go.Scatter(**line))
    return figure.update_layout(title=title, height=500, xaxis_title='Axis 1', yaxis_title='Axis 2')


def curves_figure(history, keys, title):
    figure = go.Figure()
    for key in keys:
        figure.add_trace(go.Scatter(x=list(range(1,len(history)+1)),y=[row[key] for row in history],name=key,mode='lines+markers'))
    return figure.update_layout(title=title,xaxis_title='Epoch',yaxis_title='Measured value')


def ranking_figure(rows, title='Exact retrieval ranking'):
    return go.Figure(go.Bar(x=[row['score'] for row in rows],y=[row['token'] for row in rows],orientation='h')).update_layout(
        title=title,yaxis=dict(autorange='reversed'),xaxis_title='Similarity score')


def architecture_graph(rows):
    nodes = [f'n{i} [label="{row["operator"]}\\n{row["shape"]}\\nP={row["parameters"]}"];' for i,row in enumerate(rows)]
    return 'digraph {rankdir=LR; node [shape=box]; '+ ' '.join(nodes)+ ' '.join(f'n{i}->n{i+1};' for i in range(len(rows)-1))+'}'
