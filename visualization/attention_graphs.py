"""Attention visualization preparation; position labels preserve duplicate tokens."""
import plotly.graph_objects as go


def attention_figure(values, labels, title):
    return go.Figure(go.Heatmap(z=values.detach().cpu().numpy(), x=labels, y=labels,
        colorscale='Viridis', hovertemplate='Query %{y}<br>Key %{x}<br>Value %{z:.5f}<extra></extra>')).update_layout(
        title=title, xaxis_title='Key position', yaxis_title='Query position', yaxis_autorange='reversed')


def distribution_figure(probabilities, labels, query):
    return go.Figure(go.Bar(x=labels, y=probabilities[query].detach().cpu().numpy())).update_layout(
        title=f'Routing from {labels[query]}', xaxis_title='Key token', yaxis_title='Attention probability', yaxis_range=[0,1])
