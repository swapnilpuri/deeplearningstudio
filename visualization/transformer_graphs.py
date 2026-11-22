"""Small adapters over existing Plotly attention figures."""
from plotly.subplots import make_subplots
from .attention_graphs import attention_figure


def head_figure(probabilities, labels, title):
    return attention_figure(probabilities, labels, title).update_traces(zmin=0, zmax=1)


def all_heads_figure(probabilities, labels):
    count = probabilities.shape[0]
    figure = make_subplots(rows=(count+1)//2, cols=2,
                           subplot_titles=[f'Head {i+1}' for i in range(count)])
    for i, values in enumerate(probabilities):
        heatmap = head_figure(values, labels, '').data[0]
        heatmap.update(showscale=(i==0))
        figure.add_trace(heatmap, row=i//2+1, col=i%2+1)
    figure.update_yaxes(autorange='reversed')
    figure.update_layout(title='Independent head probabilities', height=300*((count+1)//2))
    return figure


def block_diagram(norm: str) -> str:
    if norm == 'Pre-LayerNorm':
        return '''digraph block {
rankdir=LR; X [label="Input X"]; N1 [label="LN1"]; A [label="MHA"];
U [label="Add: U"]; N2 [label="LN2"]; F [label="FFN"]; Y [label="Add: Output Y"];
X -> N1 -> A -> U -> N2 -> F -> Y;
X -> U [style=dashed,label="residual"]; U -> Y [style=dashed,label="residual"];
}'''
    return '''digraph block {
rankdir=LR; X [label="Input X"]; A [label="MHA"]; R1 [label="Add"];
U [label="LN1: U"]; F [label="FFN"]; R2 [label="Add"]; Y [label="LN2: Output Y"];
X -> A -> R1 -> U -> F -> R2 -> Y;
X -> R1 [style=dashed,label="residual"]; U -> R2 [style=dashed,label="residual"];
}'''
