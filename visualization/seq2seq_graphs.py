"""Configuration-driven Graphviz paths and rectangular attention heatmaps."""
import plotly.graph_objects as go


def attention_figure(values, source_labels, target_labels, title, probability=True):
    values = values.detach().cpu().numpy()
    figure = go.Figure(go.Heatmap(z=values, x=source_labels, y=target_labels,
        zmin=0 if probability else None, zmax=1 if probability else None,
        colorscale='Blues', hovertemplate='Query %{y}<br>Key %{x}<br>Value %{z:.6f}<extra></extra>'))
    figure.update_layout(title=title, xaxis_title='Key tokens', yaxis_title='Query tokens', height=360)
    figure.update_yaxes(autorange='reversed')
    return figure


def architecture_diagram(spec):
    lines = ['digraph transformer {', 'rankdir=TB;',
             'graph [ranksep=0.3,nodesep=0.4];',
             'node [shape=box,style=rounded,fontsize=12]; edge [fontsize=10];',
             'subgraph cluster_encoder { label="Encoder: source stream";',
             f'src [label="Source IDs\\nEmbeddings + positions\\nDropout({spec.dropout})"];']
    pre = spec.block.norm == 'Pre-LayerNorm'

    def layer(prefix, previous, operations):
        for j, operation in enumerate(operations):
            name = f'{prefix}_{j}'
            operation = f'{operation}\\nDropout({spec.dropout})'
            if pre:
                lines.extend([f'{name}n [label="LN{j+1}"];', f'{name}a [label="{operation}"];',
                              f'{name}r [label="Add"];',
                              f'{previous} -> {name}n -> {name}a -> {name}r;',
                              f'{previous} -> {name}r [style=dashed,label="residual"];'])
                previous = name+'r'
            else:
                lines.extend([f'{name}a [label="{operation}"];', f'{name}r [label="Add"];',
                              f'{name}n [label="LN{j+1}"];',
                              f'{previous} -> {name}a -> {name}r -> {name}n;',
                              f'{previous} -> {name}r [style=dashed,label="residual"];'])
                previous = name+'n'
        return previous

    previous = 'src'
    for i in range(spec.encoder_layers):
        previous = layer(f'e{i}', previous, ('Bidirectional self-attention', 'FFN'))
    lines.extend([f'{previous} -> memory;', 'memory [label="Encoder memory [B,S,D]"];', '}',
                  'subgraph cluster_decoder { label="Decoder: target stream";',
                  f'tgt [label="Target prefix IDs\\nEmbeddings + positions\\nDropout({spec.dropout})"];'])
    previous = 'tgt'
    for i in range(spec.decoder_layers):
        previous = layer(f'd{i}', previous, ('Causal self-attention', 'Cross-attention', 'FFN'))
    lines.extend([f'{previous} -> projection -> logits;',
                  f'projection [label="Vocabulary projection: D → {spec.target_vocabulary}"];',
                  'logits [label="Logits [B,T,V] → softmax"];', '}'])
    for i in range(spec.decoder_layers):
        lines.append(f'memory -> d{i}_1a [label="K,V"];')
    lines.append('}')
    return '\n'.join(lines)
