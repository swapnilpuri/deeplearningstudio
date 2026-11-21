import plotly.graph_objects as go

def network_figure(sizes):
    # Show a representative subset to prevent 784-input MNIST networks from freezing the browser.
    max_visible = 20
    visible = [min(n, max_visible) for n in sizes]
    positions = [[(i, j-(n-1)/2) for j in range(n)] for i,n in enumerate(visible)]
    ex, ey = [], []
    for left,right in zip(positions, positions[1:]):
        for a in left:
            for b in right:
                ex.extend([a[0],b[0],None]); ey.extend([a[1],b[1],None])
    fig = go.Figure(go.Scatter(x=ex,y=ey,mode='lines',line=dict(color='rgba(130,145,170,0.23)',width=1),hoverinfo='skip',showlegend=False))
    for i,points in enumerate(positions):
        label = 'Input' if i==0 else 'Output' if i==len(sizes)-1 else f'Hidden {i}'
        color = '#2563eb' if i==0 else '#059669' if i==len(sizes)-1 else '#8b5cf6'
        fig.add_trace(go.Scatter(x=[p[0] for p in points],y=[p[1] for p in points],mode='markers+text',text=[str(j+1) if sizes[i]<=max_visible else ('…' if j==len(points)-1 else str(j+1)) for j in range(len(points))],textposition='middle center',textfont=dict(color='white',size=10),marker=dict(size=25,color=color),name=label,hovertemplate=f'{label} neuron %{{text}}<extra></extra>',showlegend=False))
    fig.update_layout(height=max(380,34*max(visible)+100),margin=dict(l=25,r=25,t=35,b=65),xaxis=dict(tickmode='array',tickvals=list(range(len(sizes))),ticktext=[f'{"Input" if i==0 else "Output" if i==len(sizes)-1 else f"Hidden {i}"}<br>{n} neurons' for i,n in enumerate(sizes)],showgrid=False,zeroline=False,range=[-.45,len(sizes)-.55]),yaxis=dict(visible=False),dragmode='pan')
    return fig
