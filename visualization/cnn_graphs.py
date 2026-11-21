"""Reusable Plotly tensor/architecture graphics; no neural-network computation."""
import numpy as np
import plotly.graph_objects as go


def heatmap(values,title='',highlight=None):
    a=np.asarray(values)
    if a.ndim!=2: raise ValueError('heatmap expects 2D array')
    fig=go.Figure(go.Heatmap(z=a,colorscale='RdBu',zmid=0,text=np.round(a,3),texttemplate='%{text}',
                            colorbar={'title':'value'}))
    if highlight is not None:
        top,left,h,w=highlight
        fig.add_shape(type='rect',x0=left-.5,x1=left+w-.5,y0=top-.5,y1=top+h-.5,
                      line={'color':'limegreen','width':4})
    fig.update_yaxes(autorange='reversed',scaleanchor='x')
    fig.update_layout(title=title,height=max(280,min(600,40*len(a)+100)),margin=dict(l=20,r=20,t=60,b=20))
    return fig


def architecture_diagram(rows):
    labels=[f"{row['operator']}<br>{row['shape']}<br>{row['parameters']:,} params" for row in rows]
    n=len(labels)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=list(range(n)),y=[0]*n,mode='lines+markers+text',
                             line={'width':2},marker={'size':26},text=labels,textposition='top center',
                             hovertemplate='%{text}<extra></extra>'))
    fig.update_xaxes(visible=False,range=[-.5,n-.5]);fig.update_yaxes(visible=False,range=[-.8,.8]);
    fig.update_layout(height=280,margin=dict(l=15,r=15,t=90,b=15),showlegend=False)
    return fig
