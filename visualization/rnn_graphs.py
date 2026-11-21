"""Reusable time/state/gradient plots; computation remains in adapters."""
import numpy as np
import plotly.graph_objects as go


def state_heatmap(values, title='Hidden state across time'):
    values = np.asarray(values)
    return go.Figure(go.Heatmap(z=values.T,x=list(range(1,len(values)+1)),colorscale='RdBu',zmid=0)).update_layout(
        title=title,xaxis_title='Time step',yaxis_title='Feature / hidden unit')


def unrolled_figure(inputs, states, selected=0, outputs=None):
    inputs,states = np.asarray(inputs),np.asarray(states)
    t = len(states); fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(t)),y=[1]*t,mode='lines+markers+text',
                            marker={'size':22,'color':['#e57a23' if i==selected else '#d4ac0d' if i==selected-1 else '#2f81b7' for i in range(t)]},
                            text=[f'h{i+1}' for i in range(t)],textposition='top center',
                            customdata=states,hovertemplate='h: %{customdata}<extra></extra>',name='Shared recurrent weights'))
    for i in range(t):
        fig.add_annotation(x=i,y=.85,ax=i,ay=.15,axref='x',ayref='y',xref='x',yref='y',showarrow=True,text='',arrowhead=2)
        if i:
            fig.add_annotation(x=i-.15,y=1,ax=i-.85,ay=1,axref='x',ayref='y',xref='x',yref='y',showarrow=True,text='',arrowhead=2)
        fig.add_trace(go.Scatter(x=[i,i],y=[0,1],mode='lines+markers',showlegend=False,
                                text=[str(inputs[i].round(3)),str(states[i].round(3))],hovertemplate='%{text}<extra></extra>'))
    fig.add_trace(go.Scatter(x=list(range(t)),y=[0]*t,mode='text',text=[f'x{i+1}' for i in range(t)],showlegend=False))
    if outputs is not None:
        for i in range(t):
            fig.add_trace(go.Scatter(x=[i,i],y=[1,2],mode='lines+markers',showlegend=False,
                                    text=['',str(np.asarray(outputs)[i].round(3))],hovertemplate='%{text}<extra></extra>'))
    return fig.update_layout(title=f'Unrolled recurrence — step {selected+1}; h{selected} → h{selected+1}; shared W_ih / W_hh',
                             xaxis_title='Zero-based time index',yaxis={'visible':False},height=330)


def gradient_figure(norms, boundaries=()):
    fig = go.Figure(go.Bar(x=list(range(1,len(norms)+1)),y=norms,name='||dL/dh_t||'))
    for boundary in boundaries:
        fig.add_vline(x=boundary+.5,line_dash='dash',annotation_text='detach')
    return fig.update_layout(title='Gradient flow from terminal loss',xaxis_title='Time step',yaxis_title='Gradient norm (zeros mean disconnected)')


def prediction_figure(target, prediction):
    fig = go.Figure()
    for name,values in [('Target',target),('Prediction',prediction)]:
        fig.add_trace(go.Scatter(y=np.asarray(values).reshape(-1),mode='lines+markers',name=name))
    return fig.update_layout(title='Prediction versus target',xaxis_title='Forecast step / held-out example',yaxis_title='Value')


def recurrent_architecture(rows):
    return 'digraph RNN { rankdir=LR; '+ ' '.join(
        f'n{i} [label="{row["operator"]}\\n{row["shape"]}\\n{row["parameters"]} parameters"];'+(f'n{i-1} -> n{i};' if i else '')
        for i,row in enumerate(rows))+' }'
