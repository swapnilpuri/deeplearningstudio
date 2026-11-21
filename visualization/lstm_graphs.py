"""Gate/state plots composed from actual computed tensors."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def state_panels(values, unit=0):
    names = ('input_vector','forget','input','candidate','cell','output','hidden')
    fig = make_subplots(rows=4,cols=2,subplot_titles=names)
    for index,name in enumerate(names):
        data = values[name].detach().cpu().numpy()
        column = min(unit,data.shape[-1]-1)
        fig.add_trace(go.Scatter(y=data[:,column],mode='lines+markers',name=name),
                      row=index//2+1,col=index%2+1)
    return fig.update_layout(height=850,title='Synchronized inputs, gates and states',showlegend=False)


def gate_diagram():
    return '''digraph LSTM { rankdir=LR;
    previous_c -> retain; forget -> retain; retain -> add;
    input_gate -> write; candidate -> write; write -> add;
    add [label="c_t = retained + written"]; add -> tanh; tanh -> expose;
    output_gate -> expose; expose [label="h_t = o_t * tanh(c_t)"];
    x_t -> gates; previous_h -> gates; gates -> forget; gates -> input_gate;
    gates -> candidate; gates -> output_gate; }'''
