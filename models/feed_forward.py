import torch
from torch import nn

ACTIVATIONS = {name: getattr(nn, name) for name in ('ReLU', 'Sigmoid', 'Tanh', 'GELU')}

def build_model(sizes, activation='ReLU', bias=True, seed=42):
    if len(sizes) < 2 or any(n < 1 for n in sizes):
        raise ValueError('At least two positive layer sizes are required')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        layers = []
        for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(a, b, bias=bias))
            if i < len(sizes)-2:
                layers.append(ACTIVATIONS[activation]())
        return nn.Sequential(*layers).eval()

def generate_code(sizes, activation, bias, seed):
    lines = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        lines.append(f'    nn.Linear({a}, {b}, bias={bias}),')
        if i < len(sizes)-2:
            lines.append(f'    nn.{activation}(),')
    return '\n'.join(['import torch', 'from torch import nn', '', f'torch.manual_seed({seed})', 'model = nn.Sequential(', *lines, ').eval()', '', f'x = torch.zeros(1, {sizes[0]})', 'with torch.no_grad():', '    print(model(x))', '', 'torch.onnx.export(model, (x,), "network.onnx",', '    input_names=["input"], output_names=["output"],', '    dynamo=True, external_data=False)', '', 'import netron', 'netron.start("network.onnx", browse=True)'])
