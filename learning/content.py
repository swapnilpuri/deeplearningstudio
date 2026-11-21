"""Educational metadata is independent from the frontend and runtime backend."""
LESSONS = {
 'feed_forward': {
    'title': 'Feed-Forward Neural Network',
    'goal': 'Understand a single hidden layer and the path from inputs to predictions.',
    'concepts': [
        ('Neuron', 'A neuron computes a weighted sum plus a bias: z = Σ(xᵢwᵢ) + b.'),
        ('Matrix multiplication', 'For a batch X of shape [B, D], W has [H, D]; Z = X Wᵀ + b has [B, H].'),
        ('Nonlinearity', 'Apply a = f(z). Without nonlinear activations, stacked linear layers collapse to one linear map.'),
        ('Loss', 'Mean squared error MSE = mean((prediction − target)²) measures numerical prediction error.'),
        ('Backpropagation', 'Autograd uses the chain rule to obtain ∂Loss/∂W and ∂Loss/∂b.'),
        ('Update', 'SGD moves every parameter: θ_new = θ_old − learning_rate × gradient.'),
    ],
    'questions': [
        ('A dense layer accepts 3 inputs and has 4 neurons with bias. How many trainable parameters?', ['12', '16', '7'], 1, '3 × 4 weights plus 4 biases = 16.'),
        ('Which operation provides nonlinearity?', ['Matrix multiply', 'ReLU', 'Bias addition'], 1, 'ReLU clips negative values and is nonlinear.'),
        ('What does optimizer.step() do after backward()?', ['Calculates predictions', 'Updates parameters', 'Loads data'], 1, 'SGD uses gradients to change parameters.'),
    ],
 },
 'deep_neural_network': {
    'title': 'Deep Neural Network',
    'goal': 'Trace multiple hidden layers and see how gradients flow through depth.',
    'concepts': [
        ('Depth', 'A deep network composes hidden transformations: h₁ = f(XW₁ᵀ+b₁); h₂ = f(h₁W₂ᵀ+b₂).'),
        ('Shapes', 'Every linear weight tensor has shape [out_features, in_features].'),
        ('Chain rule', 'The gradient of a parameter in an early layer depends on all later layers.'),
        ('Vanishing gradients', 'Repeated multiplication by small derivatives can shrink upstream gradients.'),
        ('Activation selection', 'Sigmoid/Tanh saturate; ReLU/GELU often help optimization but have other tradeoffs.'),
        ('Controlled experiment', 'Hold seed, input and target constant while changing depth to compare gradients.'),
    ],
    'questions': [
        ('Which tool computes gradients in this studio?', ['NumPy only', 'PyTorch autograd', 'Plotly'], 1, 'loss.backward() invokes the PyTorch autograd graph.'),
        ('Why might an early layer have tiny gradients?', ['Vanishing gradients', 'More examples', 'More output neurons'], 0, 'Derivatives compound through layers via the chain rule.'),
        ('What shape is Linear(5, 3).weight?', ['(5, 3)', '(3, 5)', '(5,)'], 1, 'PyTorch stores weights as [output, input].'),
    ],
 },
}
EXERCISES = {
 'feed_forward': ['Try 2 → 3 → 1 with bias: predict parameter count before checking it.',
                  'Change ReLU to Tanh with seed fixed and inspect outputs.',
                  'Set the learning rate to 0.01, then 0.5; compare parameter updates and MSE.'],
 'deep_neural_network': ['Build 3 → 5 → 4 → 1; calculate weight and bias shapes.',
                         'Change hidden depth from two to four while fixing seed and observe gradient norms.',
                         'Compare Sigmoid and ReLU for the same input; explain where gradients become small.'],
}
