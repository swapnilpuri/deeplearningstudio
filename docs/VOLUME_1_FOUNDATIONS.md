# Volume 1 — Feed-Forward Neural Networks: Interactive Foundations

**PyTorch Neural Network Studio v4 · Milestone 4**

## 1. Learning outcomes

After this volume, you can (1) explain tensors, weights, biases and activations; (2) construct a feed-forward model; (3) trace a complete forward pass; (4) derive MSE and gradients; (5) verify one SGD update; and (6) interpret a training loss curve. Use the **Learning Studio** workspace for hands-on activities and **Training Studio** for multi-epoch experiments.

## 2. Quick start

Follow the Windows commands in the root README. Open the app and choose **Learning Studio** from the left sidebar. Choose **Feed-Forward Neural Network**. Every change to model settings creates a *new*, deterministically initialized PyTorch model; the single optimizer step displayed is an independent demonstration, not persistent training.

## 3. Mental model

```text
     x1 ----\
             [ Hidden neuron h1 ] ----\
     x2 ----/                       [ Output y_hat ] -> MSE target -> gradients
          \  [ Hidden neuron h2 ] ----/                    |
           ->[ Hidden neuron h3 ]--/                  SGD changes W, b
```

A fully connected layer with `D` input features and `H` output neurons has `D × H` weights and, when enabled, `H` biases. **Parameter count = D × H + H.** A 2 → 3 → 1 model with biases has `(2×3+3) + (3×1+1) = 13` parameters.

A batch `X` of `B` examples with `D` features has shape `[B,D]`. The weight tensor is stored as `[H,D]`. PyTorch computes `Z = X @ W.T + b`, resulting in `[B,H]`. The activation `A=f(Z)` has the same shape. A new layer repeats this process.

## 4. Mathematical walk-through

Consider a single neuron with `x=[2,3]`, `w=[0.5,-0.25]`, `b=0.1`.

- Weighted sum: `z = 2×0.5 + 3×(-0.25) + 0.1 = 0.35`.
- ReLU: `max(0, 0.35) = 0.35`.
- Sigmoid: `1 / (1+exp(-0.35)) ≈ 0.5866`.
- Tanh: `tanh(0.35) ≈ 0.3364`.

These numbers are a **manual illustration**, not the platform's seeded weights. To see the platform's **actual** values, inspect **Architecture & tensors** and **Forward propagation**.

### Activations

| Function | Formula | Key property |
|---|---|---|
| ReLU | `max(0,z)` | Zero for negative `z`; positive branch has derivative 1 |
| Sigmoid | `1/(1+exp(-z))` | Output in (0,1), saturates at extremes |
| Tanh | `(exp(z)-exp(-z))/(exp(z)+exp(-z))` | Output in (-1,1), saturates |
| GELU | `z × Φ(z)` | Smooth, input-dependent gating |

`Φ(z)` is the standard normal cumulative distribution function. In a deep stack, nonlinear activations matter: without them, a composition of affine layers can be reduced to one affine layer.

### Loss, gradients and update

For `N` scalar predictions: `MSE = (1/N) × Σ(prediction_i - target_i)^2`. A parameter gradient tells us the local rate of change of loss with respect to that parameter. Backpropagation uses the chain rule to compute all parameter gradients efficiently. For simple SGD: `parameter_new = parameter_old - learning_rate × gradient`.

PyTorch operations:

```python
optimizer.zero_grad()           # clear gradients from previous steps
pred = model(x)                 # differentiable forward pass
loss = torch.nn.MSELoss()(pred, target)
loss.backward()                 # autograd chain rule
optimizer.step()                # update each parameter
```

## 5. Guided platform lab: forward pass

1. Choose Learning Studio → **Feed-Forward Neural Network**.
2. Set input features **2**, width **3**, output neurons **1**, **Tanh**, bias **enabled**, seed **42**.
3. Predict **13 parameters** before checking the displayed parameter count.
4. Open **Architecture & tensors**. Inspect the first layer's weight shape `(3,2)` and bias shape `(3,)`, and the output layer's `(1,3)` and `(1,)`.
5. Open **Forward propagation**. Enter `1,-1`. Examine the input tensor shape `(1,2)`, first Linear output `(1,3)`, Tanh output `(1,3)`, and final Linear output `(1,1)`.
6. Explain why `[1,3] @ [3,2]` is **not** the correct multiplication without transposing the weight orientation. The correct calculation is `[1,2] @ [2,3]`.
7. Change the activation to ReLU. Hold the seed fixed, compare the intermediate values and identify differences caused by activation rather than initial weight changes.

## 6. Guided platform lab: one update

1. Retain 2 → 3 → 1 and enter input `1,-1` and target `0`.
2. Open **Backward & SGD**. Choose SGD learning rate **0.1**.
3. Record the **before** and **after** MSE. (The loss often decreases for small steps, but not for every dataset/learning rate.)
4. Inspect `0.weight` and one row in its update table. Verify `after ≈ before - 0.1 × gradient` (float32 rounding applies).
5. Change learning rate to **0.01** and **0.5**. Compare maximum updates. A larger step is not always better.
6. Review the gradient bar chart. A low gradient norm indicates little local change signal, not necessarily correctness.

**Important distinction:** The Learning Studio demonstrates one recreated optimizer step; **Training Studio** runs multiple epochs and preserves a training experiment state within the session.

## 7. Guided platform lab: supervised learning

1. In **Dataset Management**, select built-in **Two moons** with seed 42 and apply to Training Studio.
2. In **Training Studio**, use two hidden layers of 16 neurons, Tanh activation and Adam optimizer; train in batches of epochs.
3. Inspect validation loss and training loss. Watch for a widening generalization gap; do not infer generalization from training accuracy alone.
4. Evaluate the held-out test split *after* choosing configuration based on validation.
5. Optionally export and inspect the ONNX model in Netron. ONNX export needs the optional runtime dependencies and has not been verified in all environments.

## 8. Knowledge checks

- Why are nonlinearities needed between layers?
- What shape is `Linear(2,3).weight`? Why?
- What do `zero_grad`, `backward`, and `step` do?
- How many parameters does a `4 → 6 → 2` network have, with bias? **Answer:** `(4×6+6)+(6×2+2)=44`.

Complete the built-in quiz and **Export progress JSON** to retain a portable record. By default this progress is stored only in the current Streamlit browser session; import your JSON later to restore it.

## 9. Troubleshooting

- Input count warning: enter exactly the configured number of finite comma-separated values.
- MSE target count warning: enter one target per output neuron.
- Gradient results appear different: ensure all network settings, the target, learning rate and seed match.
- Charts are clipped: use small widths in foundational lessons; large architecture graphs are summarized for readability.

## 10. Scope

No persistent user accounts, automatic grading of free-text exercises, or TensorFlow execution is provided. This is a CPU-friendly educational model; not a production model selection pipeline.
