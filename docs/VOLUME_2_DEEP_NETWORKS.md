# Volume 2 — Deep Neural Networks, Autograd and Experimentation

**PyTorch Neural Network Studio v4 · Milestone 4**

## 1. Learning outcomes

Build a deep multilayer perceptron, compute tensor shapes and parameter counts, explain the chain rule across layers, visualize gradient magnitude, and design controlled comparisons across activation, width and depth. This volume follows Volume 1.

## 2. Deep network architecture

```text
Inputs [B,3]       Hidden-1 [B,5]        Hidden-2 [B,4]      Output [B,1]
     X ------> Linear(3,5) -> ReLU -> Linear(5,4) -> ReLU -> Linear(4,1)
                   W1[5,3]               W2[4,5]               W3[1,4]
                   b1[5]                 b2[4]                 b3[1]
```

The computation is:

```
z1 = X @ W1.T + b1            # [B,5]
h1 = activation(z1)           # [B,5]
z2 = h1 @ W2.T + b2           # [B,4]
h2 = activation(z2)           # [B,4]
y_hat = h2 @ W3.T + b3        # [B,1]
```

Parameter count for 3 → 5 → 4 → 1 with biases:

- Layer 1: `3×5+5 = 20`.
- Layer 2: `5×4+4 = 24`.
- Output: `4×1+1 = 5`.
- **Total: 49 trainable parameters.**

## 3. Backpropagation and chain rule

Let `L` be a scalar loss. An early weight influences the loss through all later layers. With column-vector notation, a three-layer network follows `h1=f(W1x+b1)`, `h2=f(W2h1+b2)`, and `y_hat=W3h2+b3`.

The gradient to `W1` includes derivatives of loss against prediction, prediction against `h2`, `h2` against `h1`, and `h1` against `W1`:

`∂L/∂W1 = (∂L/∂y_hat) × (∂y_hat/∂h2) × (∂h2/∂h1) × (∂h1/∂W1)`

These are Jacobian products (or compatible vector-Jacobian products), **not** ordinary scalar multiplication when tensors have multiple dimensions. PyTorch autograd efficiently computes the correct gradients through the computational graph.

### Gradient flow

```text
FORWARD:      X -> Layer 1 -> activation -> Layer 2 -> activation -> Output -> Loss
BACKWARD:     dW1 <- dh1 <-  dz2      <- dh2  <-         dy   <- dLoss
UPDATE:       W1,b1     W2,b2       W3,b3  := old - η × gradient
```

If many derivatives have magnitude substantially below 1, gradients through deep chains can diminish. Saturated Sigmoid/Tanh activations are one common mechanism. ReLU reduces saturation on its positive branch, but can have zero gradients on negative inputs. Modern architectures also employ appropriate initialization, normalization and skip/residual connections.

## 4. Guided lab: change network depth

1. Open **Learning Studio** → **Deep Neural Network**.
2. Set input **3**, hidden layers **2**, hidden width **5**, output **1**, ReLU, seed **42**.
3. Predict parameter count for the resulting **3 → 5 → 5 → 1** architecture: `(3×5+5)+(5×5+5)+(5×1+1)=56`.
4. In **Architecture & tensors**, verify parameter shapes. Compare each weight tensor's dimensions with adjacent layers.
5. In **Forward propagation**, type `1,-1,0.5`. Confirm the flow from `(1,3)` to `(1,5)` to `(1,5)` and finally `(1,1)`.
6. In **Backward & SGD**, type target `0.5` and choose learning rate 0.1. Inspect gradient norms and one early-layer weight's before/gradient/after values.
7. Increase hidden layers to **4**. The parameter count and computation path increase. Observe gradient norms; do not assume they must decrease monotonically with depth.

## 5. Guided lab: compare activations fairly

Keep input, target, hidden width, depth, bias, seed, and learning rate unchanged. Test **Tanh**, **Sigmoid**, **ReLU**, then **GELU**. For each, record:

| Activation | Loss before | Loss after | First layer grad norm | Observation |
|---|---|---|---|---|
| Tanh | fill in | fill in | fill in | |
| Sigmoid | fill in | fill in | fill in | |
| ReLU | fill in | fill in | fill in | |
| GELU | fill in | fill in | fill in | |

Changing activation also changes intermediate values, so comparisons isolate the effect of a configuration choice *with the same seeded linear weight initialization*. It does not imply equal effective initialization statistics for every activation.

## 6. Guided lab: width and parameter count

For a fixed network, increase hidden width 3 → 8 → 16. Write down the parameter count and consider compute/memory tradeoffs. Dense layer matrix multiplications scale roughly with `batch_size × input_width × output_width`; a deeper or wider architecture is not automatically better. More parameters increase capacity but can also increase overfitting risk.

## 7. Multi-epoch learning with Training Studio

The Learning Studio is a **single-step microscope**. To study optimization trajectories:

1. Use Dataset Management to prepare **Iris** or **Two moons**.
2. Configure the network in Training Studio and record baseline epoch-0 loss/score.
3. Run 10–25 epochs. Inspect both train and validation curves.
4. Change learning rate or architecture; understand that Training Studio resets the experiment on configuration changes.
5. Compare train/validation and perform test evaluation only after selecting model settings.

Typical diagnostic patterns:

- Train loss decreases, validation loss rises: possible overfitting.
- Both losses remain high: consider learning rate, underfitting, preprocessing or data quality.
- Loss oscillates: try smaller learning rate or adjusted optimizer.
- Tiny early gradients: inspect saturation, initialization and network depth.

These heuristics are not definitive diagnoses; hold other variables fixed and run controlled experiments.

## 8. Framework-neutral design for future TensorFlow/Keras

The definition of a network is a serializable `ModelSpec` with `architecture`, `input_features`, ordered `LayerSpec` records, bias, seed, and `framework`. This is pure Python and does not import PyTorch. **PyTorchAdapter** owns model instantiation, forward trace, gradient demonstration, and parameter count. `learning/ui.py` owns Streamlit rendering. Visualizations accept ordinary arrays/data structures rather than PyTorch-only tensors.

```text
                       Streamlit Learning UI
                                |
                          ModelSpec (DTO)
                                |
                         Adapter Registry
                       /                  \
                 PyTorchAdapter      TensorFlowAdapter
                   IMPLEMENTED       FUTURE / NOT INSTALLED
                       |
                  PyTorch autograd
                       |
                 JSON / Plotly views
```

A future TensorFlow adapter must implement matching semantics, data types, seed rules, and value/gradient result schemas. **Do not** assume numerical identity between frameworks under the same seed; their random initializers, gradient details, and computational kernels differ. Add backend-specific parity tests with aligned imported weights when TensorFlow is introduced.

## 9. Knowledge checks and practice

1. Calculate the total parameters in a 4 → 6 → 4 → 2 network with bias: `(24+6)+(24+4)+(8+2)=68`.
2. Why does `[B,4] @ W.T` work when `W` is stored as `[6,4]`?
3. What does the chain rule imply about the influence of later layers on an early weight gradient?
4. Why might ReLU yield a zero gradient for some units?
5. What information does validation loss provide that training loss does not?

Answer the built-in quiz and save exercise notes. Export progress JSON if you want to resume in another browser session.

## 10. Looking ahead

The modular tensor visualization and adapter contract can support future lessons: CNN → RNN → LSTM → embeddings → single/multi-head attention → Transformer encoder → Transformer decoder → Mini GPT. Those architectures **are not implemented in Milestone 4**. TensorFlow/Keras remains future work.
