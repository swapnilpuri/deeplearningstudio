# Documentation Volume 3 — Convolutional Neural Networks

**DeepLearningStudio M5 — educational reference and practical user guide**

## 1. Why CNNs?

A dense model first flattens an image. An 8×8 grayscale image becomes 64 features, and a `Linear(64, 128)` layer learns 8,320 parameters including bias. A CNN instead scans a small shared filter across the image. `Conv2d(1, 8, 3, padding=1)` learns just `8 × (1 × 3 × 3 + 1) = 80` parameters and preserves spatial organization. Parameter sharing, local receptive fields and hierarchical feature extraction are the defining differences.

## 2. Images as tensors

PyTorch stores image batches in **NCHW** order:

- N: batch size, number of images processed together
- C: image channels, 1 for grayscale and 3 for RGB
- H: image height in pixels
- W: image width in pixels

A batch of 32 RGB 28×28 images has shape `[32, 3, 28, 28]`. The scikit-learn Digits dataset consists of 8×8 grayscale images. Its original intensities are in the known range 0 to 16; dividing by 16 maps values into `[0,1]` without estimating a statistic on validation/test data.

## 3. Convolution versus cross-correlation

A `Conv2d` filter has one kernel for each input channel. For each output position it multiplies corresponding input window and kernel entries, sums them over spatial positions **and** input channels, and optionally adds a bias. PyTorch `Conv2d` implements **cross-correlation**, which does **not** flip the kernel; the common CNN term “convolution” follows machine-learning convention.

For output channel `o` at image position `(i,j)`:

```text
Y[n,o,i,j] = bias[o] + sum over c,u,v of
              W[o,c,u,v] * X_padded[n,c,i*stride_h+u*dilation_h,
                                      j*stride_w+v*dilation_w]
```

The bias term is omitted when disabled. PyTorch weight shape is `[output_channels, input_channels, kernel_height, kernel_width]` for standard ungrouped convolutions. This studio deliberately does not introduce grouped or depthwise convolution in M5.

### Hand calculation: 4×4 input with a 2×2 filter

```text
Image                  Kernel
1   2   3   4          1   0
5   6   7   8          0  -1
9  10  11  12
13 14  15  16
```

At the first output position, the top-left window is `[[1,2],[5,6]]`.

```text
1*1 + 2*0 + 5*0 + 6*(-1) = -5
```

With stride 1 and padding 0, output is 3×3, with each value -5. Try this using the **Convolution explorer** with a custom 2×2 kernel `[1,0,0,-1]`, input size 4, a single channel, padding 0, stride 1, and a zero bias. Note that the explorer's default image is scaled by its width, so multiply the displayed values by four to compare with the exact integer example; the automated test uses the literal integer input.

### Multi-channel calculation

For RGB (`Cin=3`), a single output filter contains three kernels. At each location compute `sum(input_red * kernel_red)`, `sum(input_green * kernel_green)`, and `sum(input_blue * kernel_blue)`, then add their sums and the one output-channel bias. Multiple output filters perform separate calculations and produce distinct output feature maps.

## 4. Stride, padding, dilation and output sizes

For each spatial dimension separately:

```text
output = floor((input + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
```

For example, height 8, kernel 3, stride 1, padding 1, dilation 1: `floor((8+2-2-1-1)/1+1) = 8`. For width 12, kernel width 5, stride 2, padding 0: `floor((12-4-1)/2+1) = 4`.

Changing stride skips some possible window positions and shrinks output size. Padding preserves border context or desired dimensions. Dilation spreads sampled kernel cells farther apart without increasing learned parameter count.

## 5. Feature maps, ReLU, and filter examples

A feature map is the response of one learned filter across all spatial positions. Each filter can emphasize certain patterns. Manually constructed demonstrations are **not** the same as weights obtained from learning:

```text
Identity:         Vertical edge:     Horizontal edge:
0 0 0              -1  0  1           -1 -1 -1
0 1 0              -1  0  1            0  0  0
0 0 0              -1  0  1            1  1  1

Blur:             Sharpen:
1/9 1/9 1/9        0 -1  0
1/9 1/9 1/9       -1  5 -1
1/9 1/9 1/9        0 -1  0
```

The activation `ReLU(z) = max(0,z)` clips negative feature-map values, providing nonlinearity. A model can learn an orientation-selective or more complex filter. Some learned kernels are difficult to interpret visually because they operate on intermediate feature channels, not raw images.

## 6. Pooling

Max pooling selects the largest value in a window. Average pooling computes its arithmetic average.

```text
Window:  1  4
         6  2

MaxPool = 6        AvgPool = (1+4+6+2)/4 = 3.25
```

For zero padding, dilation 1, and `ceil_mode=False`:

```text
Hout = floor((Hin - Kh)/Sh + 1)
Wout = floor((Win - Kw)/Sw + 1)
```

Pooling usually reduces spatial size, lowering work for later layers. Local translation tolerance is limited: a moved feature may retain a similar maximum within the same pooling window, but pooling is **not** general translation invariance. Pooling has no trainable weights.

## 7. CNN architecture walkthrough

The Digits training model follows this pipeline:

```text
Input                [N, 1, 8, 8]
  ↓ Conv2d(1,8,3,pad=1) [N, 8, 8, 8]       80 params
  ↓ ReLU              [N, 8, 8, 8]
  ↓ MaxPool2d(2,2)    [N, 8, 4, 4]
  ↓ Conv2d(8,16,3,pad=1) [N, 16, 4, 4]  1168 params
  ↓ ReLU              [N, 16, 4, 4]
  ↓ MaxPool2d(2,2)    [N, 16, 2, 2]
  ↓ Flatten           [N, 64]
  ↓ Linear(64,10)     [N, 10]             650 params
                                          ----------
                                          1898 params
```

The fully connected layer's 64 input features are *inferred*, not a fixed value. Changing input dimensions, kernel, padding, strides or channel counts changes the propagated shapes. An invalid intermediate shape is rejected before constructing the PyTorch model.

In PyTorch, linear `nn.Linear(in_features, out_features)` weight shape is `[out_features, in_features]`. A CNN model's final ten values are unnormalized **class logits**. The predicted class is `argmax(logits)`.

## 8. Training process and loss

Digits uses a deterministic, stratified partition into approximately 70% training, 15% validation, 15% test. A fixed pixel divisor of 16 provides image normalization without leakage. No validation or test example appears in training mini-batches. The test set stays held out until the learner selects **Evaluate held-out test set**.

For an example whose true class is `t`, cross-entropy is:

```text
L = -log(exp(logit[t]) / sum_j(exp(logit[j])))
```

`torch.nn.CrossEntropyLoss` accepts raw logits and class indices (not one-hot targets). For each mini-batch:

```python
optimizer.zero_grad(set_to_none=True)
logits = model(images)
loss = torch.nn.functional.cross_entropy(logits, targets)
loss.backward()
optimizer.step()
```

The history shows training and validation loss/accuracy and average convolutional gradient norms. Each epoch uses the training split only for updates. The test confusion matrix displays actual classes in rows and predicted classes in columns. Incorrect samples reveal patterns the classifier has not learned well.

## 9. Backpropagation through convolution

Every filter weight participates in several window computations; autograd accumulates the derivative contribution from all uses. The chain rule propagates gradients from cross-entropy through the linear head, Flatten, pooling, ReLU and convolutional weights and inputs.

With plain SGD:

```text
grad = ∂Loss/∂W
W_new = W_old - learning_rate * grad
```

Biases follow the same rule. The Backpropagation tab captures weights and gradients before `optimizer.step()` and then shows the updated values. Compare numbers to verify the equation. Adam instead maintains adaptive estimates of first and second gradient moments; its update is not exactly the displayed SGD equation.

### Numerical verification

For a small double-precision convolution, perturb one input or weight by `±epsilon` and approximate its derivative by:

```text
(f(x + epsilon) - f(x - epsilon)) / (2*epsilon)
```

The automated CNN test uses PyTorch `gradcheck` to compare numerical derivatives against autograd. This is a teaching validation on a tiny deterministic expression, not a proof that every arbitrary CNN model is numerically stable.

## 10. Step-by-step platform tutorials

**Convolution explorer:** Open Learning Studio → CNN → Convolution explorer. Start with one channel, a 3×3 vertical-edge filter, stride 1, padding 1. Move output row/column. The “Selected padded window” and “Window × kernel” tables show the exact dot product. Switch to RGB channels and observe per-channel contributions. Increase filters to see separate maps. Toggle ReLU to see which negative values become zero.

**Pooling explorer:** Switch between Max and Average with the same feature map. Move the selected window and predict its pooled value before reading the result. Increase stride to show fewer output positions. Compare the two summaries on the same 2×2 region.

**Architecture builder:** Configure channels, input size, two convolution widths, kernel size, padding, pooling type and number of class outputs. Inspect each computed shape and parameter count. Try a configuration that collapses spatial dimensions to see validation reject it. Download the JSON representation; it uses CNN schema v2 and does not modify old M4 ModelSpec JSON.

**Backpropagation:** Start with SGD learning rate 0.01. Pick a convolutional weight parameter, read its gradient L2 norm and inspect the old, gradient and new values. Verify at least one value manually. Compare a smaller learning rate to understand step size.

**Training laboratory:** Run five epochs on Digits using Adam. Review separate loss and accuracy curves for training and validation. Compare the initial and learned filter from the first convolution block. When ready, select **Evaluate held-out test set** and inspect the confusion matrix, examples and mistakes. Retest with SGD to see how optimization differs.

**Quiz and progress:** Answer the six CNN questions, save observations, and export the progress file. A previously exported M4 JSON file remains importable. Completing CNN adds `cnn` to the existing completed-lessons list without deleting older completions or quiz scores.

## 11. Troubleshooting and limitations

| Symptom | What to check |
|---|---|
| Invalid kernel shape | Kernel height/width must fit padded input. Reduce kernel or increase padding/image size. |
| Negative computed output | This is normal before ReLU. ReLU maps negative values to zero. |
| Architecture does not build | Inspect shape propagation: two pooling blocks may make the feature map too small. |
| Training takes a while | Use fewer filters/epochs; CPU is sufficient for Digits. |
| Test accuracy lower than validation | A small dataset and random variation can make estimates differ. Do not tune repeatedly against test results. |
| Flat loss curve | Compare learning rate, optimizer, gradient norms and filter sizes. |
| Imported progress appears empty | Export JSON from original v4 interface and ensure the file has `schema_version: 1`. |
| ONNX or Netron | M5 does not extend the existing v3 ONNX export path to CNN; existing v3 export should remain separate. |

## 12. Comparison and further reading

| Feature | Dense networks | CNNs |
|---|---| 
| Spatial structure | Typically flattened input | Preserves local neighborhoods |
| Parameter sharing | Different weight per input-output pair | Same filter reused across spatial positions |
| Typical input | Structured vectors, tabular features | Images and spatial tensors |
| Invariance | Not inherent | Limited shift tolerance, especially with pooling |
| Output head | Dense layers | Often convolution blocks followed by dense logits |

Future roadmap: RNN, LSTM, embedding network, single-head and multi-head attention, Transformer encoder/decoder, and Mini GPT. The framework-neutral specification can be interpreted by a future TensorFlow/Keras adapter; that adapter is **not included** in M5.
