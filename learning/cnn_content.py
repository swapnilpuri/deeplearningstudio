"""Beginner CNN explanations shared by the lesson and documentation."""

TOPICS = [
    ('Images as tensors', 'A pixel is a number. An 8×8 digit is 64 intensities; a batch of 32 grayscale digits has shape [32,1,8,8]. Digits intensities range from 0 to 16; x = pixel/16 maps them to [0,1].'),
    ('Grayscale and RGB', 'Grayscale has one intensity channel; RGB has separate red, green and blue planes. One 6×8 RGB image has shape [1,3,6,8], containing 144 numbers. Changing a red pixel changes only its red-channel entry.'),
    ('Input and output channels', 'Each filter sees every input channel and produces one output channel. Weights have shape [C_out,C_in,K_h,K_w]. Four filters on RGB with 3×3 kernels use [4,3,3,3] weights; [1,3,8,8] becomes [1,4,6,6] without padding.'),
    ('Filters and kernels', 'A kernel is a small grid of coefficients. For one input channel, [[1,0],[0,-1]] compares opposite corners. With several input channels, a filter contains one kernel per channel, then sums their responses. A bias adds one learned offset per output filter.'),
    ('Sliding cross-correlation', 'At output (i,j), multiply aligned kernel and window entries, sum across space and channels, then add bias. PyTorch does not flip the kernel. On [[1,2],[5,6]], [[1,0],[0,-1]] gives 1−6 = −5. A 4×4 input with a 2×2 kernel yields a 3×3 map at stride 1.'),
    ('Stride', 'Stride S is the distance between window origins. H_out = floor((H+2P−D(K−1)−1)/S+1). With H=8,K=3,P=0,D=1,S=2, H_out=3: origins are 0,2,4. Trailing pixels need not start a full window.'),
    ('Padding', 'Zero padding P adds a border without adding learned parameters. For H=8,K=3,S=1,D=1,P=1, H_out=8. The padded input is 10×10 while the original remains 8×8; border windows include zeros.'),
    ('Dilation', 'Dilation D spaces sampled kernel entries apart. Effective kernel = D(K−1)+1. A 3×3 kernel at dilation 2 spans 5×5 but still uses nine weights per channel. An 8×8 unpadded input gives 4×4 output at stride 1. Gaps are not multiplied.'),
    ('Feature maps', 'The spatial response of one filter is one feature map. Four filters yield [N,4,H_out,W_out]. A strong positive edge response means the local pattern aligns with the signed filter; negative responses indicate the opposite orientation, not a probability.'),
    ('Multiple filters', 'Different output filters learn different features while sharing their own weights across positions. Two 3×3 filters on one-channel 8×8 input with padding 1 yield [N,2,8,8]. Manually copying the same preset into both filters gives identical maps until the weights or biases differ.'),
    ('ReLU', 'ReLU(z)=max(0,z) acts element by element and preserves shape. Values [−5,0,2] become [0,0,2]. Its derivative is 1 for positive inputs and 0 for negative inputs; PyTorch chooses derivative 0 at zero.'),
    ('Max pooling', 'MaxPool2d keeps the maximum in each channel window. [[1,3],[2,4]] becomes 4. A [N,8,8,8] tensor with 2×2 windows and stride 2 becomes [N,8,4,4]. There are no learned weights; gradients route to selected maxima.'),
    ('Average pooling', 'AvgPool2d takes the mean: [[1,3],[2,4]] becomes 2.5. With zero padding, dilation 1 and ceil_mode=False, H_out=floor((H−K)/S+1), and likewise for width. Pooling reduces downstream cost and gives limited local translation tolerance; it does not guarantee shift invariance.'),
    ('Flatten', 'Flatten preserves every value and the batch axis: [N,C,H,W] → [N,C×H×W]. Thus [32,16,2,2] becomes [32,64]. It has no parameters. The builder infers this width from all preceding layers.'),
    ('Fully connected classification', 'Linear computes z = x W^T + b. From 64 flattened features to 10 classes, W has shape [10,64], b has shape [10], output is [N,10], and parameter count is 64×10+10=650.'),
    ('CNN forward propagation', 'Data flows Conv2d → ReLU → pooling → Conv2d → ReLU → pooling → Flatten → Linear. For Digits: [N,1,8,8] → [N,8,8,8] → [N,8,4,4] → [N,16,4,4] → [N,16,2,2] → [N,64] → [N,10]. Inspect actual intermediate tensors in the builder.'),
    ('Classification logits', 'Logits are unbounded class scores, not probabilities. Prediction = argmax(logits). For [2,1,0], class 0 wins; softmax(z)_j=exp(z_j)/sum_k exp(z_k) gives approximately [0.665,0.245,0.090]. Ten Digits classes require [N,10] logits.'),
    ('Cross-entropy loss', 'For target class t, L = −z_t + log(sum_j exp(z_j)). For logits [2,1,0] and t=0, L≈0.408. PyTorch CrossEntropyLoss accepts [N,C] raw logits and [N] integer class indices; do not apply softmax first. The batch loss defaults to a mean.'),
    ('Backpropagation through convolution', 'The chain rule sends dL/dz backward through the head, Flatten, pooling and ReLU. A shared filter weight receives the sum of contributions from every window: dL/dW[o,c,u,v] = sum_(n,i,j) dL/dY[n,o,i,j] × X_padded[n,c,iS+uD,jS+vD]. Its gradient has the same OIHW shape as the weights. Bias gradients sum dL/dY over batch and spatial axes.'),
    ('Gradient descent and updates', 'Call optimizer.zero_grad(), forward, loss.backward(), then optimizer.step(). Plain SGD uses W_new=W_old−η×gradient; weight 0.5 with gradient 0.2 and η=0.1 becomes 0.48. Adam keeps first and second gradient moment estimates and rescales updates. Validation and test examples never enter this update path.'),
]

EXPERIMENTS = [
    ('Switch vertical to horizontal edge filters.', 'Strong responses move from vertical boundaries to horizontal boundaries; signs depend on edge direction.'),
    ('Increase stride with fixed image, kernel and padding.', 'Output spatial size stays the same or decreases according to the floor formula; parameter count stays fixed.'),
    ('Increase output filter count.', 'Output channels and convolution parameter count grow linearly; spatial dimensions stay fixed.'),
    ('Compare max and average pooling on the same window.', 'Max preserves the strongest response; average smooths it. Neither has trainable parameters.'),
    ('Increase dilation from 1 to 2.', 'The sampled receptive field grows while weight count stays fixed; without more padding, output size falls.'),
    ('Train five epochs, then compare initial and trained filters.', 'Weights change where loss gradients are nonzero. Validation can fluctuate; training improvement is not guaranteed each epoch.'),
]
