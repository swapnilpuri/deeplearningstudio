# Volume 8 — Multi-head attention and Transformer blocks

Run `.\.venv\Scripts\python.exe -m streamlit run app.py`. Choose Learning Studio
→ Multi-Head Attention & Transformer Blocks. Twelve sections lead from M9's
single routing pattern to one trainable Transformer block. Start at D=8, H=2,
F=16 and seed 42. These are small demonstrations, not language models.

## Why several heads?

One head gives each query one distribution over keys. Multiple heads can learn
different matching and value subspaces, so several mixtures can contribute at
once. Separate parameters permit complementary behavior; random initialization
does not establish linguistic specialization. More heads do not guarantee better
performance. Inspect actual trained behavior on the chosen task.

Each row is a token. For head h, Qh=X@Wqh, Kh=X@Wkh, Vh=X@Wvh. The inspector
shows the selected head's own columns from each packed weight. Standard mode
uses dk=dv=D/H and requires D divisible by H. D=8,H=2 gives four coordinates per
head. D=7,H=2 is invalid and receives an immediate warning.

| Tensor | Example B=1,T=6,D=8,H=2 |
|---|---|
| Input X | [1,6,8] |
| Each packed projection weight | [8,8] |
| Each head's projection weight | [8,4] |
| Q/K/V split by head | [1,2,6,4] |
| Scores and probabilities | [1,2,6,6] |
| Per-head outputs | [1,2,6,4] |
| Concatenated heads | [1,6,8] |
| Wo | [8,8] |
| Final MHA output | [1,6,8] |

Synthetic inputs and M8 lookup tables are untrained. Session-trained M8 skip-gram
weights can be selected after training M8; set D to that table's width and choose
a dividing head count. Unknown words map to UNK. Custom JSON must match [T,D].
Repeated words use the same lookup vector but retain separate position labels.

## Calculate attention and join heads

Ah=softmax(Qh@Kh.T/sqrt(dk)+Mask), headh=Ah@Vh. Softmax acts across keys, so each
query row sums to one over visible keys. For q=[1,2], k=[3,4], the dot product
is 11 and scaled score is 11/sqrt(2). Compare other scores in the row before
softmax; a single score cannot determine a probability by itself.

Bidirectional attention sees all non-padding keys. Causal mode permits j<=i;
future probabilities are exactly zero. Padding marks trailing keys; padded query
rows still exist. Fully masked rows are rejected. All head heatmaps preserve
token order and use a common 0–1 scale. Compare selected-query distributions
across heads and masked/unmasked selected-head heatmaps.

Concatenation joins head feature columns without averaging or dropping heads.
For two one-token head rows [1,2] and [3,4], concatenation gives [1,2,3,4].
Multiplication by Wo mixes those features and returns D columns. The tensor
tables reveal each head, concatenation, Wo and projected output.

The single/multi-head comparison holds input and seeded packed weights fixed.
Only splitting and sqrt(dk) scaling differ. M9 context is shown before Wo; the
controlled projected single-head comparison adds the same Wo. Both projected
models have 4D² parameters. This comparison is untrained, not an accuracy contest.

## Residuals, normalization and FFN

A residual sublayer returns X+Sublayer(X), requiring identical shapes. If X row
is [1,2] and sublayer row [.2,-.5], result is [1.2,1.5]. The identity path can
support gradient flow through depth. Toggle it in the isolated MHA explorer.
Wo and FFN's second linear projection restore D before residual addition.

LayerNorm computes mean and population variance across features separately per
token. For [1,2,3,4], mean=2.5, variance=1.25. Subtract mean, divide by
sqrt(1.25+epsilon), multiply by learned gamma, and add learned beta. With gamma=1,
beta=0 and tiny epsilon, output is approximately [-1.342,-.447,.447,1.342].
Constant rows remain finite because epsilon is positive. Try larger epsilon,
gamma=2 or beta=.5. BatchNorm instead uses batch statistics in its normal use;
LayerNorm does not need batch-wide statistics or running averages.

FFN=Linear2(Activation(Linear1(X))). D=8,F=16 maps [B,T,8]→[B,T,16]→[B,T,16]
→[B,T,8]. Compare ReLU with GELU and inspect the actual weight/bias matrices.
The same FFN parameters act independently on each position. Attention mixes
positions; FFN transforms features at each position without recurrence.

## One block, two orders

Pre-LayerNorm:

    U = X + MHA(LN1(X))
    Y = U + FFN(LN2(U))

Post-LayerNorm:

    U = LN1(X + MHA(X))
    Y = LN2(U + FFN(U))

Choose the order in the block explorer. The diagram moves the two norms to match
the equation; dashed paths carry residuals. Select each stage to see its operation,
shape and numerical sample. The displayed attention is from that block's own
forward pass, including normalized attention input in pre-norm. No extra final
normalization is silently appended to pre-norm. There is no dropout in this lab.

## Learn through the block

Training uses eight fixed synthetic length-four sequences. Target token i is its
own vector plus half the mean of visible token vectors. Other positions affect
this target. Causal mode uses prefix means. Post-norm uses a feature-normalized
target to fit its normalized output range. No positional encoding is needed for
this permutation-compatible mean task.

Press Train explicitly. Ordinary UI reruns keep results without retraining.
Reset restores the seeded initialization; changing training settings clears stale
results. Default Adam, seed 42, 80 steps and rate .01 reduced measured MSE from
0.475769 to 0.002988. Other settings need not converge. Inspect loss curves,
per-parameter gradient norms, parameter deltas and before/after attention/output.
Gradients describe the last backward pass before its update; losses are measured
after updates. Head patterns alone do not prove semantic understanding or
generalization. This laboratory has no held-out performance claim.

## Count parameters and computation

For bias-free MHA, biased FFN and two affine norms: 4D² + (2DF+F+D) + 4D.
D=8,F=16 gives 256+280+32=568. The inspector counts real module parameters,
including gamma/beta and FFN biases, rather than trusting this formula alone.

Dense attention stores H*T² score cells. QK and AV each cost roughly T²D
multiply-accumulates, projections 4TD², FFN 2TDF. Doubling sequence length
quadruples score work while doubling projection/FFN work. At fixed D, changing
H leaves MHA parameter count and leading QK/AV arithmetic unchanged; storage and
softmax overhead can still change. These estimates omit several elementwise costs.

Complete twelve exercises, save notes, mark the lesson complete and export progress
JSON. Importing earlier progress preserves earlier lessons. Future encoder/decoder
milestones can compose these blocks with positions and task-specific structure;
M10 intentionally stops at one block without generation, caching or model stacks.
