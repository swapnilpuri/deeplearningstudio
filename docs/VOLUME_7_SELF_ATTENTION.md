# Volume 7 — Single-head self-attention

Start in **Learning Studio → Single-Head Self-Attention Learning Studio**.
M8 converts token IDs into static vectors. M9 mixes information across positions.
All core computation is offline CPU PyTorch. Synthetic embeddings and untrained
projections do not imply semantic understanding.

## Row-vector convention and shapes

For T token positions, embedding width D, query/key width dk and value width dv:

| Stage | Operation | Shape |
|---|---|---|
| Input | M8 lookup or custom matrix | [T,D] |
| Query | X @ Wq, Wq [D,dk] | [T,dk] |
| Key | X @ Wk, Wk [D,dk] | [T,dk] |
| Value | X @ Wv, Wv [D,dv] | [T,dv] |
| Scores | Q @ K.T | [T,T] |
| Scaling | scores / sqrt(dk) | [T,T] |
| Mask | forbidden logits become -infinity | [T,T] |
| Routing | softmax over permitted keys (last axis) | [T,T] |
| Output | A @ V | [T,dv] |

The implementation uses explicit bias-free matrices. A PyTorch Linear would store
[Dout,Din] weights and compute X @ weight.T (plus bias if enabled).

A query is a matching request, a key a matching descriptor, and a value the content
to mix. Every token has all three roles, produced by different learned projections.
Rows of A answer: for this query, what fraction of each value should be routed here?
They sum to one; causal and padding-forbidden entries are exactly zero.

## Worked numerical example

Let X=[[1,0],[0,1]], Wq=Wk=identity(2), Wv=[[1],[0]].
Then Q=K=X [2,2], V=[[1],[0]] [2,1], scores=identity(2) [2,2].
For query 0 and key 1, [1,0] dot [0,1] = 1×0 + 0×1 = 0.
Divide by sqrt(2): logits are [[0.707107,0],[0,0.707107]].
Row-wise softmax yields approximately [[0.669762,0.330238],
[0.330238,0.669762]]. The first output is
0.669762×[1] + 0.330238×[0] = [0.669762]. Z has shape [2,1].
This example is verified in tests/test_attention.py; the document does not carry
an alternative numerical implementation.

Causal visibility is [[1,0],[1,1]]. The first row's masked logits are
[0.707107,-infinity], so its probabilities become [1,0] and output [1].
Bidirectional mode can read both positions; causal mode prevents future-token
leakage during autoregressive training. Padding mode ignores selected trailing
keys; padded query rows are still displayed, and should be excluded from any
real sequence loss. Fully masked query rows are rejected rather than normalized.
Intentional -infinity in masked logits is not an overflow; other supported stage
outputs remain finite. Oversized input magnitudes that overflow are rejected.

## Scaling and stable softmax

For independent unit-variance query/key coordinates, variance of their dot product
grows with dk. Division by sqrt(dk) moderates this growth. Scaling does not guarantee
stability for arbitrary input distributions. PyTorch's stable softmax avoids directly
exponentiating enormous logits without subtracting their row maximum. Switch scaling
off in the lab and compare both distributions. Softmax operates along keys because
each query row needs its own normalized mixture, not a normalization across queries.

## Using the studio

Nine progressive sections connect motivation, projections, scores, scaling,
masking, outputs, visualization, training, and exercises. The matrix inspector
shows dtype, shape, operation and numerical samples. Token labels include position
indices so repeated words remain distinct. Use query/key selectors or select a
score-table cell to inspect its coordinate products; hover heatmaps for values.
The selected-query bar chart shows actual token-to-token routing coefficients.
Weighted value contribution rows sum to the selected output representation.

Input modes reuse the M8 tokenizer and seeded table, accept a custom [T,D] JSON
matrix, or reuse a completed M8 skip-gram run from the same session. For the latter,
visit M8 Word embedding training, train explicitly, then return to M9 and choose
M8 trained skip-gram. Its actual width and vocabulary take precedence; OOV tokens
use UNK. If you retrain M8, M9 reads the updated weights. No downloads are required.
Custom projection JSON must match the displayed dimensions. Normal, rectangular
identity and zero initialization are available. Defaults are T=6,D=4,dk=3,dv=2.
UI widths are 1–16; reused M8 widths and mathematical model widths are capped at 32;
sequences are 1–24 tokens. All demonstrations use reproducible seeds.

## Training and gradients

Training has its own fixed four-token identity input, independent of the visual
sequence. The target routes A→B, B→C, C→D, D→A. The loss combines supervised routing
cross entropy and output MSE against the first two coordinates of each target token.
Routing supervision provides an easily checked pattern; MSE propagates gradients
through Wv as well. Autograd propagates through weighted mixing, softmax, and Q/K
projections into Wq and Wk. The target is synthetic, not language data.

Press **Train attention** to initialize from the selected seed and perform the
requested SGD or Adam steps. Results show the pre-training and post-training routing,
loss, gradient norms, initial/trained weights, final pre-update gradients and weight
deltas. **Reset to initial weights** discards trained results and displays the same
seeded initial matrices. Changing a training control invalidates stale results;
ordinary section/widget reruns do not train. Loss history records the initial
objective and each post-update objective; gradient curves record pre-update norms.
Every Train action starts a fresh reproducible run. The default reduces loss
substantially, but arbitrary rates/seeds/budgets need not converge.

## Exercises and limitations

The nine checked exercises cover shapes, dot products, highest probability, scaling,
softmax, causal visibility, weighted output, measured learning and static/contextual
representations. Scores, notes and completion reuse progress schema v1; export JSON
to retain them beyond the session and import older progress without losing lessons.

Try the same word in two sequences and compare its static row and contextual output.
Random projections may create contextual differences without meaningful semantics.
Attention weights are routing/mixing coefficients, not definitive explanations of
reasoning. A single head has one routing distribution per query. Without positional
inputs, bidirectional attention is permutation equivariant, not aware of word order.
M10 can compose multiple heads using the tested primitives; positional inputs,
residual connections, normalization and full Transformer blocks remain later work.

Implementation source of truth: learning/attention_adapter.py (math),
attention_spec.py (dimensions), attention_training.py (experiment),
attention_content.py (exercises), attention_ui.py (presentation), and
visualization/attention_graphs.py (Plotly preparation).
