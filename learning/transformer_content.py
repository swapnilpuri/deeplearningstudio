"""M10 lesson content and checked exercises, independent of Streamlit."""
SECTIONS = (
    'Introduction to Multi-Head Attention', 'Attention Head Configuration',
    'Query, Key, and Value Head Projections', 'Parallel Attention Heads',
    'Head Concatenation and Output Projection', 'Single-Head vs Multi-Head Comparison',
    'Residual Connections', 'Layer Normalization', 'Position-Wise Feed-Forward Network',
    'Transformer Block Explorer', 'Transformer Training Experiments', 'Interactive Learning Exercises')
EXPLANATIONS = (
    'Multiple heads route information through independently learned projections. Random patterns illustrate mechanics; meaningful complementary behavior requires training. Attention weights alone do not explain a model’s reasoning.',
    'Standard mode uses equal heads: d_head = d_model / num_heads. Q, K and V each project to d_model, split into heads, then rejoin. Invalid divisibility is rejected immediately. Custom head widths are outside this laboratory.',
    'Rows are tokens: Q_h=X@Wq_h, K_h=X@Wk_h, V_h=X@Wv_h. Each head uses its own columns of the packed projection weights. nn.Linear stores the transpose of this row-vector convention.',
    'Each head computes softmax(Q_h@K_h.T / sqrt(d_head) + mask)@V_h. Rows are queries and columns are keys. Causal visibility permits j<=i; padding excludes keys. Fully blocked query rows are invalid. Padded query outputs remain present and should be ignored in downstream losses.',
    'Concatenate head outputs in head order, then multiply by Wo. Wo mixes features across heads and restores d_model for residual addition. The inspector uses the exact differentiable model forward pass.',
    'At fixed d_model, bias-free standard MHA has 4*d_model² parameters regardless of head count. Different head widths change routing capacity; more heads do not guarantee better results. This comparison is untrained and controlled by identical inputs and packed weights. M9 supplies each head’s tested attention operation; its context output precedes Wo.',
    'Y=X+Sublayer(X) keeps an identity gradient path alongside the learned path. This can help learning in depth, without guaranteeing convergence. Wo restores the attention width before addition; FFN also returns d_model.',
    'LayerNorm normalizes model features separately for each token, not across a batch. Variance uses correction=0. Output = gamma*(x-mean)/sqrt(variance+epsilon)+beta. Epsilon stabilizes constant feature rows; gamma and beta are learned. This per-token behavior suits variable sequence lengths.',
    'FFN applies Linear2(Activation(Linear1(X))) independently at every token using shared weights. Hidden width can expand feature capacity; ReLU and GELU introduce nonlinearity. Batch and sequence dimensions stay fixed.',
    'Pre-norm: U=X+MHA(LN1(X)); Y=U+FFN(LN2(U)). Post-norm: U=LN1(X+MHA(X)); Y=LN2(U+FFN(U)). Select a stage to inspect the actual tensors. Dashed diagram edges are residual paths.',
    'Train eight fixed synthetic sequences to predict X + 0.5*mean(visible tokens). The target depends on other positions; causal mode uses prefix means. Post-norm uses a feature-normalized target compatible with its output. This demonstrates fitting and gradient flow, not linguistic specialization or held-out generalization.',
    'Use the numerical inspectors to check your answers. Completion, notes, quiz results and progress import/export reuse the existing progress v1 format.')
QUESTIONS = (
    ('d_model=12, heads=3. Per-head width?', ('4','3','12'), 0, '12 / 3 = 4 features per head.'),
    ('Which standard configuration is valid?', ('d_model=8, heads=2','d_model=7, heads=2'), 0, 'The model width must be divisible by the head count.'),
    ('Why separate projections per head?', ('Learn different routing and value subspaces','Force identical attention'), 0, 'Each head owns different learned projection columns.'),
    ('A head has 5 query and 5 key tokens. Attention shape?', ('[5,5]','[5,d_model]'), 0, 'Every query has one probability for every key.'),
    ('Three head outputs are [5,4]. Concatenation shape?', ('[5,12]','[15,4]'), 0, 'Concatenation joins feature columns in head order.'),
    ('What does Wo do?', ('Mix concatenated features and return model width','Compute softmax'), 0, 'Output = ConcatHeads @ Wo.'),
    ('Residual operation?', ('X + Sublayer(X)','X @ Sublayer(X)'), 0, 'Addition preserves an identity path and needs equal shapes.'),
    ('LayerNorm normalizes over?', ('Features within each token','Tokens across the batch'), 0, 'LayerNorm uses feature-axis mean and population variance.'),
    ('Which expression is pre-norm attention?', ('X + MHA(LN(X))','LN(X + MHA(X))'), 0, 'Pre-norm normalizes before the sublayer.'),
    ('The position-wise FFN uses?', ('Shared parameters independently at each token','A recurrent hidden state'), 0, 'Two linear projections and an activation act on features.'),
    ('D=8, FF=16: bias-free MHA, biased FFN, two affine LNs. Total?', ('568','512'), 0, '256 MHA + (128+16+128+8) FFN + 32 LayerNorm = 568.'),
    ('Complementary head behavior is?', ('Possible through learning, not guaranteed','Guaranteed by random initialization'), 0, 'Inspect learned behavior for the actual task; avoid universal superiority claims.')
)


def evaluate_answers(answers) -> int:
    if len(answers) != len(QUESTIONS) or any(a not in q[1] for a,q in zip(answers,QUESTIONS)):
        raise ValueError('Answer every exercise first')
    return sum(a == q[1][q[2]] for a,q in zip(answers,QUESTIONS))
