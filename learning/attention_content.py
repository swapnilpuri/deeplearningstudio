"""Progressive explanations and deterministic checked exercises."""
SECTIONS = ('Introduction to Self-Attention', 'Query, Key, and Value', 'Attention Score Computation',
            'Scaling and Softmax', 'Attention Masking', 'Contextual Representation Generation',
            'Attention Visualization', 'Attention Training and Gradients', 'Experiments and Exercises')
EXPLANATIONS = (
    'M8 looks up a static vector for each token. Attention routes information between token positions to form contextual mixtures. Random embeddings and untrained projections are demonstrations, not learned language understanding.',
    'Each row is one token: Q=X@Wq, K=X@Wk, V=X@Wv. Queries describe what to match, keys what can be matched, and values what to mix. Bias-free weights use [input width, output width]; torch.nn.Linear stores their transpose.',
    'S=Q@K.T. Row i is a query; column j is a key. A score is the sum of coordinate-wise query/key products. Repeated token labels keep distinct position indices.',
    'Divide logits by sqrt(dk) before row-wise softmax. Under roughly independent, unit-variance coordinates, dot-product variance grows with dk. Scaling moderates this growth; it does not guarantee stability for every distribution. Softmax normalizes across keys because each query chooses a mixture of values.',
    'Causal visibility permits j<=i, preventing future-token leakage in autoregressive language models. Padding masks exclude padding keys. Padded query rows remain visible in this inspector; downstream losses should ignore them. Fully blocked rows are rejected. Negative infinity is intentional only in masked logits.',
    'Z=A@V: each output row is a weighted sum of value vectors. Its width is dv, which can differ from input width and dk. Changing other tokens can change an output even when its own static embedding stays fixed. No semantic learning is implied before training.',
    'Rows are queries, columns are keys in every heatmap. The selected-query distribution shows actual routing probabilities, not definitive explanations of reasoning. This single head has one attention distribution per query.',
    'Train a synthetic cyclic routing task: four identity input tokens should route to the next token (wrapping at the end). Loss is target-routing cross entropy plus output MSE against the first two coordinates of the routed token. Autograd differentiates through output mixing and softmax into all three projections. This is supervised routing, not language modeling.',
    'Try identity versus random projections; compare scaled and raw softmax; apply a causal mask; replace another token; and compare training seeds. Single-head attention has one routing pattern and no inherent positional order. M10 will introduce multiple heads; residuals, normalization and Transformer blocks remain future work.'
)
QUESTIONS = (
    ('X is [3,4], Wq/Wk [4,2], Wv [4,5]. Q,K,V shapes?', ('[3,2], [3,2], [3,5]', '[4,2], [4,2], [4,5]'), 0, 'Projection changes width, preserving token rows.'),
    ('q=[1,2], k=[3,4]. Dot product?', ('11', '10'), 0, '1×3 + 2×4 = 11.'),
    ('Which key has highest probability for logits [0,2,1]?', ('Key 1 (zero based)', 'Key 0'), 0, 'Softmax preserves logit ordering.'),
    ('Why scale by sqrt(dk)?', ('Moderate dimension-related logit magnitude', 'Guarantee all gradients stay finite'), 0, 'Variance assumptions motivate scaling, without universal guarantees.'),
    ('Softmax([0,0]) across keys is?', ('[0.5,0.5]', '[0,0]'), 0, 'Equal logits yield equal probabilities summing to one.'),
    ('Causal query position i=1 in length 3 blocks?', ('Key 2', 'Key 0'), 0, 'Only future keys j>i are blocked.'),
    ('A row=[.25,.75], V=[[2,0],[0,4]]. Output?', ('[.5,3]', '[2,4]'), 0, '.25×[2,0] + .75×[0,4] = [.5,3].'),
    ('Lower synthetic training loss proves?', ('Improved fit to this routing objective', 'General language understanding'), 0, 'Inspect measured before/after patterns; no general convergence claim.'),
    ('How does output differ from static input?', ('It mixes projected values using context-dependent weights', 'It is always the same lookup row'), 0, 'The surrounding keys and values affect the mixture.')
)


def evaluate_answers(answers):
    if len(answers) != len(QUESTIONS) or any(a not in q[1] for a,q in zip(answers,QUESTIONS)):
        raise ValueError('Answer every exercise first')
    return sum(a == q[1][q[2]] for a,q in zip(answers,QUESTIONS))
