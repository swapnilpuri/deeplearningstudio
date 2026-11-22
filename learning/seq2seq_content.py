"""M11 curriculum, following existing question tuples and progress v1 contracts."""
SECTIONS = ('Concepts & Math', 'Encoder', 'Decoder', 'Cross-Attention',
            'Seq2Seq & Training', 'Quiz', 'Exercises & Progress')

QUESTIONS = (
    ('Why add positional information?', ('To identify token order', 'To remove all masks'), 0,
     'Self-attention without position information is permutation equivariant; positions distinguish order.'),
    ('Which self-attention is causal?', ('Encoder', 'Autoregressive decoder'), 1,
     'An encoder can see all non-PAD source keys; a decoder can see only its current and past keys.'),
    ('For query position i, causal visibility permits?', ('j <= i', 'j > i'), 0,
     'The lower triangle including the diagonal allows current and previous keys.'),
    ('B=2, source length=5, D=8. Encoder output?', ('[2,5,8]', '[2,8,5]'), 0,
     'The studio uses batch-first [B,S,D]; residuals preserve feature width.'),
    ('What is encoder memory?', ('Final contextual source representations', 'Future target labels'), 0,
     'The final encoder output supplies cross-attention keys and values.'),
    ('B=2, H=2, target=3, source=5. Cross-score shape?', ('[2,2,3,5]', '[2,2,5,3]'), 0,
     'Scores have axes [batch,head,target query,source key].'),
    ('Cross-attention Q / K,V originate from?', ('Decoder / encoder memory', 'Encoder / decoder'), 0,
     'Decoder queries select information from encoder keys and values.'),
    ('Teacher forcing supplies the decoder with?', ('Previous ground-truth tokens', 'Future labels at the same position'), 0,
     'Shifted inputs supply the correct prefix during training; causal attention blocks future inputs.'),
    ('Target [BOS,A,B,EOS]: input / labels?', ('[BOS,A,B] / [A,B,EOS]', '[A,B,EOS] / [BOS,A,B]'), 0,
     'A position predicts the next token; PAD labels contribute no loss.'),
    ('How are inference target tokens produced?', ('One at a time from prior predictions', 'All true labels are provided'), 0,
     'The encoder runs once; greedy decoding appends the largest-logit token to its prefix.'),
    ('Which D/head configuration is valid?', ('D=12, H=3', 'D=10, H=3'), 0,
     'Standard equal heads require D divisible by H.'),
    ('Which architecture typically has cross-attention?', ('Encoder-decoder', 'Encoder-only'), 0,
     'Encoder-decoder models connect distinct source and target streams through memory.'),
    ('Cross-attention QK/AV cost scales as?', ('O(T*S*D)', 'O(T+S)'), 0,
     'Every target query compares with every source key; both QK and AV have this leading cost.'),
    ('What does a source padding mask block?', ('PAD source keys in encoder and cross-attention', 'Only future target keys'), 0,
     'Source padding and target causality have separate roles and may have different lengths.'),
    ('Why stop greedy decoding at EOS?', ('It marks sequence completion', 'It always marks a padding input'), 0,
     'EOS is learned as a prediction target. A maximum length also bounds generation.'),
)

# Same question representation and correctness rule as M9/M10; no new quiz engine.
def evaluate_answers(answers):
    if len(answers) != len(QUESTIONS) or any(a not in q[1] for a, q in zip(answers, QUESTIONS)):
        raise ValueError('Answer every exercise first')
    return sum(a == q[1][q[2]] for a, q in zip(answers, QUESTIONS))


EXERCISES = (
    'Set sinusoidal positions to length 3 and D=4. Verify position 0 is [0,1,0,1]; compare learned positions.',
    'Build two encoder layers with B=1,S=4,D=8. Compare the layer outputs; identify final memory [1,4,8].',
    'Inspect the 3×3 causal mask: rows are [1,0,0], [1,1,0], [1,1,1]. Change a future target and explain why earlier predictions stay fixed.',
    'Use source length 5 and target length 3. Verify cross-scores [1,H,3,5], Q from decoder and K/V from memory.',
    'Add source PAD, then inspect encoder and cross probabilities. PAD columns must be zero; permitted rows sum to one.',
    'Shift [BOS,T0,T1,EOS,PAD] into inputs [BOS,T0,T1,EOS] and labels [T0,T1,EOS,PAD]. Identify the ignored loss position.',
    'Train the default copy task for 150 Adam steps at 0.01, seed 42. Compare teacher-forced accuracy and greedy exact matches; they measure different behaviors.',
    'Compare pre- and post-norm diagrams; trace every residual. Estimate encoder S², decoder T² and cross T*S score cells per head.',
)
