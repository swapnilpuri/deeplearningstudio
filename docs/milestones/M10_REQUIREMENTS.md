# M10 requirements and scope

Implement Multi-Head Attention & Transformer Blocks in the existing Learning Studio,
incrementally from merged M9. Source request: user-supplied M10 specification,
2026-11-21. No earlier milestone rewrite or additional learning application.

1. Twelve sections: introduction, configuration, Q/K/V projections, parallel heads,
   concatenation/output projection, single/multiple comparison, residuals,
   LayerNorm, FFN, block explorer, training, exercises.
2. Reuse M8 embeddings and M9 differentiable attention and visibility. Show input
   values/shapes; accept seeded synthetic, M8 lookup, session-trained skip-gram,
   and custom matrix inputs. Label untrained representations honestly.
3. Standard equally sized heads: D divisible by H, dk=dv=D/H. Validate dimensions
   immediately. Inspect head-specific Q/K/V/weights, scores, probabilities,
   output, concatenation and Wo. Custom head widths are optional and not included.
4. Bidirectional, causal, padding and combined masks. Keep query/key token order,
   reject all-blocked rows, preserve autograd. Probability heatmaps share [0,1].
5. Controlled untrained comparison with single-head/M9; actual parameter counts,
   tensor sizes and approximate costs; no universal superiority claims.
6. Residual addition, affine PyTorch feature-axis LayerNorm with population
   variance and epsilon, shared position-wise ReLU/GELU FFN.
7. One block with both specified pre/post-norm equations, adaptive residual-path
   diagram and selectable stages with operations, shapes and actual tensors.
8. Explicit deterministic CPU training, SGD/Adam, seed/rate/steps/heads/FF width/
   activation controls, loss/gradient curves, before/after heads and outputs,
   parameter changes and reset. No retraining on ordinary reruns.
9. Twelve checked exercises, feedback, notes and existing progress v1 portability.
10. PyTorch reference parity, masking/autograd/normalization/composition tests,
    UI/progress integration and full M1–M9 regression. Record actual evidence.

Keep mathematical code independent of Streamlit, no new framework/service,
downloads, GPU requirement or external APIs. No encoder/decoder stacks,
cross-attention, GPT, generation, caching, optimized attention or serving.

Execution batches: core/reference tests; head UI; block components; training and
exercises; regression/docs/feature branch/PR. Do not merge master.
