# DeepLearningStudio — Milestone 8: Embedding Network Learning Studio

## 1. Project Context

We are developing **DeepLearningStudio**, an interactive educational platform for learning neural networks, deep learning, and Generative AI through real model execution, mathematical explanations, interactive visualizations, experiments, and quizzes.

**GitHub repository:**
https://github.com/swapnilpuri/deeplearningstudio

**Local VS Code workspace:**
`C:\Personal\Swapnil\Learning\Innovation\neural-network-studio`

**Current milestone:** M8 — Embedding Network Learning Studio.

**Technology stack:**
- Windows 11
- Python 3.12
- PyTorch
- Streamlit
- Plotly
- NumPy
- scikit-learn
- ONNX/Netron
- uv package manager
- VS Code

### Completed learning milestones

- M4 — Feed-Forward and Deep Neural Networks.
- M5 — Convolutional Neural Networks.
- M6 — Recurrent Neural Networks.
- M7 — Long Short-Term Memory Networks.

The application also contains Neural Network Studio, Dataset Management, Training Studio, and Learning Studio.

M7 was implemented and merged into master.

Latest master commit observed when this prompt was prepared:

`dd878cac84c789ce14c17eb6061e32ec220baf6a`

Verify the current Git repository before beginning. Do not assume this commit remains the latest.

The M7 completion documentation reports:
- 181 automated tests passed.
- Running Streamlit workflows exercised.
- Core browser interaction checks completed.
- Some file upload/import and independent acceptance checks remained unverified.

M8 must preserve all existing functionality and documentation.

## 2. Mandatory AI Engineering Standards

Before beginning implementation, read:

1. `AGENTS.md`
2. `docs/AI_DEVELOPMENT_OPERATING_MODEL.md`
3. `docs/ROADMAP.md`
4. `docs/milestones/M7_HANDOFF.md`
5. `docs/milestones/M7_COMPLETION_REPORT.md`
6. Relevant M7 architecture, requirements, and tests.

Both engineering-standard files already exist. Do not recreate them or replace established standards unnecessarily.

### Responsibilities

ChatGPT Web UI:
- Requirements and solution architecture.
- Educational content and milestone design.
- Architecture reviews.
- Independent completion reviews.
- Long-term roadmap planning.

Codex in VS Code:
- Implementation and refactoring.
- Debugging.
- Automated testing.
- Local Streamlit execution.
- Documentation maintenance.
- Git branching, commits, and pull requests.

GitHub:
- Authoritative source of truth for committed code, architecture decisions, requirements, tests, and milestone handoffs.

Use repository documentation rather than conversational history to synchronize ChatGPT and Codex.

## 3. Codex Execution Efficiency — Mandatory

The user has experienced excessive thinking time during Codex implementation.

Optimize development efficiency without sacrificing quality.

Follow these rules:

1. Start with a concise execution plan of no more than seven steps.
2. Inspect the existing repository once at the beginning.
3. Read only files relevant to the current implementation batch.
4. Avoid repeatedly scanning unchanged source files.
5. Reuse established project architecture and utilities.
6. Avoid unnecessary architecture redesign or speculative abstraction.
7. Implement in small, independently testable functional batches.
8. Run targeted tests after each batch.
9. Run full regression at major integration checkpoints and final completion.
10. Avoid repeating expensive test commands without a reason.
11. Batch independent read-only operations when safe.
12. Never parallelize conflicting source modifications or Git writes.
13. Keep command output and progress reports concise.
14. Make reasonable routine engineering decisions autonomously.
15. Do not pause implementation merely to produce long planning documents.
16. Maintain milestone handoff documentation at meaningful checkpoints.
17. Do not create duplicate project folders or ZIP files.
18. Do not introduce unnecessary dependencies.
19. Do not disable tests merely to make regression results green.
20. Preserve security, correctness, backward compatibility, and educational accuracy.

Treat this comprehensive requirements document as the milestone reference, but execute it in focused implementation batches.

The prompt cannot directly control model inference latency. Use the configured model and reasoning effort, and rely on narrow tasks and targeted tool use to reduce avoidable overhead.

**Priority: When the next safe engineering action is clear, implement it promptly instead of repeatedly reconsidering the full milestone.**

## 4. Primary M8 Objective

Implement an interactive **Embedding Network Learning Studio** that teaches how discrete inputs become learned continuous vector representations.

Learners must understand:

- What embeddings are.
- Why neural networks need numerical representations.
- How text is tokenized.
- How vocabularies and token IDs work.
- One-hot representations versus dense embeddings.
- How embedding matrices work.
- How PyTorch Embedding layers execute.
- How embedding vectors are learned through gradients.
- How semantic relationships emerge from training.
- How embedding similarity is calculated.
- How nearest-neighbor retrieval works.
- How word and sentence embeddings differ.
- How positional embeddings represent sequence order.
- How embedding representations are evaluated.
- Why static embeddings differ from contextual embeddings.

M8 must lay the conceptual and architectural foundation for later attention and Transformer lessons.

Do not implement complete self-attention, multi-head attention, or Transformer networks during M8.

## 5. Phase A — Baseline Inspection

Before modifying code:

1. Verify the local workspace.
2. Inspect Git status, current branch, and master.
3. Protect existing uncommitted changes.
4. Confirm M7 has been merged.
5. Review existing learning modules, framework-neutral specifications, adapters, visualization utilities, training infrastructure, and progress tracking.
6. Run baseline tests.
7. Record the starting commit and baseline test results.

Create the feature branch:

`feature/m8-embedding-learning-studio`

Create:

- `docs/milestones/M8_REQUIREMENTS.md`
- `docs/milestones/M8_DESIGN.md`
- `docs/milestones/M8_ACCEPTANCE_CRITERIA.md`
- `docs/milestones/M8_HANDOFF.md`
- `docs/milestones/M8_COMPLETION_REPORT.md`

Use this prompt as the initial requirements source.

Document architecture decisions where appropriate.

Do not stop after creating documentation. Continue implementation.

## 6. Phase B — Embedding Learning Module

Add a new lesson:

**Embedding Network**

Preserve all existing Learning Studio lessons.

Provide beginner-friendly explanations and demonstrations for:

1. Discrete versus continuous representations.
2. Categorical feature encoding.
3. One-hot encoding.
4. Dense vector representations.
5. Tokenization.
6. Vocabulary construction.
7. Token IDs.
8. Embedding matrices.
9. Embedding dimensions.
10. Embedding lookup.
11. Random initialization.
12. Learned embedding weights.
13. Gradient-based embedding learning.
14. Similarity and distance metrics.
15. Nearest-neighbor retrieval.
16. Word embeddings.
17. Sentence embeddings.
18. Static versus contextual embeddings.
19. Positional embeddings.
20. Embedding evaluation.
21. Applications in search, recommendations, and NLP.
22. Limitations and risks of learned representations.

For each major concept provide:
- Intuitive explanation.
- Mathematical explanation.
- Tensor dimensions.
- Worked numerical example.
- Real PyTorch demonstration.
- Interactive visualization.
- Guided experiment.

Explain why embeddings are foundational for attention, Transformers, and LLMs.

## 7. Phase C — Tokenization and Vocabulary Explorer

Implement an interactive text-processing laboratory.

Allow learners to enter text and observe:

`Raw Text → Tokenization → Vocabulary → Token IDs → Embedding Lookup`

Support at minimum:
- Whitespace/word-level tokenization.
- A simple character-level tokenizer.
- A predefined toy vocabulary.
- Vocabulary generation from a small text corpus.
- Special tokens such as PAD and UNK.

Display:
- Raw input text.
- Token sequence.
- Vocabulary mapping.
- Token IDs.
- Vocabulary size.
- Unknown-token handling.
- Padding behavior.
- Token-frequency distribution.

Explain:
- Why tokenization is required.
- How vocabulary size affects embedding tables.
- Why token IDs are categorical labels, not continuous numerical quantities.
- Differences between word and character tokenization.
- Why real LLMs often use subword tokenization.

Optional: demonstrate a simple BPE-style tokenizer if it can be done without adding unnecessary complexity.

Do not implement a production tokenizer or require downloading tokenizer models.

Keep tokenization separate from embedding computation.

## 8. Phase D — One-Hot Encoding versus Embeddings

Provide side-by-side demonstrations.

For a vocabulary of size V:

One-hot representation:

`x ∈ R^V`

Embedding vector:

`e ∈ R^D`

Where:
- V = vocabulary size.
- D = embedding dimension.

Show:
- One-hot vectors.
- Dense embedding vectors.
- Sparsity differences.
- Memory representation differences.
- Dimensionality differences.
- Embedding table lookup.

Demonstrate mathematically:

`e = x × E`

Where E is the embedding matrix with shape `[V, D]`.

Show that a one-hot matrix multiplication selects the corresponding embedding row.

Verify numerical agreement with PyTorch `nn.Embedding` for small examples.

Explain that direct lookup avoids explicitly constructing large one-hot vectors.

Do not imply that dense embeddings always require less total storage than one-hot token IDs; distinguish vector representation size, sparse storage, and embedding-table parameters.

## 9. Phase E — Interactive Embedding Matrix Explorer

Build a configurable embedding explorer.

Allow:
- Vocabulary size.
- Embedding dimension.
- Random seed.
- Token selection.
- Weight initialization.
- Optional custom embedding values.
- Padding index.

Use actual PyTorch `nn.Embedding`.

Display:
- Embedding weight matrix.
- Token-to-row mapping.
- Selected embedding vector.
- Tensor shape.
- Parameter count.
- Selected row values.
- Embedding matrix heatmap.

Explain:

`Embedding Parameters = Vocabulary Size × Embedding Dimension`

For an embedding table E:

`E.shape = [V, D]`

For token-ID input:

`X.shape = [B, T]`

The embedding output shape is:

`[B, T, D]`

Explain:
- B = batch size.
- T = sequence length.
- D = embedding dimension.

Show how different token IDs select different rows.

Verify that repeated token IDs initially return identical embedding vectors from the same table.

Explain that tokens are not contextualized merely because they appear in a sequence.

## 10. Phase F — Embedding Learning and Backpropagation

Implement a learning demonstration showing how embedding vectors are updated.

Use a small deterministic PyTorch model.

Demonstrate:

1. Initial embedding matrix.
2. Input token IDs.
3. Embedding lookup.
4. Downstream prediction.
5. Loss computation.
6. `optimizer.zero_grad()`.
7. `loss.backward()`.
8. Embedding gradients.
9. Optimizer update.
10. Updated embeddings.
11. Loss after one update.

Show:

`E_new = E_old - learning_rate × gradient`

For plain SGD.

Explain that Adam uses adaptive updates rather than this simple equation.

Highlight which embedding rows received gradients.

Explain:
- Repeated token occurrences accumulate gradient contributions.
- Unused token rows normally receive no gradient in simple lookup-only demonstrations.
- Padding indices can be excluded from updates.
- Dense and sparse embedding gradients are different implementation options.
- Optimizer compatibility matters when sparse gradients are enabled.

Support numerical gradient verification on a small deterministic example where practical.

## 11. Phase G — Semantic Similarity Explorer

Implement a configurable vector-similarity laboratory.

Support:
- Dot product.
- Cosine similarity.
- Euclidean distance.
- Manhattan distance.

Explain the equations.

### Cosine similarity

`cosine(A, B) = dot(A, B) / (norm(A) × norm(B))`

### Dot product

`dot(A, B) = Σ(A_i × B_i)`

### Euclidean distance

`distance(A, B) = sqrt(Σ(A_i - B_i)^2)`

### Manhattan distance

`distance(A, B) = Σ|A_i - B_i|`

Allow learners to:
- Select two embedding vectors.
- Modify values.
- Normalize vectors.
- Compare similarity results.
- Inspect vector norms.
- Observe how magnitude affects dot product.
- Observe how direction affects cosine similarity.

Handle zero vectors safely.

Explain that smaller Euclidean/Manhattan distance means closer vectors, while larger cosine/dot-product values usually indicate greater similarity under the relevant assumptions.

Clearly distinguish raw randomly initialized vectors from semantically trained embeddings.

Do not claim random vectors encode semantic relationships.

## 12. Phase H — Word Embedding Training Laboratory

Implement a small offline word-embedding learning experiment.

Use a deterministic toy corpus with repeated semantic or co-occurrence patterns.

Prefer a simple **Skip-gram with Negative Sampling** implementation using real PyTorch operations.

Explain:

- Target word.
- Context word.
- Context window.
- Positive pairs.
- Negative samples.
- Input embedding matrix.
- Output/context embedding matrix.
- Training objective.
- Gradient updates.

Demonstrate the negative-sampling objective for positive target/context pairs and sampled negative contexts.

Use an appropriate log-sigmoid-based loss.

Make the loss numerically stable.

Support:
- Configurable embedding dimension.
- Context-window size.
- Negative-sample count.
- Learning rate.
- Batch size.
- Training epochs.
- Random seed.

Display:
- Training loss curve.
- Initial embeddings.
- Trained embeddings.
- Nearest neighbors.
- Similarity changes.
- Embedding parameter counts.

Prevent positive context pairs from being deliberately treated as negatives within a training example where practical.

Ensure negative sampling is reproducible under fixed seeds.

Explain that toy-corpus nearest neighbors reflect co-occurrence patterns and may not match human semantic intuition.

Do not claim meaningful semantic quality based solely on decreasing training loss.

## 13. Phase I — Embedding Space Visualization

Provide interactive embedding-space plots.

Support:
- Direct 2D embeddings.
- Direct 3D embeddings.
- PCA projections for higher-dimensional embeddings.
- Optional t-SNE visualization for sufficiently sized datasets.

Use Plotly.

Display:
- Token labels.
- Vector coordinates.
- Nearest-neighbor relationships.
- Selected-token highlighting.
- Initial versus trained embeddings.

Allow learners to:
- Change embedding dimension.
- Change random seed.
- Train the embedding model.
- Select a word.
- Inspect nearest neighbors.
- Compare projections.

Explain:
- High-dimensional embeddings cannot be completely represented by 2D projections.
- PCA preserves directions of maximum variance.
- t-SNE emphasizes local neighborhoods and can distort global distances.
- Projected distances need not match distances in the original embedding space.

Keep results deterministic where supported.

Validate t-SNE perplexity relative to sample count.

Avoid using t-SNE when too few samples are available.

## 14. Phase J — Nearest-Neighbor Retrieval Explorer

Implement a small embedding-based retrieval demonstration.

Given a query embedding, calculate similarity against a collection of candidate vectors.

Show:

`Query → Embedding → Similarity Calculation → Ranked Neighbors`

Support:
- Cosine similarity.
- Dot-product similarity.
- Top-K selection.
- Optional normalization.
- Similarity score inspection.

Use exact vector search with PyTorch or NumPy for the small datasets.

No external vector database is required.

Explain:
- Exact versus approximate nearest-neighbor search.
- Why vector indexes are useful for large collections.
- Why embedding model consistency matters.
- Why query/document embeddings must be compatible.
- Why cosine and dot product can rank differently without normalization.

Provide an optional simple semantic-search exercise using a trained toy embedding model.

Do not claim high-quality semantic retrieval from an untrained embedding table.

Do not implement a complete RAG system during M8.

## 15. Phase K — Sentence Embedding Laboratory

Demonstrate how word embeddings can be combined into fixed-size sentence representations.

Support:
- Mean pooling.
- Sum pooling.
- Max pooling.
- Optional masked pooling for padded tokens.

Display:
- Sentence tokens.
- Individual token embeddings.
- Pooling operation.
- Final sentence embedding.
- Similarity between sentences.

Explain the limitations of simple pooling.

Important:
- Ignore padding tokens when appropriate.
- Handle empty and all-padding sequences gracefully.
- Do not confuse pooling with context-sensitive encoding.
- Explain that word order is largely lost in simple commutative pooling.

Provide examples demonstrating that differently ordered sentences can produce identical mean-pooled representations when they contain the same token embeddings.

## 16. Phase L — Positional Embedding Explorer

Introduce positional information as preparation for Transformer learning.

Cover:
- Why word order matters.
- Position indices.
- Learned positional embeddings.
- Sinusoidal positional encodings.
- Adding token and positional vectors.

Explain:

`InputRepresentation = TokenEmbedding + PositionalEncoding`

For compatible dimensions.

Provide two demonstrations.

### Learned positional embeddings

Use a PyTorch embedding table indexed by positions.

Display:
- Token IDs.
- Token vectors.
- Position IDs.
- Position vectors.
- Combined vectors.

### Sinusoidal positional encodings

Explain the standard sine/cosine formulation using position and embedding-dimension indices.

Implement correct handling of even and odd embedding dimensions.

Display positional encoding heatmaps.

Allow users to:
- Change sequence length.
- Change embedding dimension.
- Compare positions.
- Reverse token order.
- Inspect resulting combined vectors.

Clarify that positional embeddings/encodings introduce position-dependent information, but do not by themselves perform contextual mixing or attention.

Do not implement Transformer attention in M8.

## 17. Phase M — Static versus Contextual Embeddings

Provide a conceptual comparison.

Explain:
- Static word embeddings.
- Contextual representations.
- Polysemy.
- Position-dependent input representations.
- Why attention enables contextualization.

Use examples:

"The weather is cool."

"That movie is cool."

Explain why a conventional static lookup returns the same vector for the token "cool" in both sentences, while a contextual encoder can produce different representations.

Do not falsely represent token-plus-position embeddings as fully contextual embeddings.

Provide a small interactive comparison of static token vectors in different sentences.

A full contextual Transformer encoder is outside scope.

## 18. Phase N — Embedding Architecture Builder

Extend the existing framework-neutral model architecture where appropriate.

Support at minimum:
- Token-ID input.
- Embedding lookup.
- Optional positional representation addition.
- Token pooling.
- Linear classification head.

Display:
- Architecture diagram.
- Vocabulary size.
- Embedding dimensions.
- Tensor shapes at every stage.
- Weight tensor shapes.
- Parameter counts.
- Output dimensions.

Example:

`Token IDs [B,T]`
→ `Embedding [B,T,D]`
→ `Masked Mean Pooling [B,D]`
→ `Linear [B,C]`
→ `Class Logits`

Infer dimensions rather than hardcoding them.

Avoid adding general-purpose abstractions beyond M8 needs.

## 19. Phase O — Framework-Independent Specifications

Extend the existing architecture with typed embedding-related specifications.

Requirements:

1. No PyTorch imports in framework-neutral specification modules.
2. Versioned serializable model definitions.
3. Vocabulary size and embedding dimension validation.
4. Padding-index validation.
5. Optional positional-encoding specification.
6. Pooling specification.
7. Shape propagation.
8. Parameter-count validation.
9. PyTorch-specific execution in adapters.
10. Backward compatibility with dense, CNN, RNN, and LSTM specifications.

Do not change old serialized formats unnecessarily.

Use a new compatible schema version if required.

Maintain extensibility for future attention and Transformer lessons without implementing those architectures prematurely.

Create a concise ADR documenting design decisions.

## 20. Phase P — Downstream Embedding Classification Laboratory

Implement a small supervised task demonstrating how embeddings are learned jointly with a classifier.

Use a deterministic offline text dataset or a generated toy corpus.

Support:

`Token IDs → Embedding → Masked Pooling → Linear Classifier`

Provide:
- Train/validation/test split.
- Train-only vocabulary fitting.
- UNK handling for unseen tokens.
- Padding.
- Mini-batches.
- CrossEntropyLoss.
- SGD and Adam.
- Configurable epochs.
- Learning rate.
- Embedding dimension.
- Batch size.

Display:
- Training loss.
- Validation loss.
- Training accuracy.
- Validation accuracy.
- Final held-out test metrics.
- Confusion matrix.
- Sample predictions.
- Embedding gradients.
- Initial versus trained embeddings.

Avoid vocabulary/data leakage.

Do not select hyperparameters using the final test split.

Keep the experiment lightweight enough for CPU execution.

## 21. Phase Q — Embedding Evaluation Laboratory

Provide evaluation demonstrations for:

1. Training-loss behavior.
2. Similarity stability.
3. Nearest-neighbor inspection.
4. Retrieval ranking.
5. Downstream classification accuracy.
6. Embedding-space visualization.
7. Sensitivity to random initialization.

Include limitations:

- Small training corpora.
- Rare tokens.
- Out-of-vocabulary words.
- Biased training distributions.
- Polysemy.
- Domain shift.
- Language coverage.
- Dimensionality tradeoffs.

Optionally provide simple word-analogy experiments.

Explain that vector analogies such as:

`king - man + woman ≈ queen`

are illustrative patterns observed in some trained embedding spaces, not guaranteed properties of embeddings.

Do not hardcode expected semantic relationships as if they were learned.

## 22. Phase R — Reusable Visualization Infrastructure

Develop reusable visualizations for:
- Token sequences.
- Vocabulary tables.
- One-hot vectors.
- Embedding matrices.
- Embedding heatmaps.
- Token-to-vector mappings.
- Similarity matrices.
- Nearest-neighbor rankings.
- 2D and 3D vector spaces.
- Positional encoding.
- Embedding gradients.
- Before/after training comparisons.

Reuse existing visualization infrastructure when appropriate.

Keep mathematical and tensor computations outside Streamlit UI components.

Avoid unnecessary plotting-library dependencies.

## 23. Phase S — Quizzes and Guided Experiments

Add quizzes covering:
- Tokenization.
- Vocabulary.
- Token IDs.
- One-hot encoding.
- Embedding matrices.
- Embedding dimensions.
- Lookup semantics.
- Parameter counts.
- Gradient updates.
- Similarity algorithms.
- Normalization.
- Nearest neighbors.
- Word embeddings.
- Sentence pooling.
- Positional embeddings.
- Static versus contextual embeddings.
- Embedding limitations.

Include guided experiments such as:

1. Change vocabulary size and predict parameter count.
2. Change embedding dimension and inspect matrix shapes.
3. Compare one-hot and lookup computations.
4. Train a small embedding model and inspect vector changes.
5. Compare cosine similarity and dot product.
6. Normalize vectors and inspect ranking changes.
7. Explore 2D and higher-dimensional embeddings.
8. Compare sentences with reversed word order.
9. Add positional embeddings and inspect changed representations.
10. Compare initial and trained nearest neighbors.

Include expected observations and explanations.

Continue using the existing lesson-progress and quiz infrastructure.

Maintain M4–M7 progress JSON compatibility.

Do not reset historical scores, notes, or completed lessons.

## 24. Phase T — Testing and Regression

Add deterministic tests covering:

1. Vocabulary construction.
2. Tokenization.
3. Unknown-token handling.
4. Padding behavior.
5. Token-ID validation.
6. One-hot versus embedding equivalence.
7. Embedding output shapes.
8. Embedding parameter counts.
9. Repeated-token lookup.
10. Embedding gradients.
11. Padding-index gradient behavior.
12. SGD updates.
13. Adam updates.
14. Cosine similarity.
15. Dot-product similarity.
16. Euclidean distance.
17. Manhattan distance.
18. Zero-vector behavior.
19. Normalization.
20. Top-K nearest-neighbor ranking.
21. Skip-gram context generation.
22. Negative-sampling correctness.
23. Word-embedding training smoke tests.
24. Reproducibility.
25. Sentence pooling.
26. Padding-aware pooling.
27. Learned positional embeddings.
28. Sinusoidal positional encoding.
29. Odd/even embedding dimensions.
30. Architecture construction.
31. Framework-neutral specification serialization.
32. Previous model-specification compatibility.
33. Supervised embedding-classifier training.
34. Dataset and vocabulary leakage prevention.
35. Held-out evaluation.
36. Progress JSON backward compatibility.
37. Existing dense, CNN, RNN, and LSTM functionality.
38. Existing training and dataset management.
39. Existing ONNX functionality where applicable.
40. Streamlit UI interactions where feasible.

Use deterministic random seeds.

Separate numerical correctness tests from statistical model-quality tests.

Do not make model-quality claims from small training smoke tests.

Run targeted tests during development.

Run the full regression suite before M8 acceptance.

Document passed, failed, skipped, blocked, and untested checks.

Never claim visual/browser validation passed unless it was actually performed.

## 25. Phase U — Documentation Volume 6

Create:

**Documentation Volume 6 — Embeddings and Representation Learning**

Follow existing documentation naming conventions.

Include:

- Introduction to embeddings.
- Tokenization and vocabularies.
- One-hot versus dense representations.
- Embedding matrix mathematics.
- PyTorch Embedding API.
- Gradient updates.
- Word embedding learning.
- Skip-gram and negative sampling.
- Similarity algorithms.
- Vector normalization.
- Nearest-neighbor retrieval.
- Embedding-space visualization.
- PCA and t-SNE limitations.
- Sentence embeddings.
- Pooling strategies.
- Positional embeddings.
- Sinusoidal positional encodings.
- Static versus contextual embeddings.
- Downstream classification.
- Embedding evaluation.
- Limitations and risks.
- Detailed tutorials for every major M8 explorer.
- Guided experiments.
- Quizzes with explanations.
- Troubleshooting.
- Foundations for attention and Transformers.

Include worked mathematical examples and diagrams.

Update:
- README.
- `docs/ROADMAP.md`.
- Relevant ADRs.
- M8 milestone documents.

Preserve Documentation Volumes 1–5.

## 26. Phase V — Handoff and ChatGPT Synchronization

Maintain:

`docs/milestones/M8_HANDOFF.md`

Include:
- Current milestone status.
- Starting commit.
- Current branch and commit.
- Requirements/design references.
- Implemented features.
- Architecture decisions.
- Files changed.
- Tests executed.
- Passed/failed/skipped/untested checks.
- Known defects.
- Outstanding acceptance criteria.
- Highest-priority next task.
- Commands needed to resume.

Update the handoff at meaningful checkpoints, not after every small source edit.

At final implementation completion, update:

`docs/milestones/M8_COMPLETION_REPORT.md`

Record actual verification evidence rather than inferred success.

## 27. Phase W — Final Validation and Git Delivery

After implementation:

1. Run full automated regression.
2. Verify embedding mathematical correctness.
3. Execute embedding training smoke tests.
4. Verify all previous learning modules.
5. Run available Streamlit integration tests.
6. Perform actual browser workflows where available.
7. Check numerical stability and edge cases.
8. Review architecture compatibility.
9. Inspect dependency changes.
10. Check for secrets, caches, and generated datasets.
11. Update documentation and handoff.
12. Commit changes on `feature/m8-embedding-learning-studio`.
13. Push the feature branch.
14. Prepare a pull request against `master`.

Do not merge automatically.

Do not generate ZIP files.

Do not create duplicate project directories.

Do not overwrite user changes.

## 28. Final M8 Acceptance Criteria

M8 is complete only when:

- Embedding Network is integrated into Learning Studio.
- Tokenization and vocabulary explorers work.
- One-hot versus embedding demonstrations are numerically correct.
- Embedding matrix inspection works.
- Embedding lookup uses actual PyTorch.
- Embedding gradient updates can be inspected.
- Similarity algorithms work correctly.
- Nearest-neighbor retrieval is available.
- Skip-gram embedding learning works.
- Embedding-space visualization is available.
- Sentence pooling demonstrations work.
- Learned and sinusoidal positional representations are demonstrated.
- Static versus contextual embedding differences are explained correctly.
- Embedding architecture builder works.
- Downstream classification training is supported.
- Dataset leakage protections are verified.
- Previous neural-network functionality remains intact.
- M4–M7 progress JSON remains importable.
- Required tests pass.
- Documentation Volume 6 is complete.
- README and roadmap are updated.
- M8 handoff and completion reports are accurate.
- Source changes are committed on the feature branch.
- A pull request is prepared against master.

Any incomplete or unverified criterion must be clearly identified.

Do not declare independent or manual acceptance completed without actual evidence.

## 29. Long-Term Roadmap

Learning sequence:

1. Feed-Forward Neural Network — M4 completed.
2. Deep Neural Network — M4 completed.
3. CNN — M5 completed.
4. RNN — M6 completed.
5. LSTM — M7 completed.
6. Embedding Network — M8 current.
7. Single-Head Self-Attention — M9 planned.
8. Multi-Head Attention — M10 planned.
9. Transformer Encoder — M11 planned.
10. Transformer Decoder — M12 planned.
11. Mini GPT — M13 planned.

TensorFlow/Keras support remains deferred until the PyTorch learning modules are stable.

M8 must provide reusable embedding representations for M9–M13, but must not prematurely implement those later architectures.

## 30. Autonomous Execution Instructions

Proceed autonomously through the milestone.

Use this full specification as a reference while executing focused, independently verifiable implementation batches.

Do not stop after generating requirements or architecture documentation.

Do not ask for routine engineering approvals.

Prefer existing components and conventions.

Avoid prolonged analysis when implementation requirements are clear.

Avoid repeated source scans and unnecessary full regression runs.

Do not compromise correctness, compatibility, security, or educational quality to improve speed.

At completion, provide a concise report covering:

1. Starting Git commit.
2. Final Git commit.
3. Implemented M8 features.
4. Files changed.
5. Automated test results.
6. Streamlit/browser verification.
7. Documentation updates.
8. Remaining issues and limitations.
9. Pull request URL.
10. Final M8 acceptance status.

**Begin by reading AGENTS.md and the Operating Model, verifying the merged M7 baseline, and creating the M8 feature branch. Then implement M8 in focused, verifiable batches.**