# DeepLearningStudio — Milestone 6: Recurrent Neural Network (RNN) Learning Studio

## 1. Project Context

We are developing DeepLearningStudio, an interactive learning platform for neural networks, deep learning, and Generative AI.

**Repository:** https://github.com/swapnilpuri/deeplearningstudio

**Local workspace:**
`C:\Personal\Swapnil\Learning\Innovation\neural-network-studio`

**Current milestone:** M6 — Recurrent Neural Network (RNN) Learning Studio.

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

M1–M5 introduced neural-network design, dataset management, training, feed-forward networks, deep networks, and CNN learning capabilities.

M5 has been completed locally using Codex and is expected to form the M6 baseline.

Verify the actual repository state rather than relying on this description.

## 2. Mandatory Engineering Standards

Before implementation, read:

1. `AGENTS.md`
2. `docs/AI_DEVELOPMENT_OPERATING_MODEL.md`
3. `docs/ROADMAP.md`
4. Relevant M5 implementation and completion documentation.

These documents define the mandatory AI-assisted engineering workflow.

If `AGENTS.md` still contains generic project placeholders, customize it for DeepLearningStudio using the actual repository setup and test commands.

### Division of responsibilities

ChatGPT Web UI:
- Requirements and milestone architecture.
- Learning content design.
- Architecture reviews.
- Future roadmap planning.
- Independent milestone acceptance reviews.

Codex in VS Code:
- Source code implementation.
- Refactoring.
- Debugging.
- Automated testing.
- Local integration validation.
- Documentation maintenance.
- Git operations and PR preparation.

GitHub:
- Authoritative source of truth for committed code, decisions, tests, and milestone records.

Use milestone handoff documentation rather than relying on shared conversation history.

## 3. Codex Execution Efficiency — High Priority

The goal is to minimize unnecessary thinking and tool operations without compromising correctness.

Follow these rules:

1. Start with a concise execution plan of no more than seven steps.
2. Take the next safe engineering action promptly when requirements are clear.
3. Avoid prolonged speculative reasoning.
4. Do not repeatedly scan the complete repository.
5. Inspect only files relevant to the current task.
6. Reuse existing designs, abstractions, and utilities.
7. Avoid unnecessary architectural alternatives or rewrites.
8. Divide M6 into small, independently testable implementation batches.
9. Run targeted tests after each batch.
10. Run the full regression suite at major checkpoints and final acceptance.
11. Do not repeatedly execute unchanged expensive tests.
12. Batch independent read-only operations where safe.
13. Never parallelize conflicting file modifications or Git writes.
14. Keep console output and progress reports concise.
15. Document significant decisions once rather than repeating them.
16. Update the milestone handoff at meaningful checkpoints.
17. Continue autonomously without requesting approval for routine decisions.
18. Stop for genuinely destructive, security-sensitive, or irreversible operations requiring authorization.
19. Do not compromise testing, security, educational accuracy, or compatibility for speed.
20. Do not spend excessive time redesigning established architecture without a demonstrated requirement.

Use the currently selected Codex model and reasoning effort. Do not assume prompt instructions can directly change model inference speed.

**Execution priority: Implement the next concrete, independently verifiable requirement rather than repeatedly reconsidering the complete milestone.**

## 4. M6 Objective

Implement an interactive Recurrent Neural Network Learning Studio enabling students to understand how recurrent networks process sequential data, maintain hidden states, learn temporal relationships, and train using Backpropagation Through Time.

The implementation must:

- Extend the existing Learning Studio.
- Preserve all M1–M5 workspaces and functionality.
- Use real PyTorch computations.
- Provide step-by-step mathematical explanations.
- Visualize hidden states and gradients across time.
- Enable architecture experimentation.
- Support sequence classification and sequence forecasting.
- Provide quizzes and guided learning exercises.
- Remain extensible for LSTM, GRU, attention, and Transformer milestones.

Do not implement complete LSTM, GRU, or Transformer lessons in M6.

## 5. Phase A — Repository Baseline and Architecture

Before modifying files:

1. Inspect the current Git branch, working tree, and latest commit.
2. Protect existing uncommitted work.
3. Verify the M5 completion state.
4. Review only relevant Learning Studio, model specification, PyTorch adapter, training, visualization, progress, and test modules.
5. Execute baseline automated tests.
6. Identify existing functionality suitable for reuse.
7. Create `feature/m6-rnn-learning-studio` from the verified baseline when safe.

Create:

- `docs/milestones/M6_REQUIREMENTS.md`
- `docs/milestones/M6_DESIGN.md`
- `docs/milestones/M6_ACCEPTANCE_CRITERIA.md`
- `docs/milestones/M6_HANDOFF.md`

Use this prompt as the initial requirements source.

The design should explain:
- Framework-neutral recurrent specifications.
- PyTorch adapter implementation.
- Recurrent cell execution and tracing.
- Shape inference.
- Sequence-data representation.
- Training integration.
- Reusable visualization components.
- Progress compatibility.
- Validation and testing strategy.

Make reasonable architecture decisions and document consequential ones in ADRs.

Do not stop after writing the design documents. Continue implementation.

## 6. Phase B — RNN Learning Module

Add a new lesson:

**Recurrent Neural Network (RNN)**

Preserve the existing Feed-Forward, Deep Neural Network, and CNN lessons.

Cover:

1. Sequential versus independent data.
2. Time steps and sequence length.
3. Input feature vectors.
4. Hidden state.
5. Recurrent connections.
6. Shared weights across time.
7. Initial hidden state.
8. RNN cells.
9. Sequence processing.
10. Many-to-one architectures.
11. Many-to-many architectures.
12. Single-layer versus stacked RNNs.
13. Tanh and ReLU activations.
14. Sequence outputs.
15. Loss functions.
16. Backpropagation Through Time.
17. Vanishing gradients.
18. Exploding gradients.
19. Gradient clipping.
20. Truncated BPTT.
21. RNN limitations for long sequences.
22. Motivation for LSTM, GRU, and attention.

For each major concept, provide:

- Beginner-friendly explanation.
- Mathematical formula.
- Tensor shape walkthrough.
- Practical use case.
- Interactive PyTorch demonstration.
- Guided experiment.

Include an explanation of why RNNs differ from CNNs and dense networks.

## 7. Phase C — Interactive RNN Cell Explorer

Implement a dedicated RNN cell explorer.

The fundamental recurrence is:

`h_t = tanh(W_ih × x_t + b_ih + W_hh × h_(t-1) + b_hh)`

For each time step display:

- Current input vector.
- Previous hidden state.
- Input-to-hidden weight matrix.
- Hidden-to-hidden weight matrix.
- Input contribution.
- Recurrent contribution.
- Bias contributions.
- Pre-activation value.
- Activation result.
- New hidden state.

Support configurable:

- Input size.
- Hidden size.
- Sequence length.
- Initial hidden state.
- Activation function.
- Weight initialization or editable demonstration weights.
- Input sequence.
- Selected time step.

Use real PyTorch computations.

Show numerical calculations side by side with the corresponding tensor operations.

Allow stepping forward and backward through the sequence visualization.

Explain that the parameters are shared across time steps.

Use `torch.nn.RNNCell` or mathematically equivalent PyTorch operations, and verify agreement with the reference implementation.

## 8. Phase D — RNN Unrolling and Sequence Visualization

Create an interactive unrolled RNN diagram:

`x1 → h1 → h2 → h3 → ... → hT`

Each time step should display its input, hidden state, and output where applicable.

Highlight:
- Current time step.
- Current input.
- Previous state.
- Current state.
- Recurrent dependencies.
- Shared weights.

Allow the learner to adjust sequence length.

Provide visualizations of hidden-state values across time.

Demonstrate why reversing or changing the order of input elements can change the output.

Explain the difference between recurrence and independent processing.

## 9. Phase E — Sequence Tensor Inspection

Explain PyTorch sequence shapes.

For batch-first input:

`[batch_size, sequence_length, input_size]`

For a basic unidirectional RNN:

`output: [batch_size, sequence_length, hidden_size]`

`h_n: [num_layers, batch_size, hidden_size]`

Explain how shapes differ when `batch_first=False`.

Include configurable:

- Batch size.
- Sequence length.
- Feature count.
- Hidden-state dimensions.
- Number of layers.

Show tensor shapes at every stage.

Explain the distinction between:
- Full output sequence.
- Final hidden state.
- Final valid hidden state when sequences are padded.
- Classification logits.

Demonstrate PyTorch `nn.RNN` execution using actual input tensors.

## 10. Phase F — Recurrent Architecture Builder

Extend the framework-independent architecture system to support recurrent layers.

Support at minimum:

- Sequence input.
- RNN layer.
- Supported recurrent activation.
- Optional stacked RNN layers.
- Final-state selection or sequence output.
- Linear classification/regression heads.

Provide:
- Architecture diagrams.
- Layer configuration.
- Tensor shapes.
- Weight shapes.
- Parameter counts.
- Forward tracing.
- Valid configuration checks.

Do not hardcode tensor dimensions.

For a standard single-direction RNN with biases enabled, explain the parameter-count implications of the input and recurrent matrices and the two bias vectors per layer in PyTorch.

Keep recurrent specifications framework-neutral.

Do not introduce PyTorch imports into framework-independent specification modules.

Preserve compatibility with M4 dense specifications and M5 CNN specifications.

Use explicit schema versions where necessary.

## 11. Phase G — Sequence Dataset Laboratory

Create or reuse sequence datasets that work offline.

At minimum support:

### Dataset 1 — Synthetic sequence classification

Provide a task where the target depends on sequence content or order.

Examples:
- Compare the first and last elements.
- Detect whether a sequence contains a predefined pattern.
- Classify a simple temporal pattern.

Ensure the task requires sequence reasoning rather than only independent feature classification.

### Dataset 2 — Sine-wave forecasting

Generate synthetic sine-wave sequences for next-step prediction.

Allow configuration of:
- Sequence length.
- Noise.
- Number of samples.
- Random seed.
- Forecast horizon where appropriate.

### Dataset 3 — Optional sequence-memory task

Demonstrate how increasing the distance between relevant sequence elements affects learning.

Use a controlled deterministic setup.

For time-series evaluation, use chronological or disjoint source-series splits where appropriate, prevent overlap/leakage across train/validation/test boundaries, and fit preprocessing using training data only.

For independent synthetic sequence tasks, reproducible disjoint sampling is acceptable.

## 12. Phase H — RNN Training Laboratory

Implement actual PyTorch RNN training.

Support:

- Sequence classification.
- Next-step sequence regression.
- Configurable epochs.
- Batch size.
- Learning rate.
- Hidden size.
- Number of RNN layers.
- SGD optimizer.
- Adam optimizer.
- Gradient clipping.
- Train/validation/test splits.

Use:

- CrossEntropyLoss for classification.
- MSELoss for numerical forecasting.

Display:
- Training loss.
- Validation loss.
- Training metrics.
- Validation metrics.
- Final held-out test metrics.
- Example predictions.
- Incorrect predictions.
- Hidden-state progression.
- Gradient norms.
- Parameter changes.

Reuse the existing Training Studio infrastructure when practical, but avoid destabilizing existing CNN and dense training.

No mandatory external dataset downloads.

## 13. Phase I — Backpropagation Through Time Explorer

Implement an interactive demonstration of BPTT.

Explain:

- Unrolling the recurrent network.
- Computing loss.
- Applying the chain rule through time.
- Accumulating gradients for shared weights.
- Running `loss.backward()`.
- Inspecting gradients.
- Applying optimizer updates.

Visualize gradient magnitudes across time steps.

Demonstrate vanishing and exploding gradients in carefully controlled examples.

Include an interactive gradient-clipping demonstration showing:

- Norm before clipping.
- Clipping threshold.
- Norm after clipping.
- Parameter update.

Explain that clipping limits gradient magnitude but does not solve vanishing gradients.

Include a truncated-BPTT demonstration showing how detaching hidden states limits gradient propagation.

Be explicit about truncation boundaries and what gradients can and cannot reach.

Optional numerical gradient checking may be included if it remains fast and deterministic.

## 14. Phase J — Variable-Length Sequences

Provide a beginner-level demonstration of variable-length sequences.

Cover:
- Padding.
- Valid sequence lengths.
- Masks where relevant.
- Packed sequences.
- Selecting the final valid state.

Use PyTorch utilities such as `pack_padded_sequence` where appropriate.

Ensure learners understand why using a padded final time step as a classification representation can be incorrect.

Do not let padded positions contaminate supervised losses.

Keep this functionality lightweight and focused on supporting future sequence models.

## 15. Phase K — Reusable Visualization Infrastructure

Build reusable visualizations for:

- Time-step sequences.
- Recurrent state transitions.
- Hidden-state matrices.
- Input/recurrent weight matrices.
- Gradient flow through time.
- Prediction versus target.
- Sequence classifications.
- Tensor shapes.

Separate computation and presentation.

Avoid hardcoding RNN mathematics inside Streamlit UI components.

Prefer components that can later support LSTM and attention lessons.

Do not over-engineer speculative Transformer features.

## 16. Phase L — Quizzes and Guided Experiments

Add quizzes covering:

- Sequential data.
- Hidden states.
- Recurrent weights.
- Parameter sharing.
- Tensor shapes.
- Sequence length.
- Many-to-one and many-to-many outputs.
- BPTT.
- Vanishing and exploding gradients.
- Gradient clipping.
- RNN limitations.

Create guided experiments such as:

1. Increase sequence length and inspect hidden states.
2. Change hidden-state size and predict parameter counts.
3. Compare Tanh and ReLU recurrence.
4. Change input order and observe output differences.
5. Compare short and long dependency tasks.
6. Enable gradient clipping and inspect norm changes.
7. Compare sequence classification and forecasting.

Record quiz scores and lesson progress through the existing progress-tracking infrastructure.

Maintain compatibility with M4 and M5 progress JSON files.

## 17. Phase M — Quality and Regression Testing

Add automated tests for:

1. RNN cell numerical correctness.
2. Agreement with `torch.nn.RNNCell`.
3. Agreement with `torch.nn.RNN` for supported configurations.
4. Hidden-state recurrence calculations.
5. Sequence tensor shapes.
6. Single-layer and stacked RNNs.
7. Parameter-count calculations.
8. Framework-neutral specification serialization.
9. Existing dense and CNN specification compatibility.
10. Sequence classification forward pass.
11. Sequence regression forward pass.
12. BPTT gradient propagation.
13. Gradient clipping.
14. Truncated BPTT.
15. Sequence dataset reproducibility.
16. Split isolation and preprocessing leakage prevention.
17. Variable-length sequence handling.
18. RNN training smoke tests.
19. Progress JSON backward compatibility.
20. Existing M4/M5 learning functionality.
21. Existing CNN training and visualization functionality.
22. Existing Neural Network Studio, Dataset Management, and Training Studio functionality.
23. ONNX-related paths where practical.

Use deterministic random seeds.

Separate passed, failed, skipped, blocked, and untested checks.

Do not claim Streamlit GUI, Windows, or ONNX validation passed without executing the respective checks.

Perform targeted tests during implementation and full regression before M6 acceptance.

## 18. Phase N — Documentation Volume 4

Create:

**Documentation Volume 4 — Recurrent Neural Networks**

Use Markdown and preserve the existing documentation organization.

Include:

- Introduction to sequential learning.
- RNN architecture.
- Mathematical recurrence.
- Step-by-step numerical examples.
- Unrolled RNN diagrams.
- Hidden-state explanations.
- Tensor-shape walkthroughs.
- Sequence classification.
- Sequence forecasting.
- BPTT derivation and examples.
- Vanishing/exploding gradients.
- Gradient clipping.
- Truncated BPTT.
- Variable-length sequence handling.
- PyTorch implementation explanations.
- Platform tutorials.
- Guided experiments and expected observations.
- Quiz questions with answers and explanations.
- Troubleshooting.
- Dense network versus CNN versus RNN comparison.
- Motivation for the future LSTM milestone.

Update the README and `docs/ROADMAP.md`.

Create a concise architecture decision record for recurrent specifications.

## 19. Phase O — Milestone Handoff and Synchronization

Maintain:

`docs/milestones/M6_HANDOFF.md`

Include:
- Milestone status.
- Current Git branch and commit SHA.
- Requirements and design references.
- Completed features.
- Architecture decisions.
- Files changed.
- Exact test results.
- Known issues.
- Unverified functionality.
- Next highest-priority task.
- Commands needed to resume.

Update the handoff after meaningful batches, not after every small edit.

Ensure future ChatGPT and Codex sessions can resume using repository documentation alone.

## 20. Phase P — Final Validation and Git Delivery

At milestone completion:

1. Run full automated regression.
2. Validate CNN, dense, and RNN functionality.
3. Perform applicable application smoke tests.
4. Review the final diff for correctness.
5. Check for generated files, caches, large datasets, or secrets.
6. Update `M6_COMPLETION_REPORT.md`.
7. Record baseline and final test results.
8. Commit completed changes on the M6 feature branch.
9. Push the feature branch.
10. Prepare a pull request targeting `master`.

Do not merge automatically.

Do not create ZIP files or duplicate project folders.

All development must happen directly in the existing VS Code workspace.

## 21. Final M6 Acceptance Criteria

M6 is complete only when:

- RNN lesson is integrated into Learning Studio.
- RNN cell explorer uses real PyTorch operations.
- Hidden states can be inspected across time.
- Interactive unrolling works.
- RNN tensor shapes and parameter counts are correct.
- Configurable recurrent architectures can be built.
- Sequence classification training works.
- Sine-wave forecasting training works.
- BPTT gradients can be inspected.
- Vanishing and exploding gradients are explained and demonstrated.
- Gradient clipping is functional.
- Variable-length sequence concepts are demonstrated.
- Quizzes and progress tracking work.
- M4/M5 progress JSON remains compatible.
- Existing functionality is preserved.
- Required automated tests pass.
- Documentation Volume 4 is complete.
- Roadmap and milestone reports are updated.
- Git changes are committed to the feature branch.
- A pull request is prepared against master.

If any mandatory requirement remains incomplete or unverified, record that explicitly.

## 22. Long-Term Roadmap

Current learning sequence:

1. Feed-Forward Neural Network — Completed.
2. Deep Neural Network — Completed.
3. CNN — M5 completed locally; verify repository baseline.
4. RNN — M6 current.
5. LSTM — Future milestone.
6. Embedding Network.
7. Single-Head Self-Attention.
8. Multi-Head Attention.
9. Transformer Encoder.
10. Transformer Decoder.
11. Mini GPT.

TensorFlow/Keras will be introduced after the PyTorch learning modules are sufficiently stable.

Maintain architectural extensibility without prematurely implementing future lessons.

## 23. Autonomous Execution

Proceed through M6 autonomously.

Use the full requirements as a reference, but execute in focused, verifiable batches.

Do not request routine approvals.

Do not stop at architecture or planning if implementation can safely continue.

Avoid redundant repository scans, overly detailed status messages, and repeated expensive testing.

Preserve working code, tests, data, and project conventions.

Prioritize correctness, educational quality, maintainability, backward compatibility, and efficient execution.

At completion, provide a concise report containing:

1. Starting commit.
2. Final commit.
3. Features implemented.
4. Tests executed and results.
5. UI and integration checks.
6. Documentation created or updated.
7. Remaining limitations.
8. Pull request URL if created.
9. Final M6 acceptance status.

**Begin by reading AGENTS.md and the Operating Model, inspecting the local M5 baseline, and creating the M6 feature branch. Then implement M6 in focused batches.**
