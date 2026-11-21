# DeepLearningStudio — Milestone 7: LSTM Learning Studio

## 1. Project Context

We are developing DeepLearningStudio, an interactive educational platform for learning neural networks, deep learning, and Generative AI through mathematical explanations, real model execution, visualizations, guided experiments, and quizzes.

**Repository:** https://github.com/swapnilpuri/deeplearningstudio

**Local workspace:**
`C:\Personal\Swapnil\Learning\Innovation\neural-network-studio`

**Current milestone:** M7 — Long Short-Term Memory (LSTM) Learning Studio.

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

### Existing capabilities

M1–M3:
- Neural Network Studio.
- Dataset Management.
- Training Studio.

M4:
- Feed-Forward Neural Networks.
- Deep Neural Networks.
- Mathematical demonstrations.
- Learning progress and quizzes.

M5:
- CNN Learning Studio.
- Convolution and pooling explorers.
- CNN architecture builder.
- Digits classification.
- Convolution backpropagation.

M6:
- RNN Learning Studio.
- Recurrent cell explorer.
- Hidden-state visualization.
- RNN unrolling.
- Sequence classification and forecasting.
- Backpropagation Through Time.
- Gradient clipping.
- Variable-length sequence demonstrations.

M6 was merged into master.

The latest GitHub master commit observed when preparing this prompt was:

`fcea18305f2112a2c34f2bcc8e48117f13d1d428`

Verify the latest local and remote commit before beginning. Do not assume this SHA is still current.

The M6 completion report records a final automated regression of 135 passed tests. Manual browser visual acceptance remained incomplete.

M7 must preserve all previous functionality.

## 2. Mandatory Engineering Operating Model

Read and follow:

1. `AGENTS.md`
2. `docs/AI_DEVELOPMENT_OPERATING_MODEL.md`
3. `docs/ROADMAP.md`
4. `docs/milestones/M6_HANDOFF.md`
5. `docs/milestones/M6_COMPLETION_REPORT.md`
6. The relevant M6 requirements, design, tests, and recurrent architecture ADR.

Both engineering-standard files already exist in the repository. Do not recreate or replace them unnecessarily.

### Responsibilities

ChatGPT Web UI:
- Requirements and architecture.
- Educational design.
- Milestone planning.
- Independent architecture and completion reviews.

Codex:
- Repository inspection.
- Implementation and refactoring.
- Automated testing.
- Local execution.
- Defect resolution.
- Documentation updates.
- Git and pull-request preparation.

GitHub:
- Source of truth for code, documentation, decisions, and verification evidence.

Maintain milestone handoffs so that future ChatGPT and Codex sessions can resume without importing entire conversation histories.

## 3. Codex Execution Efficiency — Mandatory

Optimize for efficient development without sacrificing correctness.

1. Begin with a concise plan of no more than seven steps.
2. Inspect relevant files rather than repeatedly scanning the entire repository.
3. Reuse existing M6 recurrent components and conventions.
4. Avoid unnecessary architectural redesign.
5. Make routine engineering decisions autonomously.
6. Implement in small, independently testable batches.
7. Run targeted tests after each batch.
8. Run full regression at significant integration checkpoints and final acceptance.
9. Avoid repeated expensive test runs without source changes or specific justification.
10. Batch independent read-only operations when safe.
11. Do not parallelize conflicting writes or Git operations.
12. Avoid repeatedly reopening unchanged files.
13. Keep reasoning and progress reports concise.
14. Maintain documentation at meaningful checkpoints.
15. Do not generate duplicate project folders, ZIP files, or unnecessary scaffolding.
16. Avoid introducing dependencies unless justified.
17. Prefer the simplest correct solution compatible with the established architecture.
18. Continue implementation without unnecessary approval prompts.
19. Respect authorization boundaries for destructive or irreversible changes.
20. Never compromise correctness, security, backward compatibility, educational accuracy, or required validation for speed.

Treat this complete milestone specification as a reference. Execute the work in focused batches rather than continuously reconsidering the entire specification.

Prompt instructions cannot directly control inference latency or change the selected model's reasoning effort.

**Priority: Take the next safe, concrete engineering action promptly whenever requirements are clear.**

## 4. M7 Objective

Implement the LSTM Learning Studio as the next lesson in the existing Learning Studio.

Learners must be able to:

- Understand why vanilla RNNs struggle with long-range dependencies.
- Understand LSTM hidden states and cell states.
- Explore forget, input, and output gates.
- Understand candidate cell memory.
- Execute actual PyTorch LSTM calculations.
- Inspect how information flows across time steps.
- Construct LSTM architectures.
- Train LSTMs on sequence tasks.
- Compare LSTM and vanilla RNN behavior.
- Inspect gradients and parameter updates.
- Understand backpropagation through LSTM gates.
- Visualize memory retention and forgetting.
- Complete quizzes and guided experiments.

M7 must extend the existing platform rather than replace it.

GRU internals, embeddings, attention, and Transformers are outside the M7 implementation scope.

## 5. Phase A — Baseline Inspection

Before modifying source code:

1. Confirm the correct local Git repository.
2. Inspect the current branch, working-tree status, and remote.
3. Protect uncommitted user work.
4. Verify M6 implementation and merged Git history.
5. Review existing RNN architecture, adapters, visualizations, training infrastructure, and tests.
6. Run baseline regression tests.
7. Record the baseline commit and test results.

Create:

`feature/m7-lstm-learning-studio`

Use the latest verified master baseline when safe.

Create the following documents:

- `docs/milestones/M7_REQUIREMENTS.md`
- `docs/milestones/M7_DESIGN.md`
- `docs/milestones/M7_ACCEPTANCE_CRITERIA.md`
- `docs/milestones/M7_HANDOFF.md`
- `docs/milestones/M7_COMPLETION_REPORT.md`

Do not stop after the design phase.

Proceed with implementation.

## 6. Phase B — LSTM Learning Module

Add a new Learning Studio lesson:

**Long Short-Term Memory (LSTM)**

Preserve:
- Feed-Forward Neural Network.
- Deep Neural Network.
- CNN.
- RNN.

Teach:

1. Recap of vanilla RNNs.
2. Why long-term dependencies are difficult.
3. Vanishing and exploding gradients.
4. Motivation for gated memory.
5. Hidden state versus cell state.
6. Forget gate.
7. Input gate.
8. Candidate cell memory.
9. Cell-state update.
10. Output gate.
11. Hidden-state update.
12. Information retention over time.
13. Information forgetting.
14. Shared parameters across time.
15. Single-layer LSTM.
16. Stacked LSTM.
17. Sequence-to-one prediction.
18. Sequence-to-sequence output.
19. LSTM parameter counts.
20. LSTM backpropagation.
21. Sequence classification.
22. Time-series forecasting.
23. LSTM limitations.
24. Motivation for attention mechanisms.

For each major concept provide:
- Beginner-friendly explanation.
- Mathematical formula.
- Tensor dimensions.
- Numerical example.
- Interactive demonstration.
- Practical use case.

## 7. Phase C — Interactive LSTM Cell Explorer

Build a dedicated LSTM cell explorer.

Use the standard equations:

Forget gate:

`f_t = sigmoid(W_if x_t + b_if + W_hf h_(t-1) + b_hf)`

Input gate:

`i_t = sigmoid(W_ii x_t + b_ii + W_hi h_(t-1) + b_hi)`

Candidate memory:

`g_t = tanh(W_ig x_t + b_ig + W_hg h_(t-1) + b_hg)`

Output gate:

`o_t = sigmoid(W_io x_t + b_io + W_ho h_(t-1) + b_ho)`

Cell-state update:

`c_t = f_t ⊙ c_(t-1) + i_t ⊙ g_t`

Hidden-state update:

`h_t = o_t ⊙ tanh(c_t)`

Explain that ⊙ is element-wise multiplication.

Use actual PyTorch tensor operations.

Allow configuration of:
- Input feature count.
- Hidden size.
- Sequence length.
- Initial hidden state.
- Initial cell state.
- Input values.
- Demonstration weights.
- Bias values.
- Current time step.

Display:
- Input vector.
- Previous hidden state.
- Previous cell state.
- Gate weight matrices.
- Linear contributions.
- Sigmoid and Tanh transformations.
- Forget-gate values.
- Input-gate values.
- Candidate-memory values.
- Output-gate values.
- Updated cell state.
- Updated hidden state.

Support navigation through time steps.

For selected units, display the complete numerical calculation.

Verify outputs against `torch.nn.LSTMCell` for supported configurations.

Explain PyTorch's gate ordering and packed weight matrices.

Avoid duplicating mathematics inside Streamlit UI components.

## 8. Phase D — Gate Visualization Laboratory

Create interactive visualizations for all LSTM gates.

Display gate values using:
- Heatmaps.
- Time-step charts.
- State matrices.
- Selected-gate comparisons.

Include controlled demonstrations of:

### Forgetting

Show how forget-gate values near zero suppress contributions from previous cell memory.

### Memory retention

Show how forget-gate values near one preserve more of the previous cell state.

### Writing new memory

Show how the input gate controls the candidate-memory contribution.

### Exposing memory

Show how the output gate controls the hidden-state representation derived from the updated cell state.

Explain that the gates are learned functions of inputs and previous hidden states, rather than fixed switches.

Allow learners to modify relevant values and observe the resulting state.

Clearly distinguish controlled gate experiments from normal learned behavior.

Keep all visualizations synchronized with actual computed tensors.

## 9. Phase E — Cell State and Hidden State Explorer

Show how `c_t` and `h_t` evolve across a complete sequence.

Provide side-by-side plots for:
- Input sequence.
- Forget gate.
- Input gate.
- Candidate memory.
- Cell state.
- Output gate.
- Hidden state.

Explain:

- Cell state acts as an internal memory pathway.
- Hidden state is the exposed recurrent representation.
- Both states are propagated between time steps.
- Their values are not generally equal.
- Cell states can accumulate information, but retention is not guaranteed.
- LSTMs mitigate, rather than eliminate, long-range gradient problems.

Use a deterministic educational sequence with a known interpretation.

Allow the learner to change sequence length and input order.

## 10. Phase F — Interactive LSTM Architecture Builder

Extend the existing framework-independent recurrent architecture.

Support:
- LSTM input.
- Single-layer LSTM.
- Stacked LSTM.
- Configurable hidden size.
- Configurable layer count.
- Batch-first and time-first layouts.
- Optional dropout between stacked layers.
- Final-state classification.
- Sequence-output prediction.
- Linear classification/regression heads.

Display:
- Architecture diagram.
- Input tensor shapes.
- Output tensor shapes.
- Hidden-state shapes.
- Cell-state shapes.
- Parameter counts.
- Parameter tensor shapes.
- Layer-by-layer tensor inspection.

For a unidirectional LSTM with one layer, demonstrate:

Input:

`[batch_size, sequence_length, input_size]`

Output:

`[batch_size, sequence_length, hidden_size]`

Final hidden state:

`[num_layers, batch_size, hidden_size]`

Final cell state:

`[num_layers, batch_size, hidden_size]`

Explain the distinction between the full output sequence and final states.

For stacked LSTMs, correctly propagate dimensions between layers.

Infer classification-head dimensions automatically.

Validate invalid configurations gracefully.

## 11. Phase G — LSTM Mathematics and Parameter Inspection

Explain the four LSTM gate transformations and their trainable parameters.

For a single-layer, unidirectional PyTorch LSTM with biases enabled, derive:

`Parameters = 4H × I + 4H × H + 8H`

where:
- I = input feature count.
- H = hidden size.

Explain the two groups of bias vectors used by PyTorch.

For subsequent stacked layers, use the preceding layer's output width as the input dimension.

Explain PyTorch weight shapes:

`weight_ih_l0: [4H, I]`

`weight_hh_l0: [4H, H]`

`bias_ih_l0: [4H]`

`bias_hh_l0: [4H]`

Include numerical parameter-count examples.

Show how each weight tensor maps to the gates.

Use explicit mapping consistent with PyTorch's gate ordering.

Do not assume that LSTM parameter counts are identical to vanilla RNN parameter counts.

## 12. Phase H — Framework-Independent LSTM Specification

Extend the existing recurrent specification architecture without breaking previous models.

Requirements:

1. Preserve dense model specification compatibility.
2. Preserve CNN specification compatibility.
3. Preserve vanilla RNN specification compatibility.
4. Preserve existing serialized progress files.
5. Introduce a versioned LSTM specification or a backward-compatible extension.
6. Keep specifications independent of PyTorch.
7. Keep actual PyTorch construction in adapters.
8. Centralize shape inference and parameter validation.
9. Support JSON serialization and round trips.
10. Design for future GRU and sequence-model backends without implementing them prematurely.

Reuse M6 recurrent abstractions when they genuinely fit.

Do not introduce a large general-purpose neural-network compiler.

Create an ADR documenting the chosen specification strategy and compatibility guarantees.

## 13. Phase I — LSTM Sequence Training Laboratory

Implement LSTM training using existing infrastructure where practical.

At minimum support two offline tasks.

### Task 1 — Long-range memory classification

Design a controlled synthetic dataset where a label depends on information from an earlier time step.

For example, the first element carries a signal that must be remembered across later distractor elements.

Allow configuration of:
- Sequence length.
- Dependency distance.
- Hidden size.
- Number of samples.
- Noise.
- Random seed.

Ensure the task is not trivially solvable through label leakage or unrelated shortcuts.

### Task 2 — Time-series forecasting

Use a synthetic sine-wave or related temporal signal.

Support:
- Historical input window.
- Next-step prediction.
- Configurable sequence length.
- Forecast horizon where practical.
- Noise.
- Random seed.

For time-series data:
- Prevent overlapping source windows from leaking between splits.
- Use chronological or suitably isolated source-series splits.
- Fit preprocessing using training data only.

No mandatory external dataset downloads.

### Training controls

Provide:
- SGD.
- Adam.
- CrossEntropyLoss for classification.
- MSELoss for regression.
- Configurable epochs.
- Batch size.
- Learning rate.
- Hidden size.
- Number of layers.
- Gradient clipping.
- Reproducible initialization.

Display:
- Training loss.
- Validation loss.
- Classification accuracy or forecasting error.
- Example predictions.
- Incorrect classifications.
- Gradient norms.
- Model parameter counts.
- Training configuration.

Keep final test evaluation held out until explicitly requested.

## 14. Phase J — RNN versus LSTM Comparison Laboratory

Create a controlled comparison between vanilla RNN and LSTM.

Reuse M6 RNN implementations.

Allow learners to compare:
- Same dataset.
- Same train/validation/test split.
- Same sequence length.
- Same training budget.
- Same optimizer where appropriate.
- Same learning rate where appropriate.
- Comparable hidden sizes.

Display:
- Training loss curves.
- Validation loss curves.
- Test metrics after explicit evaluation.
- Gradient norms.
- Parameter counts.
- Training time.
- Predictions.
- Performance at different dependency lengths.

Use transparent, reproducible experimental settings.

Explain that an LSTM often helps with long-range dependencies but is not guaranteed to outperform an RNN in every experiment.

Do not fabricate experimental results or claim superiority without measured evidence.

Provide a teaching explanation of the additional parameters and computational cost of LSTMs.

## 15. Phase K — LSTM Backpropagation Explorer

Demonstrate:

1. Forward propagation.
2. LSTM gate computations.
3. Cell-state updates.
4. Hidden-state updates.
5. Prediction logits or regression outputs.
6. Loss calculation.
7. `optimizer.zero_grad()`.
8. `loss.backward()`.
9. Gate-related parameter gradients.
10. Optimizer update.
11. New parameter values.
12. Loss after one update.

Show gradient flow across time steps.

Explain how gradients flow through both cell-state and hidden-state paths.

Visualize gradient norms for:
- Input-to-hidden weights.
- Hidden-to-hidden weights.
- Individual gate parameter slices where appropriate.
- Earlier time-step states.

Retain gradients on intermediate states when needed for educational inspection.

Include SGD update verification.

For Adam, explain the adaptive update rather than incorrectly displaying the SGD equation as its actual update.

## 16. Phase L — Vanishing Gradients and Memory Retention

Create controlled experiments demonstrating how gating affects gradient propagation.

Include:
- Short versus long sequences.
- Forget gate near zero.
- Forget gate near one.
- Input gate behavior.
- Different initializations.
- Gradient magnitude by time step.

Explain the multiplicative effect of forget gates on the direct cell-state gradient path.

Clarify that other paths through the network also contribute to the full gradient.

Demonstrate why long-term memory remains difficult even for LSTMs.

Include gradient clipping, showing norm before and after clipping.

Explain that clipping primarily addresses exploding gradients, not vanishing gradients.

## 17. Phase M — Variable-Length Sequence Support

Reuse existing M6 variable-length sequence capabilities.

Cover:
- Padding.
- Packed sequences.
- Sequence lengths.
- Final valid states.
- Masks.
- Batch processing.

Support correct LSTM processing of variable-length sequences where practical.

Ensure padded positions do not incorrectly affect loss calculations.

Show both hidden-state and cell-state dimensions for packed-sequence execution.

Distinguish final valid states from the last padded position.

Do not regress existing M6 sequence-packing functionality.

## 18. Phase N — Reusable Visualization Infrastructure

Implement or extend reusable visualizations for:

- LSTM gate diagrams.
- Recurrent sequence unrolling.
- Cell-state evolution.
- Hidden-state evolution.
- Gate activation heatmaps.
- Gradient flow.
- Parameter matrices.
- Sequence predictions.
- Model comparisons.

Reuse existing plotting components whenever suitable.

Keep computation separate from Streamlit presentation.

Avoid speculative implementations of attention or Transformer features.

## 19. Phase O — Quizzes and Guided Experiments

Add LSTM quizzes covering:

1. RNN versus LSTM.
2. Hidden state versus cell state.
3. Forget gate.
4. Input gate.
5. Output gate.
6. Candidate memory.
7. Sigmoid.
8. Tanh.
9. LSTM parameter counts.
10. Tensor shapes.
11. Sequence learning.
12. Gradient flow.
13. Memory retention.
14. Long-range dependencies.
15. LSTM limitations.

Add guided experiments such as:

- Set the forget gate near zero and inspect memory retention.
- Set the forget gate near one and compare cell states.
- Change the input gate and observe candidate-memory contributions.
- Change sequence length and inspect gradients.
- Compare RNN and LSTM on a long-range memory task.
- Increase hidden size and predict parameter counts.
- Compare SGD and Adam.
- Explore stacked LSTM tensor shapes.

Include expected observations and explanations.

Continue to use the existing learning progress infrastructure.

Preserve imported progress and quiz scores from M4, M5, and M6.

## 20. Phase P — Testing and Regression

Add deterministic tests covering:

1. LSTM cell numerical correctness.
2. Agreement with `torch.nn.LSTMCell`.
3. Agreement with `torch.nn.LSTM`.
4. Forget/input/output gate calculations.
5. Cell-state updates.
6. Hidden-state updates.
7. Gate ordering.
8. Bias behavior.
9. Parameter counts.
10. Single-layer tensor shapes.
11. Stacked LSTM tensor shapes.
12. Batch-first and time-first inputs.
13. Architecture construction.
14. Model specification serialization.
15. Backward compatibility with M4–M6 specifications.
16. Sequence classification forward propagation.
17. Sequence regression forward propagation.
18. BPTT gradient correctness.
19. Optimizer updates.
20. Gradient clipping.
21. Memory retention examples.
22. Synthetic dataset reproducibility.
23. Dataset split isolation.
24. Leakage prevention.
25. Variable-length sequences.
26. Packed-sequence processing.
27. Training smoke tests.
28. RNN versus LSTM comparison consistency.
29. Progress JSON backward compatibility.
30. Existing CNN/RNN learning functionality.
31. Existing dense-model functionality.
32. Existing dataset and training functionality.
33. Existing ONNX code paths where applicable.
34. Relevant Streamlit UI interactions.

Use numerical tolerances appropriate to floating-point computation.

Do not require model-quality thresholds that are unrealistic for tiny smoke-test datasets.

Separate correctness tests from statistical learning-performance tests.

Run targeted tests during implementation.

Run the complete regression suite before final acceptance.

Do not disable failing tests to achieve a passing result.

Document actual results and explicitly mark unexecuted validation.

## 21. Phase Q — Documentation Volume 5

Create:

**Documentation Volume 5 — Long Short-Term Memory Networks**

Preserve the existing volume naming and location conventions.

Include:

- Introduction to LSTMs.
- Limitations of vanilla RNNs.
- Long-range dependencies.
- LSTM architecture.
- Hidden and cell states.
- Forget gate mathematics.
- Input gate mathematics.
- Candidate-memory mathematics.
- Cell-state update.
- Output gate mathematics.
- Numerical worked examples.
- Tensor dimensions.
- Parameter-count derivations.
- PyTorch gate ordering.
- LSTM architecture diagrams.
- Sequence unrolling diagrams.
- BPTT through LSTM.
- Gradient behavior.
- Sequence classification.
- Forecasting.
- RNN versus LSTM comparisons.
- Guided platform tutorials.
- Experiments and expected observations.
- Quiz answers and explanations.
- Troubleshooting.
- Limitations of LSTMs.
- Motivation for future sequence and attention architectures.

Use accurate mathematical notation and readable diagrams.

Update:
- README.
- `docs/ROADMAP.md`.
- The appropriate architecture decision record.

Do not remove or rewrite existing documentation volumes unnecessarily.

## 22. Phase R — Handoff and ChatGPT Synchronization

Maintain:

`docs/milestones/M7_HANDOFF.md`

Include:
- Current milestone and status.
- Starting commit.
- Current branch and commit.
- Requirements and design references.
- Implemented functionality.
- Architecture decisions.
- Files changed.
- Tests executed.
- Passed/failed/skipped/untested checks.
- Known issues.
- Outstanding acceptance criteria.
- Recommended next action.
- Commands to resume development.

Update the handoff after meaningful implementation batches.

At completion, record exact acceptance evidence in `M7_COMPLETION_REPORT.md`.

The handoff must be sufficient for a new ChatGPT or Codex session to continue without relying on historical chat context.

## 23. Phase S — Final Validation and Git Delivery

When implementation is ready:

1. Run the full regression suite.
2. Execute LSTM training smoke tests.
3. Verify M4/M5/M6 compatibility.
4. Run available headless or actual Streamlit integration checks.
5. Review parameter counts and numerical correctness.
6. Inspect source changes for quality.
7. Check for secrets, caches, generated datasets, and unnecessary artifacts.
8. Update documentation and handoff.
9. Commit changes on `feature/m7-lstm-learning-studio`.
10. Push the feature branch.
11. Prepare a pull request against master.

Do not merge automatically.

Do not generate ZIP files.

Do not duplicate the repository.

Preserve existing user changes.

## 24. M7 Acceptance Criteria

M7 is complete only when:

- The LSTM lesson is integrated into Learning Studio.
- The LSTM cell explorer executes actual PyTorch calculations.
- Forget, input, candidate, and output-gate values can be inspected.
- Cell-state and hidden-state evolution are visualized.
- LSTM architecture building works.
- Tensor shapes and parameter counts are correct.
- Sequence classification works.
- Time-series forecasting works.
- RNN versus LSTM comparison is available.
- BPTT and gradient inspection work.
- Gradient clipping works.
- Variable-length sequence demonstrations are correct.
- M4–M6 functionality remains intact.
- Previous progress JSON files remain compatible.
- Required automated tests pass.
- Documentation Volume 5 is complete.
- README and roadmap are updated.
- M7 handoff and completion reports are accurate.
- Changes are committed to a feature branch.
- A pull request is prepared against master.

If a mandatory criterion is incomplete or unverified, report it explicitly.

Do not declare unverified tests or UI interactions passed.

## 25. Long-Term Roadmap

Learning sequence:

1. Feed-Forward Neural Network — M4 completed.
2. Deep Neural Network — M4 completed.
3. CNN — M5 completed.
4. RNN — M6 completed.
5. LSTM — M7 current.
6. Embedding Network — Future.
7. Single-Head Self-Attention — Future.
8. Multi-Head Attention — Future.
9. Transformer Encoder — Future.
10. Transformer Decoder — Future.
11. Mini GPT — Future.

GRU may be considered later if educationally valuable, but it is not a mandatory M7 deliverable.

TensorFlow/Keras support remains deferred until PyTorch modules are stable.

Maintain extensibility without prematurely implementing later milestones.

## 26. Autonomous Execution Instructions

Proceed autonomously through the milestone.

Use the full requirements as a reference while implementing in focused, independently verifiable batches.

Do not stop after generating documentation or an implementation plan.

Do not request approval for routine engineering decisions.

Do not repeatedly inspect unchanged files or rerun expensive operations without cause.

Prioritize correctness, educational clarity, modularity, maintainability, compatibility, and execution efficiency.

At completion, provide a concise report covering:

1. Starting Git commit.
2. Final Git commit.
3. Implemented M7 capabilities.
4. Files changed.
5. Automated test results.
6. Streamlit and integration checks performed.
7. Documentation updates.
8. Remaining issues or unverified functionality.
9. Pull request URL.
10. Final M7 acceptance status.

**Start immediately by reading AGENTS.md and the Operating Model, verifying the M6 baseline, and creating the M7 feature branch. Then implement M7 in focused batches.**