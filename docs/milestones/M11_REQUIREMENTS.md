# DeepLearningStudio — Milestone M11 Implementation

## M11: Transformer Encoder & Decoder Learning Studio

### 1. Project Context

We are developing DeepLearningStudio, an interactive educational platform for learning Neural Networks, Deep Learning, and Generative AI.

Repository:
https://github.com/swapnilpuri/deeplearningstudio

Technology stack:
- Windows 11
- Python 3.12
- PyTorch
- Streamlit
- Plotly
- NumPy
- scikit-learn
- ONNX/Netron where applicable
- uv package manager
- VS Code + OpenAI Codex

Current baseline:
- Milestones M1–M10 are complete.
- M10 — Multi-Head Attention and Transformer Building Blocks — has been verified and merged into master.
- The shared learning studio navigation standardization has also been implemented and merged into master.
- M11 must reuse the existing architecture and navigation components.

Important: The repository is the source of truth for actual implementation structure. Do not assume specific module names or paths until they have been inspected.

---

### 2. Mandatory AI Engineering Operating Model

Follow AGENTS.md, AI_DEVELOPMENT_OPERATING_MODEL.md, and existing repository development/UI standards where present.

#### Responsibility boundaries

ChatGPT owns:
- High-level architecture direction
- Milestone scope
- Learning objectives
- Major design decisions
- Acceptance criteria
- Long-term roadmap

Codex owns:
- Repository inspection
- Detailed implementation design
- Coding and refactoring
- UI integration
- Automated tests
- Integration validation
- Documentation
- Git delivery preparation

#### Engineering principles

1. Inspect existing code before implementing.
2. Reuse M8 embeddings, M9 attention, and M10 Transformer blocks.
3. Reuse standardized navigation without introducing another navigation mechanism.
4. Keep the implementation modular and testable.
5. Avoid rewriting completed milestones.
6. Create a compact implementation plan of 3–7 steps.
7. Execute in small, independently verifiable batches.
8. Run targeted tests after meaningful batches.
9. Run complete regression tests at the end.
10. Use parallel agents only when tasks are independent and parallelism is beneficial.
11. Avoid repeatedly analyzing unchanged files.
12. Keep conversational progress updates concise.
13. Store detailed technical findings in repository documentation.
14. Use deterministic experiments and reproducible seeds.
15. Do not introduce unnecessary libraries, services, or infrastructure.
16. Do not request approval for routine engineering decisions.
17. Request human approval for destructive or consequential architectural changes.
18. Report failures transparently.
19. Do not merge into master without authorization.
20. Optimize for fast, correct implementation, not maximum planning or reasoning time.

Do not stop after producing a design document. Proceed with implementation.

---

### 3. Mandatory Navigation Standardization

This requirement is non-negotiable.

The project previously had inconsistent navigation:
- Earlier learning studios used horizontal tabs.
- Later studios used dropdown-based section navigation.

A shared navigation standardization refactoring has been completed and merged into master.

**M11 MUST use the existing shared horizontal tab navigation implementation.**

#### Requirements

1. Locate and inspect the reusable learning navigation component implemented in the standardization refactoring.
2. Reuse it directly in M11.
3. Follow existing styling, active-tab indication, layout, and spacing conventions.
4. Preserve module-specific section labels.
5. Use stable, namespaced widget/session-state keys.
6. Reuse existing quiz and exercise progress handling.
7. Avoid resetting experiments or progress when users change sections.
8. Avoid executing expensive computations when inactive sections are rendered.
9. Avoid automatic training on Streamlit rerenders.
10. Preserve keyboard and accessibility behavior supported by the shared component.

Suggested main M11 sections:

- Concepts & Math
- Encoder
- Decoder
- Cross-Attention
- Seq2Seq & Training
- Quiz
- Exercises & Progress

These section labels are recommended, not a reason to replace or duplicate the shared component.

If the existing shared navigation API has constraints, adapt M11 to that API.

Do not create a separate navigation abstraction.

Do not use a dropdown as the primary M11 section navigation mechanism.

If subsections are required, follow established nested navigation conventions and use the minimum controls necessary.

#### Mandatory validation

Verify:

- M11 tabs have the same visual behavior as the existing learning studios.
- Every educational section is accessible.
- Active navigation state behaves correctly.
- Session state does not collide with other modules.
- Switching tabs does not unexpectedly reset training, quiz, or experiment state.
- M1–M10 navigation remains unaffected.

Add targeted tests where supported.

---

### 4. Initial Repository Inspection

Before coding:

1. Inspect git status and current branch.
2. Preserve all uncommitted user changes.
3. Read repository engineering instructions.
4. Fetch the latest remote references if available.
5. Confirm M10 and navigation standardization are present in master.
6. Inspect M8 embedding components.
7. Inspect M9 attention components.
8. Inspect M10 multi-head attention and Transformer block implementations.
9. Inspect the standardized horizontal navigation component.
10. Inspect existing lessons, exercises, tests, and progress tracking.
11. Inspect the repository roadmap and milestone documentation.

Create a feature branch from the current master:

feature/m11-transformer-encoder-decoder

If the branch exists, inspect it before proceeding. Never overwrite existing work.

Identify reusable components and minimize duplication.

---

### 5. Milestone Objective

Implement an interactive Transformer Encoder & Decoder Learning Studio.

The objective is to teach how complete Transformer architectures are assembled from the building blocks introduced in M10.

Students must be able to:

- Understand positional information in Transformer inputs.
- Construct a Transformer encoder layer.
- Stack encoder layers.
- Inspect encoder representations.
- Construct a Transformer decoder layer.
- Understand masked decoder self-attention.
- Understand encoder-decoder cross-attention.
- Inspect decoder attention to encoder memory.
- Build a small encoder-decoder Transformer.
- Understand source and target sequences.
- Visualize tensor flows through the architecture.
- Understand teacher forcing.
- Compare training and inference.
- Train a tiny sequence-to-sequence model.
- Visualize generated token predictions.
- Complete practical exercises.

This milestone is educational.

Do not implement a production-scale Transformer or large language model.

---

### 6. Functional Requirements

#### M11.1 — Transformer Architecture Overview

Create a Concepts & Math section explaining:

- Encoder-only architectures
- Decoder-only architectures
- Encoder-decoder architectures
- Token embeddings
- Positional information
- Self-attention
- Cross-attention
- Feed-forward networks
- Residual connections
- Layer normalization
- Output vocabulary projection

Provide an interactive architecture overview.

Show how previously implemented M8–M10 components fit into the larger Transformer architecture.

Explain the differences between:

- Bidirectional encoder self-attention
- Causal decoder self-attention
- Encoder-decoder cross-attention

Use accurate mathematical explanations.

Keep the distinction between Transformer architecture types clear.

#### M11.2 — Positional Encoding Learning Module

Transformers need positional information because standard self-attention alone does not encode absolute token order.

Implement an interactive positional encoding explorer.

Support:

1. Sinusoidal positional encoding
2. Learned positional embeddings

Display:

- Token embeddings
- Position indices
- Positional encoding vectors
- Combined input representations
- Tensor shapes

Implement sinusoidal position encoding using the standard sine/cosine formulation.

Provide a visualization showing positional values across token positions and embedding dimensions.

Allow learners to change:

- Sequence length
- Embedding dimension
- Encoding mode
- Random seed

Explain the difference between fixed sinusoidal encodings and learned positional embeddings.

Do not implement RoPE, ALiBi, or relative positional attention in M11 unless an existing reusable implementation is already appropriate. These are future extensions.

#### M11.3 — Transformer Encoder Layer

Build an encoder layer using reusable M10 components:

- Multi-head self-attention
- Residual connection
- LayerNorm
- Position-wise feed-forward network
- Second residual connection
- Second LayerNorm, with appropriate pre-norm/post-norm ordering

Support configuration for:

- Model dimension
- Number of attention heads
- Feed-forward hidden dimension
- Activation function
- Normalization strategy
- Dropout where appropriate

Use bidirectional self-attention.

Support source padding masks.

Show the data flow through each component.

Display input and output tensors at each stage.

#### M11.4 — Encoder Stack Explorer

Allow learners to construct a stack of encoder layers.

Support a small configurable number of layers, such as 1–4 by default.

Display:

- Layer count
- Per-layer input
- Per-layer output
- Attention matrices
- Intermediate contextual representations
- Final encoder memory

Provide a layer selector.

Allow comparison of representations across encoder layers.

Visualize how representations change between layers.

Clarify that randomly initialized layers do not necessarily produce meaningful linguistic features.

Keep default configurations computationally lightweight.

#### M11.5 — Transformer Decoder Layer

Implement a decoder layer containing:

1. Masked self-attention
2. Cross-attention
3. Position-wise feed-forward network
4. Appropriate residual connections
5. Layer normalization

Reuse M10 building blocks where practical.

The decoder must support causal masking.

Display the forward pass stage by stage.

Provide interactive inspection of:

- Decoder input
- Masked self-attention output
- Cross-attention output
- FFN output
- Final decoder representation

Ensure the implemented pre-norm or post-norm equations match the selected architecture.

Support both when consistent with M10 capabilities.

#### M11.6 — Cross-Attention Learning Studio

Implement encoder-decoder cross-attention.

This is a major educational focus of M11.

For each cross-attention head:

Q = decoder_hidden_states @ W_Q

K = encoder_memory @ W_K

V = encoder_memory @ W_V

Attention = softmax((Q @ K.T) / sqrt(d_k) + Mask) @ V

Show:

- Decoder Query vectors
- Encoder Key vectors
- Encoder Value vectors
- Cross-attention score matrix
- Cross-attention probability matrix
- Weighted encoder value output
- Combined multi-head output

Use a selected decoder token to illustrate how it attends to source tokens.

Explain that cross-attention connects decoder representations to encoder information.

Support source padding masks.

Do not use decoder causal masks as a substitute for source padding masks.

#### M11.7 — Cross-Attention Visualization

Provide interactive heatmaps with:

- Source tokens on the key axis
- Target tokens on the query axis
- Attention probabilities
- Per-head selection
- Decoder-layer selection

Allow comparing cross-attention between different decoder layers and heads.

Clearly identify source and target token labels.

Use consistent color scales for comparable attention probabilities.

Provide a selected-cell numerical inspection.

For trained examples, allow users to observe learned source-to-target alignment patterns.

Do not claim every attention head represents a direct linguistic alignment.

#### M11.8 — Encoder–Decoder Transformer Builder

Allow configuration of a small complete encoder-decoder Transformer.

Parameters:

- Source vocabulary size
- Target vocabulary size
- Model dimension
- Attention heads
- Encoder layer count
- Decoder layer count
- FFN hidden dimension
- Positional encoding
- Normalization strategy
- Dropout
- Random seed

Use practical defaults with valid dimensional relationships.

Build:

Source token IDs
→ Source embeddings
→ Positional information
→ Encoder stack
→ Encoder memory

Target input token IDs
→ Target embeddings
→ Positional information
→ Decoder stack with masked self-attention and cross-attention
→ Output projection
→ Target vocabulary logits

Share the relevant M8–M10 components where appropriate.

Validate all configuration parameters.

Allow inspection of the complete model hierarchy and parameter counts.

#### M11.9 — Attention Mask Explorer

Provide a focused mask visualization.

Support:

- Encoder source padding masks
- Decoder causal masks
- Decoder target padding masks, where applicable
- Cross-attention source padding masks

Display:

- Allowed attention positions
- Forbidden attention positions
- Masked logits
- Final attention probabilities

Handle different source and target sequence lengths.

Use correct broadcasting across attention heads and batches.

Ensure valid attention rows contain at least one permitted key.

Avoid NaNs from fully masked rows.

Demonstrate why encoder self-attention and decoder self-attention use different masking behavior.

#### M11.10 — Sequence-to-Sequence Learning Experiment

Implement a small deterministic seq2seq task.

Recommended tasks:

- Copy a short token sequence
- Reverse a short token sequence
- Map synthetic source tokens to deterministic target tokens

Choose one primary default task that is small, reliable, and testable.

Provide clearly defined special tokens:

- PAD
- BOS
- EOS
- Any additional tokens required by the selected task

Explain source sequences and target sequences.

Show encoder input IDs and decoder input IDs separately.

Do not depend on downloading real translation corpora.

Use small synthetic vocabularies.

#### M11.11 — Teacher Forcing Explorer

Demonstrate teacher forcing during decoder training.

For an example target sequence:

Target:
[BOS, A, B, C, EOS]

Decoder input:
[BOS, A, B, C]

Prediction targets:
[A, B, C, EOS]

Display:

- Decoder input sequence
- Shifted expected output sequence
- Per-position predicted logits
- Per-position target labels
- Loss computation

Explain why the decoder receives previous ground-truth tokens during teacher-forced training.

Ensure causal masking prevents information leakage from future target positions.

Ignore padding tokens appropriately in the training loss.

#### M11.12 — Training vs Inference Comparison

Create an interactive comparison of:

**Training**

- Complete source sequence
- Shifted target sequence
- Teacher forcing
- Parallel target-position calculations
- Cross-entropy loss
- Backpropagation

**Inference**

- Complete source sequence
- Encoder memory computed once
- Decoder starts with BOS
- Tokens generated one step at a time
- Each new token appended to decoder input
- Stop on EOS or maximum length

Implement simple greedy decoding for the toy model.

Use PyTorch no-grad/inference mode during generation.

The initial M11 implementation may recompute the decoder sequence at each generation step.

KV caching is explicitly out of scope.

Do not implement beam search, top-k sampling, nucleus sampling, or production inference optimizations.

#### M11.13 — Interactive Training Studio

Support training of the tiny encoder-decoder Transformer.

Expose:

- Learning rate
- Training steps
- Batch size
- Model dimensions within safe limits
- Layer count within safe limits
- Random seed
- Optimizer

Use Adam as the default optimizer.

Support SGD if appropriate to existing conventions.

Visualize:

- Training loss
- Token prediction accuracy
- Sequence-level accuracy where meaningful
- Gradient norms
- Predicted sequences before training
- Predicted sequences after training
- Selected attention patterns

Training must run only through explicit user action.

Provide reset functionality.

Ensure deterministic training configurations where supported.

The default toy task must demonstrate meaningful learning under documented settings.

Avoid claiming guaranteed convergence for arbitrary user configurations.

#### M11.14 — Tensor and Architecture Inspector

Provide an interactive complete Transformer tensor inspector.

Stages:

Source token IDs
→ Source embeddings
→ Positional information
→ Encoder layers
→ Encoder memory

Target input IDs
→ Target embeddings
→ Positional information
→ Masked self-attention
→ Cross-attention
→ FFN
→ Vocabulary logits
→ Token probabilities

For every stage, display:

- Input shape
- Output shape
- Operation
- Numerical sample
- Educational explanation

Support selecting a specific encoder layer, decoder layer, or attention head.

Use a consistent batch-first or sequence-first convention internally, and document it clearly.

Avoid silent tensor transpositions.

#### M11.15 — Architecture Visualization

Create an interactive diagram of the full encoder-decoder Transformer.

Visually separate:

- Encoder path
- Decoder path
- Encoder memory
- Cross-attention connections
- Output projection

Show residual connections and normalization in the correct order.

The diagram must adapt to model configuration.

Clicking or selecting stages should display corresponding tensor information.

Use existing visualization libraries and components.

Do not add a heavyweight frontend framework.

Ensure diagrams reflect the actual PyTorch implementation.

#### M11.16 — Transformer Architecture Comparison

Provide an educational comparison between:

1. Encoder-only Transformers
2. Decoder-only Transformers
3. Encoder-decoder Transformers

Explain typical use cases:

- Sequence classification and representation learning
- Autoregressive language generation
- Translation and other seq2seq tasks

Clarify architectural differences regarding:

- Attention masks
- Input and output sequences
- Cross-attention
- Prediction objectives
- Output representations

Do not implement full encoder-only or decoder-only pretrained model families.

This section is primarily comparative.

#### M11.17 — Parameter and Complexity Inspector

Display:

- Embedding parameter counts
- Encoder layer parameter counts
- Decoder layer parameter counts
- Self-attention parameter counts
- Cross-attention parameter counts
- FFN parameter counts
- Output projection parameters
- Total trainable parameters

Derive counts from the actual PyTorch model.

Explain how:

- Source length affects encoder attention cost
- Target length affects decoder self-attention cost
- Source and target lengths affect cross-attention cost
- Layer counts affect total model size and computation

Provide theoretical estimates without presenting them as measured hardware runtime.

#### M11.18 — Quiz and Guided Exercises

Use the existing standardized learning framework.

Include exercises covering:

1. Why Transformers require positional information.
2. Encoder self-attention vs decoder self-attention.
3. Constructing a causal mask.
4. Determining encoder output tensor dimensions.
5. Understanding encoder memory.
6. Calculating cross-attention score matrix dimensions.
7. Identifying the origin of Q, K, and V in cross-attention.
8. Explaining teacher forcing.
9. Constructing shifted decoder inputs and labels.
10. Explaining training vs inference.
11. Identifying valid encoder-decoder configurations.
12. Comparing encoder-only, decoder-only, and encoder-decoder architectures.
13. Calculating approximate attention complexity.
14. Understanding source padding masks.
15. Explaining why greedy decoding stops at EOS.

Provide correctness checks and explanatory feedback.

Integrate with existing exercise and progress tracking.

Do not create a separate quiz framework.

---

### 7. Mathematical Correctness

Use PyTorch as the mathematical source of truth.

Required invariants:

- Source embeddings have correct model dimensions.
- Positional representations match token embedding dimensions.
- Encoder output has the correct batch, sequence, and feature dimensions.
- Decoder output has the correct batch, target-sequence, and feature dimensions.
- Encoder self-attention is bidirectional except for padding restrictions.
- Decoder self-attention is causal.
- Cross-attention Queries originate from decoder states.
- Cross-attention Keys and Values originate from encoder memory.
- Cross-attention score shape is [batch, heads, target_length, source_length].
- Source padding positions receive zero probability.
- Future decoder positions receive zero self-attention probability.
- Residual connections preserve expected feature dimensions.
- Training loss aligns with shifted target labels.
- PAD labels are excluded from loss calculations where applicable.
- Greedy decoding cannot access future target tokens.
- Gradients propagate through encoder and decoder parameters.
- Valid supported inputs do not produce NaNs or infinities.

Compare the custom encoder/decoder components with equivalent PyTorch reference implementations where configurations and parameter mappings permit.

Use deterministic seeds and documented numerical tolerances.

Do not claim mathematical equivalence without verifying it.

---

### 8. Engineering Architecture

Reuse the existing M8, M9, and M10 implementations.

Separate responsibilities into:

1. Positional encoding utilities
2. Encoder layer implementation
3. Encoder stack
4. Decoder layer implementation
5. Decoder stack
6. Encoder-decoder model composition
7. Attention mask utilities
8. Synthetic seq2seq dataset and vocabulary
9. Training and inference utilities
10. Visualization adapters
11. Streamlit learning UI
12. Exercises and documentation

Avoid circular dependencies.

Keep mathematical logic independent from Streamlit.

Maintain typing and established error-handling practices.

Use established session-state and caching conventions.

Do not add TensorFlow support in M11.

Preserve the architecture's potential for future backend extensions without premature abstraction.

---

### 9. Scope Boundaries

The following are excluded from M11:

- Large pretrained Transformer models
- Full GPT implementation
- Production machine translation
- Downloading large language datasets
- Production tokenizers
- Byte Pair Encoding implementation
- Beam search
- Top-k and top-p sampling
- KV caching
- Grouped Query Attention
- Multi-Query Attention
- FlashAttention
- PagedAttention
- Distributed training
- Fine-tuning pretrained foundation models
- TensorFlow backend implementation
- Production LLM serving

M11 provides the architecture foundation for later milestones.

Do not implement M12 or subsequent milestones prematurely.

---

### 10. Testing Requirements

Add comprehensive automated tests using repository conventions.

#### Positional encoding tests

- Sinusoidal values
- Shape correctness
- Learned positional embeddings
- Deterministic initialization
- Invalid configuration handling

#### Encoder tests

- Forward pass
- Self-attention masking
- Encoder layer output shape
- Multi-layer encoder operation
- Gradient propagation
- Pre-norm/post-norm behavior where supported

#### Decoder tests

- Causal masking
- Decoder self-attention
- Cross-attention
- Different source and target lengths
- Padding mask handling
- Decoder output shape
- Gradient propagation

#### Full Transformer tests

- End-to-end forward pass
- Target vocabulary logits
- Parameter counts
- Batch handling
- Deterministic outputs in evaluation mode
- Numerical comparison with equivalent PyTorch reference components

#### Seq2Seq tests

- Synthetic dataset generation
- BOS/EOS/PAD handling
- Target shifting
- Teacher forcing
- Padding-aware loss
- Greedy decoding
- EOS termination
- Maximum-length termination
- Default training task loss reduction

#### Navigation tests

Mandatory:

- M11 uses existing standardized horizontal navigation.
- All M11 sections are accessible.
- No primary section dropdown is introduced.
- Widget keys are namespaced.
- Section switching preserves relevant state.
- Training does not restart on navigation.
- Quiz and progress state is retained.
- Existing M1–M10 navigation behavior remains intact.

#### Full regression

Run all existing M1–M10 automated tests.

Run the project's established lint and type checks.

If UI interactions require manual testing, document explicit verification steps.

Do not silently skip failing tests.

---

### 11. Performance and Usability

M11 must run on typical local developer hardware.

Requirements:

- CPU-first execution
- Optional GPU acceleration
- Small model defaults
- Short synthetic sequences
- Limited layer and head counts by default
- Explicit training action
- Responsive Streamlit navigation
- No automatic large-model downloads
- No mandatory external API requests
- No new backend services
- No unnecessary database dependencies
- Reproducible experiments
- Clear input validation

Training state must survive ordinary Streamlit rerenders when appropriate.

Avoid expensive repeated model construction or inference caused by navigation changes.

Use caching carefully and avoid sharing mutable model state between unrelated sessions.

---

### 12. Implementation Sequence

Execute M11 in six focused batches.

#### Batch 1 — Baseline and Integration

- Inspect repository standards.
- Verify M10 and shared navigation implementation.
- Create feature branch.
- Identify reusable PyTorch components.
- Prepare compact implementation plan.

#### Batch 2 — Positional Encoding and Encoder

- Implement positional encoding.
- Implement encoder layer.
- Implement encoder stack.
- Add encoder visualizations.
- Add targeted tests.

#### Batch 3 — Decoder and Cross-Attention

- Implement decoder layer.
- Implement cross-attention.
- Implement masks.
- Implement decoder stack.
- Add attention visualizations.
- Add targeted tests.

#### Batch 4 — Full Transformer and Seq2Seq

- Compose encoder-decoder model.
- Implement toy dataset.
- Implement teacher forcing.
- Implement training.
- Implement greedy decoding.
- Add end-to-end tests.

#### Batch 5 — Educational UI

- Integrate standardized horizontal tabs.
- Implement architecture explorer.
- Implement tensor inspection.
- Implement training visualization.
- Implement quizzes and exercises.
- Add UI tests.

#### Batch 6 — Validation and Delivery

- Run full regression.
- Execute established quality checks.
- Verify navigation consistency.
- Update learning documentation.
- Update roadmap.
- Generate completion report.
- Prepare commits and pull request.

Do not stop between batches for routine confirmations.

---

### 13. Documentation Artifacts

Follow existing repository milestone documentation conventions.

Create or update the equivalent of:

- M11 requirements
- M11 architecture and design
- M11 acceptance criteria
- M11 learning guide
- M11 completion report
- Project roadmap

The learning guide must explain:

1. Overall Transformer architecture
2. Positional encoding
3. Encoder layers
4. Encoder stacking
5. Decoder layers
6. Masked self-attention
7. Cross-attention
8. Encoder memory
9. Source and target sequences
10. Teacher forcing
11. Target shifting
12. Training vs inference
13. Greedy decoding
14. Complete Transformer forward pass
15. Tensor dimensions
16. Parameter and attention complexity
17. Encoder-only vs decoder-only vs encoder-decoder

Include worked numerical examples and diagrams.

The completion report must contain actual execution results.

---

### 14. Definition of Done

M11 is complete only when:

- M11 appears in the application.
- Existing standardized horizontal navigation is reused.
- No new primary dropdown-based section navigation is introduced.
- Positional encoding visualizations work.
- Encoder layers work.
- Encoder stacking works.
- Decoder layers work.
- Decoder causal masking works.
- Cross-attention is mathematically correct.
- Encoder memory inspection works.
- Source and target masks are correct.
- Full encoder-decoder Transformer forward pass works.
- Teacher forcing is demonstrated.
- Target shifting is correct.
- Tiny seq2seq training works.
- Greedy decoding works.
- Training loss and predictions can be visualized.
- Architecture and tensor inspectors are functional.
- Parameter inspection is accurate.
- Guided exercises work.
- Existing progress infrastructure is reused.
- M1–M10 functionality remains intact.
- Navigation consistency regression tests pass.
- All automated tests pass.
- Documentation is complete.
- Completion report is generated.
- Feature branch is ready for review.

If any criterion is incomplete, report it clearly.

---

### 15. Git and Delivery

Feature branch:

feature/m11-transformer-encoder-decoder

Do not commit secrets, temporary datasets, model caches, or unnecessary large binaries.

Use meaningful commits.

Push the feature branch if remote access is available.

Create a pull request targeting master if supported.

If permission errors prevent PR creation, report the issue and provide the feature branch or comparison URL where available.

Do not merge to master automatically.

---

### 16. Final Completion Report

Provide a concise final summary:

**M11 Status:** Complete / Partial / Blocked

**Implemented:** Major delivered features

**Navigation:** Confirmation that the standardized horizontal tabs were reused

**Validation:** Executed test commands and actual pass/fail counts

**Documentation:** Generated documentation paths

**Git:** Branch, commits, PR URL

**Known Limitations:** Outstanding issues

**Next Milestone:** Suggested next milestone

Put detailed implementation findings into repository documentation.

---

### 17. Execution Instruction

Begin now.

1. Inspect the repository.
2. Confirm M10 and navigation standardization in master.
3. Create the M11 feature branch.
4. Reuse existing M8–M10 mathematical components.
5. Implement positional encoding and encoder.
6. Implement decoder and cross-attention.
7. Build a complete small encoder-decoder Transformer.
8. Implement seq2seq training and inference.
9. Integrate the standardized horizontal navigation.
10. Add educational visualizations and exercises.
11. Run targeted and full regression tests.
12. Update documentation.
13. Prepare the feature branch and pull request.

Do not stop after planning.

Do not ask for routine implementation approvals.

Optimize for efficient implementation, mathematical correctness, reuse, testability, and a consistent learning experience.

Complete as much of M11 as possible in this execution and prepare the result for review.