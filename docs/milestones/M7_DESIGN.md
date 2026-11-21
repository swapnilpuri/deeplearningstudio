# M7 design

Preserve dense v1, CNN v2, RNN v3 and progress v1. Add framework-neutral LSTM v4
with the established recurrent shape/count conventions. Reuse RNN validation and
packed execution; LSTM adapter handles the (h_n,c_n) state pair. Explicit cell
arithmetic follows PyTorch i,f,g,o packing. No tensor math belongs in the UI.

Reuse M6 training via optional spec/adapter injection, with unchanged defaults.
LSTM training adds a validated configuration, seeded dropout, independently drawn
memory distractors, configurable signal distance and shared-data comparisons.
Forecasts use existing independent source series across splits, no fitted scaling.
Training reads only train/validation; held-out evaluation requires a button.

Gate interventions are a separately labelled controlled recurrence. Full BPTT
retains cell and hidden gradients, records gate slices and actual SGD/Adam updates.
Plots reuse M6 primitives; content, computation, visualizations and UI stay separate.
See ADR-0003 and the acceptance evidence matrix.
