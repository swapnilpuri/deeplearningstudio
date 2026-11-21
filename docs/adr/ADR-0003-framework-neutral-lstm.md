# ADR-0003: LSTM schema v4 alongside recurrent v3

Status: accepted for M7 implementation. Date: 2026-11-21.

Use a small separate LSTM dataclass, preserving all existing serialized formats.
Reuse recurrent dimension validation and shape-row conventions, packed execution,
training and plotting. The LSTM-specific adapter constructs nn.LSTM and understands
both states. Four packed gate slices use PyTorch order input, forget, candidate,
output. Each recurrent layer counts 4HI + 4HH + 8H with bias, plus its linear head.
Dropout is between stacked layers only and is disabled in evaluation traces.

A separate version avoids ambiguously interpreting an old RNN specification as
an LSTM. No migration is needed. JSON dispatch is explicit by version and architecture.
Future GRU support can follow this pattern when required; no generic compiler or
premature backend implementation is introduced. Progress remains v1 keyed by lstm.
