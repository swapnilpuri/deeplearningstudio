# ADR-0002: Separate recurrent schema v3

Status: accepted for M6 implementation. Date: 2026-11-21.

Use a framework-neutral recurrent specification alongside dense schema v1 and CNN schema v2. Represent stacked unidirectional vanilla RNNs with shared hidden width, explicit activation, output selection and linear head. Infer batch/time shapes at runtime. Only the PyTorch adapter imports torch. Preserve existing serialization and progress formats.

This follows the M5 separation and avoids changing stable dense/CNN dispatch. Future LSTM/GRU/attention may add specifications and adapters after their requirements exist; M6 does not emulate their gates. Variable-length final states use packed execution, not a padded last position. Both PyTorch bias vectors count independently even though their forward contributions add.
