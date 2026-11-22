"""M11 mathematical contracts and reference parity; no Streamlit dependency."""
from dataclasses import replace
import pytest
import torch
from torch import nn
from learning.embedding_adapter import sinusoidal_positions
from learning.transformer_spec import TransformerSpec
from learning.seq2seq_spec import Seq2SeqSpec
from learning.seq2seq_adapter import EncoderDecoder, ResidualLayer, CrossAttention, attention_visibility, parameter_counts


@pytest.fixture
def spec():
    return Seq2SeqSpec(block=TransformerSpec(d_model=8, num_heads=2, d_ff=16))


def inputs():
    return torch.tensor([[3, 4, 2, 0], [5, 2, 0, 0]]), torch.tensor([[1, 3, 4], [1, 5, 0]])


def test_positions_values():
    p = sinusoidal_positions(3, 5)
    torch.testing.assert_close(p[0], torch.tensor([0., 1., 0., 1., 0.]))
    torch.testing.assert_close(p[1, :2], torch.tensor([torch.sin(torch.tensor(1.)), torch.cos(torch.tensor(1.))]))


@pytest.mark.parametrize('mode', ['sinusoidal', 'learned'])
def test_positions_shape_seed_and_rng(spec, mode):
    spec = replace(spec, positional=mode)
    rng = torch.random.get_rng_state()
    a, b = EncoderDecoder(spec), EncoderDecoder(spec)
    assert torch.equal(rng, torch.random.get_rng_state())
    trace = a.source(inputs()[0])
    assert trace['position'].shape == (2, 4, 8)
    torch.testing.assert_close(trace['combined'], trace['token'] + trace['position'])
    for x, y in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(x, y)
    if mode == 'learned':
        trace['combined'].sum().backward()
        assert a.source.positions is not None
        assert a.source.positions.weight.grad is not None


@pytest.mark.parametrize('kwargs', [dict(source_vocabulary=3), dict(decoder_layers=0),
    dict(encoder_layers=True), dict(max_length=25), dict(dropout=float('nan')),
    dict(positional='rope'), dict(dropout=.6)])
def test_invalid_spec(kwargs):
    with pytest.raises(ValueError):
        Seq2SeqSpec(**kwargs)


@pytest.mark.parametrize('norm', ['Pre-LayerNorm', 'Post-LayerNorm'])
@pytest.mark.parametrize('layers', [1, 3])
def test_forward_masks_gradients(spec, norm, layers):
    spec = replace(spec, block=replace(spec.block, norm=norm), encoder_layers=layers, decoder_layers=layers)
    model = EncoderDecoder(spec).eval()
    source, target = inputs()
    trace = model.inspect(source, target)
    assert trace['logits'].shape == (2, 3, 7)
    assert trace['encoder']['memory'].shape == (2, 4, 8)
    assert len(trace['encoder']['layers']) == layers
    for layer in trace['encoder']['layers']:
        p = layer['self_attention']['probabilities']
        assert torch.all(p.masked_select((source == 0)[:, None, None]) == 0)
        assert (p[0, :, 0, 1] > 0).all()  # Bidirectional, not causal.
    for layer in trace['decoder']['layers']:
        p = layer['self_attention']['probabilities']
        assert (p.triu(1) == 0).all()
        p = layer['cross_attention']['probabilities']
        assert p.shape == (2, 2, 3, 4)
        assert torch.all(p.masked_select((source == 0)[:, None, None]) == 0)
        torch.testing.assert_close(p.sum(-1), torch.ones(2, 2, 3))
    trace['logits'].square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    torch.testing.assert_close(model(source, target), trace['logits'])
    counts = parameter_counts(model)
    assert counts['total'] == sum(p.numel() for p in model.parameters()) == sum(v for k, v in counts.items() if k != 'total')


def _copy_attention(custom, reference):
    with torch.no_grad():
        reference.in_proj_weight.copy_(torch.cat([custom.Wq.T, custom.Wk.T, custom.Wv.T]))
        reference.in_proj_bias.zero_()
        reference.out_proj.weight.copy_(custom.Wo.T)
        reference.out_proj.bias.zero_()


@pytest.mark.parametrize('norm', ['Pre-LayerNorm', 'Post-LayerNorm'])
@pytest.mark.parametrize('activation', ['GELU', 'ReLU'])
def test_pytorch_reference_encoder_decoder(spec, norm, activation):
    spec = replace(spec, block=replace(spec.block, norm=norm, activation=activation))
    model = EncoderDecoder(spec).double().eval()
    pre = norm == 'Pre-LayerNorm'
    enc = nn.TransformerEncoderLayer(8, 2, 16, dropout=0, activation=activation.lower(),
                                    batch_first=True, norm_first=pre).double().eval()
    dec = nn.TransformerDecoderLayer(8, 2, 16, dropout=0, activation=activation.lower(),
                                    batch_first=True, norm_first=pre).double().eval()
    for custom, reference in ((model.encoder[0], enc), (model.decoder[0], dec)):
        assert isinstance(custom, ResidualLayer)
        _copy_attention(custom.self_attention, reference.self_attn)
        if hasattr(reference, 'multihead_attn'):
            _copy_attention(custom.cross_attention, reference.multihead_attn)
        reference.linear1.load_state_dict(custom.ffn.linear1.state_dict())
        reference.linear2.load_state_dict(custom.ffn.linear2.state_dict())
        for i, layer in enumerate(custom.norms):
            getattr(reference, f'norm{i+1}').load_state_dict(layer.state_dict())
    source, target = inputs()
    trace = model.inspect(source, target)
    memory = enc(trace['encoder']['embedding']['output'], src_key_padding_mask=source == 0)
    out = dec(trace['decoder']['embedding']['output'], memory,
              tgt_mask=torch.ones(3, 3, dtype=torch.bool).triu(1),
              tgt_key_padding_mask=target == 0, memory_key_padding_mask=source == 0)
    torch.testing.assert_close(memory, trace['encoder']['memory'], atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(out, trace['decoder']['output'], atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(model.projection(out), trace['logits'], atol=1e-9, rtol=1e-9)


def test_future_target_and_source_padding_invariance(spec):
    model = EncoderDecoder(spec).eval()
    source, target = inputs()
    first = model(source, target)
    changed = target.clone()
    changed[:, 2] = 6
    torch.testing.assert_close(first[:, :2], model(source, changed)[:, :2])
    encoded = model.encode(source)
    memory = encoded['memory'].detach().clone()
    memory[source == 0] = 1000
    torch.testing.assert_close(first, model.decode(target, memory, source == 0)['logits'])


@pytest.mark.parametrize('padding,causal', [(torch.ones(2, 3, dtype=torch.bool), False),
    (torch.tensor([[True, False]]), True)])
def test_fully_masked_rows_rejected(padding, causal):
    with pytest.raises(ValueError, match='Fully masked'):
        attention_visibility(padding.shape[1], padding, causal)


def test_bad_ids_and_batch(spec):
    model = EncoderDecoder(spec)
    source, target = inputs()
    for bad in (source.float(), source + 99, source[:, :0], source.unsqueeze(0)):
        with pytest.raises(ValueError):
            model(bad, target)
    with pytest.raises(ValueError):
        model(source, target[:1])


from learning.seq2seq_training import (synthetic_data, shift_targets, sequence_loss, greedy_decode,
    Seq2SeqTrainingConfig, train_seq2seq, PAD, BOS, EOS)


@pytest.mark.parametrize('task', ['copy', 'reverse'])
def test_data_and_shifting(spec, task):
    data = synthetic_data(spec, task=task)
    repeat = synthetic_data(spec, task=task)
    for name in data:
        assert torch.equal(data[name], repeat[name])
    decoder, labels = shift_targets(data['target'])
    assert (decoder[:, 0] == BOS).all()
    assert torch.equal(labels, data['target'][:, 1:])
    for source, target in zip(data['source'], data['target']):
        s = source[source >= 3]
        t = target[target >= 3]
        assert torch.equal(s if task == 'copy' else s.flip(0), t)
        assert int((source == EOS).sum()) == int((target == EOS).sum()) == 1
    assert (data['target'] == PAD).any()


@pytest.mark.parametrize('target', [[[3, 2]], [[1, 3, 0]], [[1, 3, 2, 4]], [[1, 2, 2]], [[1, 0, 2]]])
def test_invalid_shifts(target):
    with pytest.raises(ValueError):
        shift_targets(torch.tensor(target))


def test_padding_loss_and_gradient():
    logits = torch.randn(1, 3, 7, requires_grad=True)
    labels = torch.tensor([[3, 2, 0]])
    loss = sequence_loss(logits, labels)
    torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(logits[0, :2], labels[0, :2]))
    loss.backward()
    assert logits.grad is not None
    assert (logits.grad[:, 2] == 0).all()
    with pytest.raises(ValueError):
        sequence_loss(logits, torch.zeros_like(labels))


def test_greedy_eos_once_encode_and_restore_mode(spec, monkeypatch):
    model = EncoderDecoder(spec).train()
    calls = []
    original = model.encode
    def encode(source):
        assert not torch.is_grad_enabled()
        calls.append(1)
        return original(source)
    monkeypatch.setattr(model, 'encode', encode)
    with torch.no_grad():
        model.projection.weight.zero_()
        model.projection.bias.fill_(-10.)
        model.projection.bias[EOS] = 10.
    source, _ = inputs()
    prediction = greedy_decode(model, source)
    assert prediction.tolist() == [[BOS, EOS], [BOS, EOS]]
    assert calls == [1] and model.training
    assert not prediction.requires_grad
    with torch.no_grad():
        model.projection.bias[EOS] = -10.
        model.projection.bias[3] = 10.
    prediction = greedy_decode(model, source, 4)
    assert prediction.shape == (2, 5) and (prediction[:, 1:] == 3).all()
    for limit in (0, spec.max_length+1, True):
        with pytest.raises(ValueError):
            greedy_decode(model, source, limit)


def test_greedy_per_sequence_eos_padding(spec, monkeypatch):
    model = EncoderDecoder(spec)
    def decode(target, memory, padding):
        logits = torch.full((*target.shape, 7), -10.)
        logits[0, -1, EOS] = 10.
        logits[1, -1, EOS if target.shape[1] > 1 else 3] = 10.
        return dict(logits=logits)
    monkeypatch.setattr(model, 'decode', decode)
    assert greedy_decode(model, inputs()[0]).tolist() == [[BOS, EOS, PAD], [BOS, 3, EOS]]


@pytest.mark.parametrize('kwargs', [dict(steps=0), dict(batch_size=33), dict(learning_rate=float('inf')),
                                   dict(optimizer='Other'), dict(task='translation')])
def test_invalid_training_config(kwargs):
    with pytest.raises(ValueError):
        Seq2SeqTrainingConfig(**kwargs)


def test_default_training_learns_and_reproduces():
    run = train_seq2seq(Seq2SeqTrainingConfig())
    assert run['losses'][-1] < run['losses'][0] * .1
    assert run['greedy_accuracy_after'] >= .875
    assert run['history'][-1]['token_accuracy'] >= .95
    config = Seq2SeqTrainingConfig(steps=2, spec=Seq2SeqSpec(dropout=.1))
    rng = torch.random.get_rng_state()
    a, b = train_seq2seq(config), train_seq2seq(config)
    assert a['losses'] == b['losses']
    assert torch.equal(a['after'], b['after'])
    assert torch.equal(rng, torch.random.get_rng_state())


def test_cross_projection_origins_and_parameter_formula(spec):
    model = EncoderDecoder(spec).eval()
    source, target = inputs()
    trace = model.inspect(source, target)
    layer = trace['decoder']['layers'][0]
    cross = layer['cross_attention']
    weights = model.decoder[0].cross_attention
    assert isinstance(weights, CrossAttention)
    memory = trace['encoder']['memory']
    for name, states, matrix in (('Q', layer['stages']['cross_input'], weights.Wq),
                                 ('K', memory, weights.Wk), ('V', memory, weights.Wv)):
        projected = (states @ matrix).reshape(2, states.shape[1], 2, 4).transpose(1, 2)
        torch.testing.assert_close(cross[name], projected)
    torch.testing.assert_close(cross['heads'], cross['probabilities'] @ cross['V'])
    assert parameter_counts(model)['total'] == 1583


@pytest.mark.parametrize('task,optimizer', [('reverse', 'Adam'), ('copy', 'SGD')])
def test_alternative_training_paths(task, optimizer):
    run = train_seq2seq(Seq2SeqTrainingConfig(steps=2, task=task, optimizer=optimizer))
    assert all(torch.isfinite(torch.tensor(run['losses'])))


def test_position_boundaries_and_visibility_validation(spec):
    for mode in ('learned', 'sinusoidal'):
        model = EncoderDecoder(replace(spec, positional=mode))
        with pytest.raises(ValueError):
            model.source(torch.full((1, spec.max_length+1), 3))
    for bad in (torch.zeros(2, 3), torch.zeros(3, dtype=torch.bool), torch.zeros(1, 25, dtype=torch.bool)):
        with pytest.raises(ValueError):
            attention_visibility(3, bad)
