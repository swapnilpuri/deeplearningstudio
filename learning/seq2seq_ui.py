"""M11 lazy horizontal panels; mathematical logic lives outside Streamlit."""
import pandas as pd
import streamlit as st
import torch
from .navigation import render_learning_section_navigation
from .progress import empty_progress, record_quiz, mark_complete, export_progress, load_progress
from .transformer_spec import TransformerSpec
from .seq2seq_spec import Seq2SeqSpec
from .seq2seq_adapter import EncoderDecoder, parameter_counts, complexity
from .seq2seq_training import (Seq2SeqTrainingConfig, train_seq2seq, synthetic_data, shift_targets,
                               sequence_loss, greedy_decode, token_labels)
from .seq2seq_content import SECTIONS, QUESTIONS, EXERCISES, evaluate_answers
from visualization.seq2seq_graphs import attention_figure, architecture_diagram


def _select(label, options, key):
    return st.selectbox(label, options, key='m11_'+key, persist_state='session')


def _slider(label, low, high, default, key):
    return st.slider(label, low, high, default, key='m11_'+key, persist_state='session')


def _tensor(value, name):
    value = value.detach().cpu()
    st.caption(f'{name} · shape {list(value.shape)}')
    if value.ndim > 2:
        value = value[0]
    if value.ndim == 1:
        value = value[None]
    st.dataframe(pd.DataFrame(value.numpy()), alt=name)


def _plot(values, keys, queries, title, key, probability=True):
    st.plotly_chart(attention_figure(values, keys, queries, title, probability),
                    key='m11_'+key, width='stretch', alt=title)


def _configuration():
    with st.expander('Transformer builder configuration'):
        d = _slider('Model dimension', 2, 32, 16, 'd')
        h = _slider('Attention heads', 1, 8, 2, 'h')
        ff = _slider('FFN hidden dimension', 1, 128, 32, 'ff')
        en = _slider('Encoder layers', 1, 4, 1, 'en')
        de = _slider('Decoder layers', 1, 4, 1, 'de')
        sv = _slider('Source vocabulary size', 4, 64, 7, 'sv')
        tv = _slider('Target vocabulary size', 4, 64, 7, 'tv')
        positional = _select('Positional encoding', ('sinusoidal', 'learned'), 'positional')
        norm = _select('Normalization strategy', ('Pre-LayerNorm', 'Post-LayerNorm'), 'norm')
        activation = _select('Activation', ('GELU', 'ReLU'), 'activation')
        dropout = _slider('Dropout on inputs and residual branches', 0., .5, 0., 'dropout')
        seed = int(st.number_input('Random seed', 0, 2147483647, 42, key='m11_seed', persist_state='session'))
    return Seq2SeqSpec(TransformerSpec(d, h, ff, activation, norm, seed=seed), sv, tv, en, de, positional, dropout)


def _demo_model(spec):
    if st.session_state.get('m11_demo_spec') != spec:
        st.session_state.m11_demo_model = EncoderDecoder(spec).eval()
        st.session_state.m11_demo_spec = spec
    return st.session_state.m11_demo_model


def _experiment(spec):
    run = st.session_state.get('m11_run')
    trained = bool(run is not None and run['config'].spec == spec and st.checkbox(
        'Inspect trained model and examples', True, key='m11_use_trained', persist_state='session'))
    if trained:
        model = run['model']
        i = _select('Training example', range(len(run['data']['source'])), 'example')
        source = run['data']['source'][i:i+1]
        target, _ = shift_targets(run['data']['target'][i:i+1])
        identity = id(model)
        st.caption('Trained parameters and training example; this is not held-out evaluation.')
    else:
        model = _demo_model(spec)
        source_text = st.text_input('Source token IDs (PAD=0, BOS=1, EOS=2)', '3 4 2 0',
                                    key='m11_source', persist_state='session')
        target_text = st.text_input('Decoder prefix token IDs', '1 3 4', key='m11_target', persist_state='session')
        if len(source_text) > 200 or len(target_text) > 200:
            raise ValueError('Use short token ID sequences')
        source = torch.tensor([[int(i) for i in source_text.split()]], dtype=torch.long)
        target = torch.tensor([[int(i) for i in target_text.split()]], dtype=torch.long)
        identity = id(model)
        st.caption('Randomly initialized layers illustrate arithmetic; meaningful linguistic features are not implied.')
    signature = (identity, tuple(source.flatten().tolist()), tuple(target.flatten().tolist()))
    if st.session_state.get('m11_trace_signature') != signature:
        with torch.no_grad():
            trace = model.inspect(source, target)
        st.session_state.m11_trace = trace
        st.session_state.m11_trace_signature = signature
    st.caption('Batch-first [B,S,D] encoder; [B,T,D] decoder; attention [B,H,T,S]. PAD masks block keys, not query rows.')
    return model, source, target, st.session_state.m11_trace


def _architecture(model, trace, source, target):
    st.graphviz_chart(architecture_diagram(model.spec), alt='Configured encoder-decoder with residual, normalization and memory paths')
    stages = {
        'Source IDs': (source, 'Integer source tokens [B,S]'),
        'Source embeddings': (trace['encoder']['embedding']['token'], 'M8 token embedding lookup [B,S] → [B,S,D]'),
        'Source positions': (trace['encoder']['embedding']['position'], 'Position index lookup or fixed sinusoid [S] → [B,S,D]'),
        'Combined source': (trace['encoder']['embedding']['combined'], 'Add token + position, equal [B,S,D] shapes'),
        'Encoder memory': (trace['encoder']['memory'], 'Final encoder output [B,S,D] supplies cross K,V'),
        'Target input IDs': (target, 'Shifted target or generated prefix [B,T]'),
        'Target embeddings': (trace['decoder']['embedding']['token'], 'Independent target lookup [B,T] → [B,T,D]'),
        'Target positions': (trace['decoder']['embedding']['position'], 'Target positional representation [B,T,D]'),
        'Combined target': (trace['decoder']['embedding']['combined'], 'Add token + position, equal [B,T,D] shapes'),
        'Vocabulary logits': (trace['logits'], 'Linear projection [B,T,D] → [B,T,V_target]'),
        'Token probabilities': (trace['logits'].softmax(-1), 'Softmax over vocabulary [B,T,V_target] → same shape'),
    }
    input_shapes = {name: list(value.shape) for name, (value, _) in stages.items()}
    input_shapes.update({
        'Source embeddings': list(source.shape), 'Source positions': [source.shape[1]],
        'Target embeddings': list(target.shape), 'Target positions': [target.shape[1]],
        'Vocabulary logits': list(trace['decoder']['output'].shape),
    })
    for path in ('encoder', 'decoder'):
        for i, layer in enumerate(trace[path]['layers']):
            for name, tensor in layer['stages'].items():
                label = f'{path} {i+1}: {name}'
                width = 'D → FF' if name == 'ffn_hidden' else 'FF → FF' if name == 'ffn_activated' else 'FF → D' if name == 'ffn' else 'D → D'
                stages[label] = (tensor, f'{path} layer {i+1}: {name}; order: {model.spec.block.norm}; feature width {width}; B and L preserved')
                input_shapes[label] = list(layer['ffn']['activation'].shape if name in ('ffn', 'ffn_activated')
                                            else layer['stages']['input'].shape)
    stage = _select('Architecture / tensor stage', list(stages), 'architecture_stage')
    tensor, operation = stages[stage]
    st.write(operation)
    st.write('Input shape:', input_shapes[stage], 'Output shape:', list(tensor.shape))
    _tensor(tensor, stage+' — batch 0 numerical sample')
    st.dataframe(pd.DataFrame([dict(stage=k, input_shape=str(input_shapes[k]), output_shape=str(list(v.shape)), operation=op,
        sample=str(v.flatten()[:4].tolist())) for k, (v, op) in stages.items()]), alt='Tensor operations, shapes and numerical samples')
    st.code(str(model), language='text')
    st.dataframe(pd.DataFrame([dict(component=k, parameters=v) for k, v in parameter_counts(model).items()]),
                 alt='Actual disjoint trainable parameter counts and total')
    st.dataframe(pd.DataFrame([complexity(model.spec, source.shape[1], target.shape[1])]),
                 alt='Theoretical attention score storage and QK AV multiply-accumulate estimates')
    st.caption('Counts are actual model parameters. Encoder layer categories and decoder layer categories include their own norms, FFN, self-attention and cross-attention. '
               'Score cells scale as encoder H*S², decoder H*T² and cross H*T*S per layer. QK/AV MAC estimates exclude projections, FFN, softmax and normalization; these are not measured runtimes.')


def _concepts(model, source, target, trace):
    st.markdown('M8 supplies token embeddings and sinusoidal positions; M9 computes scaled dot-product attention; '
                'M10 supplies packed multi-head projections, FFNs and residual building blocks. M11 assembles two sequence streams.')
    st.dataframe(pd.DataFrame([
        dict(architecture='Encoder-only', inputs='One source sequence', mask='Bidirectional + source PAD', cross='No', objective='Classification / representations', output='Contextual features'),
        dict(architecture='Decoder-only', inputs='One growing sequence', mask='Causal + padding', cross='No', objective='Next-token language modeling', output='Vocabulary logits'),
        dict(architecture='Encoder-decoder', inputs='Source + shifted target / prefix', mask='Source PAD; target causal + PAD', cross='Decoder Q, source K/V', objective='Translation / seq2seq', output='Target vocabulary logits'),
    ]), alt='Transformer architecture families, masks, objectives and outputs')
    st.latex(r'PE(p,2i)=\sin(p/10000^{2i/D}),\quad PE(p,2i+1)=\cos(p/10000^{2i/D})')
    st.write('Self-attention alone is permutation equivariant. Fixed sinusoids encode position without learned parameters; '
             'learned position embeddings are trainable table rows, bounded here by max_length. Token and position vectors are added without embedding scaling in this implementation.')
    length = _slider('Position explorer sequence length', 1, model.spec.max_length, 6, 'position_length')
    ids = torch.full((1, length), 3, dtype=torch.long)
    with torch.no_grad():
        positions = model.source(ids)
    st.write('Position indices:', list(range(length)))
    for name in ('token', 'position', 'combined'):
        _tensor(positions[name], 'Position explorer '+name)
    _plot(positions['position'][0], [str(i) for i in range(model.spec.block.d_model)],
          [str(i) for i in range(length)], 'Positions × embedding dimensions', 'positions', False)
    st.markdown('Each FFN applies **Linear → activation → Linear** independently per token. Residual addition preserves D; '
                'LayerNorm normalizes each token across its feature dimension. The final linear layer maps decoder features to target vocabulary logits.')
    _architecture(model, trace, source, target)


def _layer_explorer(model, source, target, trace, decoder=False):
    path = 'decoder' if decoder else 'encoder'
    layers = trace[path]['layers']
    i = _select(path.capitalize()+' layer', range(len(layers)), path+'_layer')
    layer = layers[i]
    st.write('Pre-norm: x + Dropout(Sublayer(LN(x))). Post-norm: LN(x + Dropout(Sublayer(x))). '
             'Decoder cross-attention normalizes the decoder query stream in pre-norm; memory remains the encoder output.')
    st.write('Decoder: causal self-attention → cross-attention → FFN.' if decoder else
             'Encoder: bidirectional self-attention → FFN. Final encoder layer output is memory.')
    for name, value in layer['stages'].items():
        _tensor(value, f'{path} {i+1}: {name}')
    head = _select('Self-attention head', range(model.spec.block.num_heads), path+'_head')
    ids = target[0] if decoder else source[0]
    labels = token_labels(ids)
    _plot(layer['self_attention']['probabilities'][0, head], labels, labels, path+' self-attention', path+'_attention')
    if not decoder:
        compare = _select('Compare encoder layer', range(len(layers)), 'compare_layer')
        _tensor(layers[compare]['output']-layer['output'], 'Encoder representation difference (comparison minus selected)')
        _tensor(trace['encoder']['memory'], 'Final encoder memory')
        st.dataframe(pd.DataFrame([dict(layer=j+1, norm=float(l['output'].norm()),
                                       change=float((l['output']-l['stages']['input']).norm())) for j, l in enumerate(layers)]),
                     alt='Representation norms and residual changes across encoder layers')
    else:
        _mask_explorer(trace, source, target, layer, head)
    _architecture(model, trace, source, target)


def _mask_explorer(trace, source, target, layer, head):
    kind = _select('Attention mask explorer', ('Encoder source padding', 'Decoder causal + target padding', 'Cross source padding'), 'mask_kind')
    if kind.startswith('Encoder'):
        attention = trace['encoder']['layers'][0]['self_attention']
        queries, keys = token_labels(source[0]), token_labels(source[0])
    elif kind.startswith('Decoder'):
        attention = layer['self_attention']
        queries, keys = token_labels(target[0]), token_labels(target[0])
    else:
        attention = layer['cross_attention']
        queries, keys = token_labels(target[0]), token_labels(source[0])
    st.caption('1 permits attention; 0 forbids it. Visibility [B,Q,K] broadcasts over heads. '
               'PAD keys are blocked; padded query rows may still attend valid keys. Fully masked rows are rejected before softmax to avoid NaNs.')
    for name in ('visibility', 'masked_logits', 'probabilities'):
        matrix = attention[name][0, head]
        _tensor(matrix, kind+' '+name)
    _plot(attention['visibility'][0, head].float(), keys, queries, kind+' allowed positions', 'mask_visibility')
    _plot(attention['probabilities'][0, head], keys, queries, kind+' probabilities', 'mask_probabilities')


def _cross(model, source, target, trace):
    i = _select('Decoder layer for cross-attention', range(len(trace['decoder']['layers'])), 'cross_layer')
    h = _select('Cross-attention head', range(model.spec.block.num_heads), 'cross_head')
    cross = trace['decoder']['layers'][i]['cross_attention']
    queries, keys = token_labels(target[0]), token_labels(source[0])
    st.latex(r'Q=X_{dec}W_Q,\ K=MW_K,\ V=MW_V,\ A=\operatorname{softmax}(QK^T/\sqrt{d_k}+mask),\ O=AV')
    st.write('Decoder queries retrieve information from encoder memory M. Source padding masks apply here; '
             'a target causal triangle cannot replace them. Attention heads need not correspond to direct linguistic alignments.')
    _plot(cross['probabilities'][0, h], keys, queries, f'Cross probabilities: decoder layer {i+1}, head {h+1}', 'cross_probabilities')
    for name in ('Q', 'K', 'V', 'scores', 'scaled_scores', 'masked_logits', 'probabilities', 'heads'):
        _tensor(cross[name][0, h], 'Cross '+name)
    for name in ('concat', 'output'):
        _tensor(cross[name], 'Combined multi-head '+name)
    q = _select('Target query position', range(len(queries)), 'cross_query')
    k = _select('Source key position', range(len(keys)), 'cross_key')
    st.write(dict(target_query=queries[q], source_key=keys[k],
                  scaled_score=float(cross['scaled_scores'][0, h, q, k]),
                  probability=float(cross['probabilities'][0, h, q, k])))
    _tensor(cross['probabilities'][0, h, q, :, None] * cross['V'][0, h],
            'Selected query: per-source weighted value contributions (sum gives head output)')
    compare_layer = _select('Comparison decoder layer', range(len(trace['decoder']['layers'])), 'cross_compare_layer')
    compare_head = _select('Comparison head', range(model.spec.block.num_heads), 'cross_compare_head')
    _plot(trace['decoder']['layers'][compare_layer]['cross_attention']['probabilities'][0, compare_head],
          keys, queries, 'Comparison cross probabilities (same 0–1 scale)', 'cross_comparison')


def _training(spec):
    st.markdown('**Training:** the full source and shifted ground-truth target are available. Causal masking permits '
                'parallel target-position computation without future-token leakage. Cross-entropy ignores PAD labels; backpropagation updates encoder and decoder.\n\n'
                '**Inference:** encode the source once, begin at BOS, append the largest-logit token, and stop at EOS or the length limit. '
                'The decoder recomputes its prefix at each step; no KV cache is used.')
    task = _select('Synthetic task', ('copy', 'reverse'), 'task')
    steps = _slider('Training steps', 1, 300, 150, 'steps')
    batch = _slider('Batch size', 1, 32, 8, 'batch')
    lr = st.number_input('Learning rate', .0001, .1, .01, format='%.4f', key='m11_lr', persist_state='session')
    optimizer = _select('Optimizer', ('Adam', 'SGD'), 'optimizer')
    config = Seq2SeqTrainingConfig(spec, steps, batch, lr, optimizer, task)
    if st.button('Train tiny Transformer', key='m11_train'):
        with st.spinner('Training on 16 fixed synthetic CPU examples'):
            st.session_state.m11_run = train_seq2seq(config)
            st.session_state.pop('m11_trace_signature', None)
            st.session_state.pop('m11_generation', None)
    if st.button('Reset training', key='m11_reset'):
        st.session_state.pop('m11_run', None)
        st.session_state.pop('m11_trace_signature', None)
        st.session_state.pop('m11_generation', None)
    run = st.session_state.get('m11_run')
    if run is not None:
        if run['config'] != config:
            st.info('Saved run uses the configuration below. Press Train to replace it with the current controls.')
        st.code(str(run['config']), language='text')
        history = pd.DataFrame(run['history'])
        st.line_chart(history.set_index('step')[['loss']], alt='Full training-set evaluation loss by optimizer step')
        st.line_chart(history.set_index('step')[['token_accuracy', 'sequence_accuracy']], alt='Teacher-forced training token and sequence accuracy')
        st.line_chart(history.set_index('step')[['gradient_norm']], alt='Gradient norms before clipping to one')
        st.write('Greedy exact match before / after:', run['greedy_accuracy_before'], run['greedy_accuracy_after'])
        st.dataframe(pd.DataFrame([dict(source=' '.join(token_labels(s)), expected=' '.join(token_labels(t)),
            before=' '.join(token_labels(b)), after=' '.join(token_labels(a))) for s, t, b, a in zip(
                run['data']['source'], run['data']['target'], run['before'], run['after'])]), alt='Training examples and greedy predictions before and after training')
        model, data = run['model'], run['data']
    else:
        st.info('Training runs only on an explicit button press. The default copy task uses 150 Adam steps, lr=0.01 and seed 42. '
                'Other settings need not converge; all displayed accuracy metrics concern the training set.')
        model = _demo_model(spec)
        data = synthetic_data(spec, task=task)
    row = _select('Teacher-forcing example', range(len(data['source'])), 'teacher_example')
    target = data['target'][row:row+1]
    decoder_input, labels = shift_targets(target)
    source = data['source'][row:row+1]
    st.write('Source:', token_labels(source[0]), 'Target:', token_labels(target[0]))
    st.write('Encoder IDs:', source.tolist(), 'Decoder input IDs:', decoder_input.tolist(), 'Labels:', labels.tolist())
    with torch.no_grad():
        trace = model.inspect(source, decoder_input)
        per_position = torch.nn.functional.cross_entropy(trace['logits'].transpose(1, 2), labels, ignore_index=0, reduction='none')[0]
    st.dataframe(pd.DataFrame(dict(decoder_input=token_labels(decoder_input[0]), expected=token_labels(labels[0]),
        predicted=token_labels(trace['logits'][0].argmax(-1)), ignored_PAD=(labels[0] == 0).tolist(),
        loss=per_position.tolist())), alt='Shifted teacher-forcing inputs, labels, predictions and individual losses')
    _tensor(trace['logits'], 'Per-position vocabulary logits')
    _tensor(trace['logits'].softmax(-1), 'Per-position vocabulary probabilities')
    st.write('Mean non-PAD cross-entropy:', float(sequence_loss(trace['logits'], labels)))
    st.caption('Per-position PAD loss is zero; the mean divides by the number of non-PAD labels.')
    if st.button('Generate greedily', key='m11_generate'):
        prediction = greedy_decode(model, source)
        st.session_state.m11_generation = dict(source=source.tolist(), prediction=token_labels(prediction[0]))
    if 'm11_generation' in st.session_state:
        st.write('Last explicit generation:', st.session_state.m11_generation)
    _cross(model, source, decoder_input, trace)


def _quiz():
    answers = [st.radio(q, options, index=None, key=f'm11_quiz_{i}', persist_state='session')
               for i, (q, options, _, _) in enumerate(QUESTIONS)]
    if st.button('Check answers', key='m11_grade'):
        score = evaluate_answers(answers)
        st.session_state.learning_progress_v1 = record_quiz(st.session_state.learning_progress_v1,
                                                            'seq2seq', score, len(QUESTIONS))
        st.session_state.m11_graded = list(answers)
    if 'm11_graded' in st.session_state:
        graded = st.session_state.m11_graded
        st.success(f'Last submitted score: {evaluate_answers(graded)}/{len(QUESTIONS)}')
        for answer, (q, options, correct, explanation) in zip(graded, QUESTIONS):
            st.caption(f'{q} Your answer: {answer}. Correct: {options[correct]}. {explanation}')


def _exercises():
    for i, text in enumerate(EXERCISES):
        st.markdown(f'{i+1}. {text}')
    st.write('Check your understanding in the Quiz tab; each of its 15 questions provides correctness and explanatory feedback.')
    p = st.session_state.learning_progress_v1
    notes = st.text_area('Transformer experiment notes', p.get('exercise_notes', {}).get('seq2seq', ''),
                         key='m11_notes', persist_state='session')
    if st.button('Save notes', key='m11_save_notes'):
        st.session_state.learning_progress_v1 = {**p, 'exercise_notes': {**p.get('exercise_notes', {}), 'seq2seq': notes}}
    if st.button('Mark lesson complete', key='m11_complete'):
        st.session_state.learning_progress_v1 = mark_complete(st.session_state.learning_progress_v1, 'seq2seq')
    st.json(st.session_state.learning_progress_v1)
    st.download_button('Export learning progress', export_progress(st.session_state.learning_progress_v1),
                        file_name='learning-progress.json', key='m11_export')
    upload = st.file_uploader('Import learning progress', type='json', key='m11_import')
    if upload and st.button('Load progress', key='m11_load'):
        st.session_state.learning_progress_v1 = load_progress(upload.getvalue())
        st.rerun()


def render_seq2seq_learning_studio():
    st.subheader('Transformer Encoder & Decoder Learning Studio')
    st.session_state.setdefault('learning_progress_v1', empty_progress())
    for section, tab in render_learning_section_navigation(SECTIONS, 'm11').items():
        if not tab.open:
            continue
        with tab:
            try:
                if section == 'Quiz':
                    _quiz()
                elif section == 'Exercises & Progress':
                    _exercises()
                else:
                    spec = _configuration()
                    if section == 'Seq2Seq & Training':
                        _training(spec)
                        continue
                    model, source, target, trace = _experiment(spec)
                    if section == 'Concepts & Math':
                        _concepts(model, source, target, trace)
                    elif section in ('Encoder', 'Decoder'):
                        _layer_explorer(model, source, target, trace, decoder=section == 'Decoder')
                    else:
                        _cross(model, source, target, trace)
            except (ValueError, TypeError, RuntimeError) as exc:
                st.warning(f'Check M11 inputs: {exc}')
