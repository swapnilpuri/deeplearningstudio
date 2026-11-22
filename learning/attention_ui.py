"""M9 presentation; all attention arithmetic lives in the PyTorch core."""
import json
import math
import pandas as pd
import torch
import streamlit as st
from .attention_spec import AttentionSpec
from .attention_adapter import projections, visibility, self_attention
from .attention_training import AttentionTrainingConfig, train_attention
from .attention_content import SECTIONS, EXPLANATIONS, QUESTIONS, evaluate_answers
from .embedding_adapter import tokenize, build_vocabulary, encode, embedding_table
from .progress import mark_complete, record_quiz, export_progress, load_progress
from visualization.attention_graphs import attention_figure, distribution_figure
from visualization.embedding_graphs import curves_figure


def _step_curve(history, keys, title):
    figure = curves_figure(history, keys, title)
    return figure.update_layout(xaxis_title='Optimizer step')


def _tensor(value, name, labels=None):
    value = value.detach().cpu()
    st.caption(f'{name} · shape {list(value.shape)} · dtype {value.dtype}')
    st.dataframe(pd.DataFrame(value.numpy(), index=labels), alt=name)


def _plot(figure, key, title):
    st.plotly_chart(figure, key=key, width='stretch', alt=title)


def _inputs():
    text = st.text_input('Token sequence', 'The cat sat on the mat', key='att_text', persist_state='session')
    tokens = tokenize(text)
    if not 1 <= len(tokens) <= 24 or len(text) > 2000:
        raise ValueError('Enter 1–24 whitespace-separated tokens (at most 2000 characters)')
    source = st.selectbox('Embedding source', ('Synthetic demonstration', 'M8 embedding table', 'M8 trained skip-gram', 'Custom matrix JSON'), key='att_source', persist_state='session')
    dim = st.slider('Embedding dimension', 1, 16, 4, key='att_dim', persist_state='session')
    seed = int(st.number_input('Random seed', 0, 2147483647, 42, key='att_seed', persist_state='session'))
    vocab, _ = build_vocabulary([text])
    ids = encode(tokens,vocab)
    if source == 'M8 trained skip-gram':
        run = st.session_state.get('emb_word_run')
        if run is None:
            raise ValueError('Train word embeddings in M8 first, then return here')
        ids = encode(tokens,run['vocabulary'])
        x = run['trained'][ids].clone()
        dim = x.shape[1]
        if dim > 32: raise ValueError('Use M8 embedding width <=32')
        st.caption('M8 learned weights reused; unknown words use UNK. Effective width: '+str(dim))
    elif source == 'M8 embedding table':
        x = embedding_table(len(vocab),dim,seed)(ids).detach()
        st.caption('M8 seeded lookup table; these weights are untrained.')
    elif source == 'Custom matrix JSON':
        raw = st.text_area('Input matrix JSON [T,D]', json.dumps(torch.eye(len(tokens),dim).tolist()), key=f'att_custom_{len(tokens)}_{dim}', persist_state='session')
        x = torch.tensor(json.loads(raw),dtype=torch.float32)
        if x.shape != (len(tokens),dim): raise ValueError('Input matrix must match token count and embedding dimension')
    else:
        # Repeated IDs use the same demonstration embedding.
        x = embedding_table(len(vocab),dim,seed,initialization='uniform')(ids).detach()
        st.caption('Synthetic demonstration embeddings: no genuine semantic relationships are implied.')
    dk = st.slider('Query/key dimension', 1, 16, 3, key='att_dk', persist_state='session')
    dv = st.slider('Value dimension', 1, 16, 2, key='att_dv', persist_state='session')
    init = st.selectbox('Projection initialization', ('normal','identity','zeros'), key='att_init', persist_state='session')
    weights = projections(AttentionSpec(dim,dk,dv,seed,init))
    if st.checkbox('Edit projection matrices as JSON', key='att_edit', persist_state='session'):
        for name, value in weights.items():
            raw = st.text_area(name+' JSON',json.dumps(value.tolist()),key=f'att_{name}_{dim}_{dk}_{dv}',persist_state='session')
            custom = torch.tensor(json.loads(raw),dtype=torch.float32)
            if custom.shape != value.shape: raise ValueError(name+' has an incompatible shape')
            weights[name] = custom
    scaled = st.checkbox('Scale logits by sqrt(dk)', True, key='att_scaled', persist_state='session')
    mode = st.selectbox('Mask', ('Bidirectional','Causal','Padding','Causal + padding'), key='att_mask', persist_state='session')
    padding_count = st.slider('Trailing padding keys',0,len(tokens),0,key=f'att_pad_{len(tokens)}',persist_state='session') if 'padding' in mode.lower() else 0
    padding = torch.arange(len(tokens)) >= len(tokens)-padding_count
    mask = visibility(len(tokens),mode.startswith('Causal'),padding)
    trace = self_attention(x,weights,mask,scaled)
    labels = [f'{i}: {token}' for i,token in enumerate(tokens)]
    st.dataframe(pd.DataFrame(dict(position=range(len(tokens)),token=tokens,token_id=ids.tolist(),padding_key=padding.tolist())),alt='Token positions, IDs and padding flags')
    return trace, weights, labels


def _training():
    st.code('optimizer.zero_grad()\ntrace = self_attention(X, weights)\nloss = routing_cross_entropy + output_mse\nloss.backward()\noptimizer.step()',language='python')
    steps = st.slider('Training steps',1,500,120,key='att_steps',persist_state='session')
    lr = st.number_input('Learning rate',.001,1.,.03,format='%.3f',key='att_lr',persist_state='session')
    optimizer = st.selectbox('Optimizer',('Adam','SGD'),key='att_optimizer',persist_state='session')
    seed = int(st.number_input('Training seed',0,2147483647,42,key='att_train_seed',persist_state='session'))
    config = AttentionTrainingConfig(steps,lr,optimizer,seed)
    if st.session_state.get('att_train_config') != config:
        st.session_state.pop('att_run',None)
    st.session_state.att_train_config = config
    if st.button('Train attention',key='att_train'):
        st.session_state.att_run = train_attention(config)
    if st.button('Reset to initial weights',key='att_reset'):
        st.session_state.pop('att_run',None)
    run = st.session_state.get('att_run')
    initial = projections(AttentionSpec(4,3,2,seed))
    if run is None:
        st.info('Initial weights restored. Training runs only when you press Train attention.')
        for name,w in initial.items(): _tensor(w,name+' initial')
        return
    st.write(f"Loss: {run['losses'][0]:.6f} → {run['losses'][-1]:.6f}")
    _tensor(run['X'],'Fixed synthetic X');_tensor(run['target_attention'],'Target cyclic routing');_tensor(run['target_output'],'Target output')
    _plot(_step_curve(run['history'],['loss'],'Post-update objective'),'att_loss','Training objective')
    _plot(_step_curve(run['history'],['Wq','Wk','Wv'],'Pre-update gradient norms'),'att_grad','Gradient norms by step')
    st.caption('Gradients below are from the final pre-update backward pass.')
    for name in initial:
        _tensor(run['initial'][name],name+' before');_tensor(run['trained'][name],name+' after')
        _tensor(run['gradients'][name],name+' gradient');_tensor(run['trained'][name]-run['initial'][name],name+' change')
    for stage in ('before','after'):
        _plot(attention_figure(run[stage]['probabilities'],list('ABCD'),stage+' training'),'att_train_'+stage,stage+' training routing')
    _tensor(run['after']['output'],'Trained output')
    st.caption('Changing training controls clears stale results. Different seeds, budgets and rates need not converge.')


def _exercises():
    answers = [st.radio(q,options,index=None,key=f'att_quiz_{i}',persist_state='session') for i,(q,options,_,_) in enumerate(QUESTIONS)]
    if st.button('Check exercises',key='att_grade'):
        score = evaluate_answers(answers)
        st.session_state.learning_progress_v1 = record_quiz(st.session_state.learning_progress_v1,'attention',score,len(QUESTIONS))
        st.success(f'{score}/{len(QUESTIONS)} correct')
        for _,_,_,explanation in QUESTIONS: st.caption(explanation)
    notes = st.text_area('Attention experiment notes',value=st.session_state.learning_progress_v1.get('exercise_notes',{}).get('attention',''),key='att_notes',persist_state='session')
    if st.button('Save notes',key='att_save_notes'):
        p = st.session_state.learning_progress_v1
        st.session_state.learning_progress_v1 = {**p,'exercise_notes':{**p.get('exercise_notes',{}),'attention':notes}}
    if st.button('Mark attention lesson complete',key='att_complete'):
        st.session_state.learning_progress_v1 = mark_complete(st.session_state.learning_progress_v1,'attention')
    st.download_button('Export learning progress',export_progress(st.session_state.learning_progress_v1),file_name='learning-progress.json',key='att_export')
    upload = st.file_uploader('Import learning progress',type='json',key='att_import')
    if upload and st.button('Load progress',key='att_load'):
        st.session_state.learning_progress_v1 = load_progress(upload.getvalue());st.rerun()


def render_attention_learning_studio():
    st.subheader('Single-Head Self-Attention Learning Studio')
    section = st.selectbox('Attention section',SECTIONS,key='att_section')
    index = SECTIONS.index(section)
    st.write(EXPLANATIONS[index])
    try:
        if index == 7: _training();return
        if index == 8: _exercises();return
        trace, weights, labels = _inputs()
        query = st.selectbox('Inspect query token',range(len(labels)),format_func=lambda i:labels[i],key='att_query')
        key = st.selectbox('Inspect key token',range(len(labels)),format_func=lambda i:labels[i],key='att_key')
        if index == 2:
            event = st.dataframe(pd.DataFrame(trace['scores'].numpy(), index=labels, columns=labels),
                key='att_score_cells_'+str(len(labels)), on_select='rerun', selection_mode='single-cell', alt='Select a query-key score cell to inspect its dot product')
            cells = event.selection.cells
            if cells:
                query, column = cells[0]
                key = labels.index(column)
                st.caption(f'Selected cell: query {labels[query]}, key {labels[key]}')
        _tensor(trace['X'],'Input X',labels)
        if index in (0,1):
            for name,w in weights.items(): _tensor(w,name)
            for name in ('Q','K','V'): _tensor(trace[name],name,labels)
        if index in (2,3):
            _tensor(trace['Q'],'Q',labels);_tensor(trace['K'].T,'K transpose')
            for name in ('scores','scaled_scores'): _tensor(trace[name],name,labels)
            products = trace['Q'][query] * trace['K'][key]
            st.write('Selected dot product: '+' + '.join(f'{a:.5f}×{b:.5f}' for a,b in zip(trace['Q'][query],trace['K'][key]))+f' = {products.sum():.6f}')
            st.write('Scaling divisor sqrt(dk):',math.sqrt(trace['Q'].shape[1]))
            _tensor(torch.softmax(trace['scores'],-1),'Unscaled softmax',labels)
            _tensor(torch.softmax(trace['scaled_scores'],-1),'Scaled softmax before mask',labels)
        if index == 4:
            _tensor(trace['visibility'].float(),'Visibility: 1 permits, 0 blocks',labels)
            _tensor(trace['masked_logits'],'Masked logits (forbidden = -infinity)',labels)
            _plot(attention_figure(trace['visibility'].float(),labels,'Visibility mask'),'att_mask_plot','Mask visibility')
        _tensor(trace['probabilities'],'Attention probabilities',labels)
        _plot(attention_figure(trace['probabilities'],labels,'Attention probabilities'),'att_prob_plot','Query-to-key attention probabilities')
        st.caption('Inspect any score cell using the query/key selectors; hover the heatmap for exact probabilities.')
        st.write('Row sums:',trace['probabilities'].sum(-1).tolist())
        st.write('Forbidden probability mass:',float(trace['probabilities'][~trace['visibility']].sum()))
        _plot(distribution_figure(trace['probabilities'],labels,query),'att_dist','Selected query routing coefficients')
        if index == 6:
            for name in ('scores','scaled_scores'):
                _plot(attention_figure(trace[name],labels,name),'att_heat_'+name,name)
        if index in (0,5,6):
            _tensor(trace['V'],'Value vectors',labels)
            _tensor(trace['probabilities'][query,:,None]*trace['V'],'Selected-query weighted value contributions',labels)
            _tensor(trace['output'],'Contextual output Z',labels)
            st.caption('Sum the weighted-contribution rows to recover the selected output row. Input and output widths can differ.')
        operations = dict(X='Embedding lookup / custom input',Q='X @ Wq',K='X @ Wk',V='X @ Wv',scores='Q @ K.T',scaled_scores='scores / sqrt(dk)',masked_logits='Selected logits + additive mask',probabilities='softmax(masked logits, key axis)',output='A @ V')
        st.dataframe(pd.DataFrame([dict(stage=name,shape=str(list(trace[name].shape)),dtype=str(trace[name].dtype),operation=op,sample=str(trace[name].flatten()[:4].tolist())) for name,op in operations.items()]),alt='Stage-by-stage tensor computation inspector')
    except (ValueError,TypeError,KeyError,IndexError,RuntimeError) as exc:
        st.warning(f'Check attention inputs: {exc}')
