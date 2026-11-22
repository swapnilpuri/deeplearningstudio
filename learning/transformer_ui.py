"""M10 presentation, using the same PyTorch traces for teaching and training."""
import json
from dataclasses import replace
import pandas as pd
import streamlit as st
from .navigation import render_learning_section_navigation
import torch
from .transformer_spec import TransformerSpec
from .transformer_adapter import (MultiHeadAttention, TransformerBlock, parameter_counts,
                                   complexity, residual, layer_norm_trace)
from .transformer_training import TransformerTrainingConfig, train_transformer
from .transformer_content import SECTIONS, EXPLANATIONS, QUESTIONS, evaluate_answers
from .attention_adapter import visibility, self_attention
from .embedding_adapter import tokenize, build_vocabulary, encode, embedding_table
from .progress import mark_complete, record_quiz, export_progress, load_progress
from visualization.transformer_graphs import head_figure, all_heads_figure, block_diagram
from visualization.attention_graphs import distribution_figure
from visualization.embedding_graphs import curves_figure


def _tensor(value: torch.Tensor, name: str, labels=None):
    value = value.detach().cpu()
    st.caption(f'{name} · shape {list(value.shape)}')
    st.dataframe(pd.DataFrame(value.numpy(), index=labels), alt=name)


def _plot(figure, key: str, title: str):
    st.plotly_chart(figure, key=key, width='stretch', alt=title)


def _configuration(prefix='m10') -> TransformerSpec:
    d = st.slider('Model dimension (d_model)', 1, 32, 8, key=prefix+'_d', persist_state='session')
    h = st.slider('Number of heads', 1, 8, 2, key=prefix+'_h', persist_state='session')
    ff = st.slider('Feed-forward dimension', 1, 128, 16, key=prefix+'_ff', persist_state='session')
    activation = st.selectbox('Activation', ('GELU', 'ReLU'), key=prefix+'_activation', persist_state='session')
    norm = st.selectbox('Normalization order', ('Pre-LayerNorm', 'Post-LayerNorm'), key=prefix+'_norm', persist_state='session')
    eps = st.number_input('LayerNorm epsilon', 1e-8, 1., 1e-5, format='%.8f', key=prefix+'_eps', persist_state='session')
    seed = int(st.number_input('Random seed', 0, 2147483647, 42, key=prefix+'_seed', persist_state='session'))
    return TransformerSpec(d, h, ff, activation, norm, eps, seed)


def _inputs(spec: TransformerSpec):
    text = st.text_input('Token sequence', 'The cat sat on the mat', key='m10_text', persist_state='session')
    tokens = tokenize(text)
    if not 1 <= len(tokens) <= 24 or len(text) > 2000:
        raise ValueError('Enter 1–24 tokens and at most 2000 characters')
    source = st.selectbox('Embedding source', ('Synthetic demonstration', 'M8 embedding table',
        'M8 trained skip-gram', 'Custom matrix JSON'), key='m10_source', persist_state='session')
    vocab, _ = build_vocabulary([text])
    ids = encode(tokens, vocab)
    if source == 'M8 trained skip-gram':
        run = st.session_state.get('emb_word_run')
        if run is None:
            raise ValueError('Train skip-gram in M8 first')
        ids = encode(tokens, run['vocabulary'])
        x = run['trained'][ids].detach().clone()
        if x.shape[-1] != spec.d_model:
            raise ValueError(f'Set d_model to the M8 trained width ({x.shape[-1]}) and choose a valid head count')
        st.caption('Learned M8 embeddings reused. Unknown tokens use UNK.')
    elif source == 'Custom matrix JSON':
        raw = st.text_area('Input matrix JSON [T,D]', json.dumps(torch.eye(len(tokens),spec.d_model).tolist()),
                           key=f'm10_custom_{len(tokens)}_{spec.d_model}', persist_state='session')
        x = torch.tensor(json.loads(raw), dtype=torch.float32)
        if x.shape != (len(tokens),spec.d_model):
            raise ValueError('Custom matrix must match [token count, d_model]')
    else:
        x = embedding_table(len(vocab),spec.d_model,spec.seed,
                            initialization='uniform' if source.startswith('Synthetic') else 'normal')(ids).detach()
        st.caption('Untrained demonstration embeddings; semantic relationships are not implied.')
    mode = st.selectbox('Mask', ('Bidirectional','Causal','Padding','Causal + padding'), key='m10_mask', persist_state='session')
    count = st.slider('Trailing padding keys', 0, len(tokens), 0, key=f'm10_pad_{len(tokens)}',
                      persist_state='session') if 'padding' in mode.lower() else 0
    padding = torch.arange(len(tokens)) >= len(tokens)-count
    mask = visibility(len(tokens), mode.startswith('Causal'), padding)
    labels = [f'{i}: {token}' for i,token in enumerate(tokens)]
    return x.unsqueeze(0), mask, labels


def _parameters(block, length):
    st.dataframe(pd.DataFrame([dict(component=k,parameters=v) for k,v in parameter_counts(block).items()]),
                 alt='Actual trainable parameter counts')
    st.dataframe(pd.DataFrame([dict(operation=k,count=v) for k,v in complexity(block.spec,length).items()]),
                 alt='Attention cells and approximate multiply-accumulates per sequence')
    st.caption('Dense scores need H*T² cells. QK and AV each cost approximately T²*d_model '
               'multiply-accumulates; projections and FFN scale linearly with T. Softmax, bias, '
               'activation and normalization costs are omitted. Fixed d_model keeps MHA parameter '
               'count and leading QK/AV arithmetic constant as heads change; score storage grows with heads.')


def _training():
    spec = _configuration('m10_train')
    steps = st.slider('Training steps',1,300,80,key='m10_steps',persist_state='session')
    lr = st.number_input('Learning rate',.001,1.,.01,format='%.3f',key='m10_lr',persist_state='session')
    optimizer = st.selectbox('Optimizer',('Adam','SGD'),key='m10_optimizer',persist_state='session')
    causal = st.checkbox('Causal training',key='m10_train_causal',persist_state='session')
    config = TransformerTrainingConfig(spec,steps,lr,optimizer,causal)
    if st.session_state.get('m10_train_config') != config:
        st.session_state.pop('m10_run',None)
    st.session_state.m10_train_config = config
    if st.button('Train Transformer block',key='m10_train'):
        with st.spinner('Training the small CPU experiment'):
            st.session_state.m10_run = train_transformer(config)
    if st.button('Reset to initial parameters',key='m10_reset'):
        st.session_state.pop('m10_run',None)
    run = st.session_state.get('m10_run')
    _parameters(TransformerBlock(spec),4)
    if run is None:
        st.info('Initial parameters restored. Training runs only on button press.')
        initial = TransformerBlock(spec)
        name = st.selectbox('Initial parameter',dict(initial.named_parameters()),key='m10_initial_parameter', persist_state='session')
        value = dict(initial.named_parameters())[name]
        _tensor(value.reshape(1,-1) if value.ndim == 1 else value,name)
        return
    st.write(f"Loss: {run['losses'][0]:.6f} → {run['losses'][-1]:.6f}")
    for keys,title,key in ((['loss'],'Post-update training loss','m10_loss'),
        (list(run['gradients']),'Pre-update gradient norms','m10_grad')):
        figure = curves_figure(run['history'],keys,title).update_layout(xaxis_title='Optimizer step')
        _plot(figure,key,title)
    labels = list('ABCD')
    for stage in ('before','after'):
        _plot(all_heads_figure(run[stage]['mha']['probabilities'][0],labels),
              'm10_training_'+stage,stage+' training attention for sequence 0')
        _tensor(run[stage]['output'][0],stage+' output, sequence 0',labels)
    _tensor(run['X'][0],'Training input, sequence 0',labels)
    _tensor(run['target'][0],'Target, sequence 0',labels)
    name = st.selectbox('Inspect trained parameter',list(run['initial']),key='m10_parameter', persist_state='session')
    if name is None:
        return
    for title,value in (('Initial',run['initial'][name]),('Trained',run['trained'][name]),
        ('Change',run['trained'][name]-run['initial'][name]),('Final pre-update gradient',run['gradients'][name])):
        _tensor(value.reshape(1,-1) if value.ndim == 1 else value,title+' '+name)
    st.caption('Patterns show sequence 0 of eight fixed training examples. Gradients are from the '
               'last backward pass, before its update. Control changes clear stale results. '
               'Small budgets, SGD, large rates and different seeds need not converge.')


def _exercises():
    answers = [st.radio(q,options,index=None,key=f'm10_quiz_{i}',persist_state='session')
               for i,(q,options,_,_) in enumerate(QUESTIONS)]
    if st.button('Check exercises',key='m10_grade'):
        try:
            score = evaluate_answers(answers)
        except ValueError as exc:
            st.warning(str(exc))
        else:
            st.session_state.learning_progress_v1 = record_quiz(st.session_state.learning_progress_v1,'transformer',score,len(QUESTIONS))
            st.success(f'{score}/{len(QUESTIONS)} correct')
            for q,options,correct,explanation in QUESTIONS:
                st.caption(f'{q} Answer: {options[correct]}. {explanation}')
    p = st.session_state.learning_progress_v1
    notes = st.text_area('Transformer experiment notes',p.get('exercise_notes',{}).get('transformer',''),key='m10_notes',persist_state='session')
    if st.button('Save notes',key='m10_save_notes'):
        st.session_state.learning_progress_v1 = {**p,'exercise_notes':{**p.get('exercise_notes',{}),'transformer':notes}}
    if st.button('Mark Transformer lesson complete',key='m10_complete'):
        st.session_state.learning_progress_v1 = mark_complete(st.session_state.learning_progress_v1,'transformer')
    st.download_button('Export learning progress',export_progress(st.session_state.learning_progress_v1),file_name='learning-progress.json',key='m10_export')
    upload = st.file_uploader('Import learning progress',type='json',key='m10_import')
    if upload and st.button('Load progress',key='m10_load'):
        st.session_state.learning_progress_v1 = load_progress(upload.getvalue())
        st.rerun()


def render_transformer_learning_studio():
    st.subheader('Multi-Head Attention & Transformer Blocks')
    for section, tab in render_learning_section_navigation(SECTIONS, 'm10').items():
        if not tab.open:
            continue
        with tab:
            index = SECTIONS.index(section)
            st.write(EXPLANATIONS[index])
            try:
                if index == 10:
                    _training()
                    return
                if index == 11:
                    _exercises()
                    return
                spec = _configuration()
                x,mask,labels = _inputs(spec)
                model = TransformerBlock(spec)
                with torch.no_grad():
                    trace = model.mha(x,mask)
                    block = model(x,mask)
                h = st.selectbox('Inspect head',range(spec.num_heads),format_func=lambda h:f'Head {h+1}',key=f'm10_head_{spec.num_heads}', persist_state='session')
                query = st.selectbox('Inspect query token',range(len(labels)),format_func=lambda i:labels[i],key=f'm10_query_{len(labels)}', persist_state='session')
                st.caption(f'Standard mode: X {list(x.shape)}; heads {spec.num_heads}; d_k=d_v={spec.d_head}; '
                           f'packed Q/K/V each {list(x.shape)}; concatenated {list(trace["concat"].shape)}; output {list(trace["output"].shape)}')
                _tensor(x[0],'Input embeddings X',labels)
                if index in (0,1,2):
                    start,end = h*spec.d_head,(h+1)*spec.d_head
                    for name in ('Wq','Wk','Wv'):
                        _tensor(getattr(model.mha,name)[:,start:end],f'{name}, selected head')
                    for name in ('Q','K','V'):
                        _tensor(trace[name][0,h],name+', selected head',labels)
                if index <= 5:
                    _plot(all_heads_figure(trace['probabilities'][0],labels),'m10_all_heads','All attention head probabilities')
                    _plot(distribution_figure(trace['probabilities'][0,h],labels,query),'m10_distribution','Selected head and query attention')
                    st.dataframe(pd.DataFrame(trace['probabilities'][0,:,query].T.numpy(),index=labels,
                        columns=pd.Index([f'Head {i+1}' for i in range(spec.num_heads)])),alt='Selected token attention across all heads')
                if index == 3:
                    key = st.selectbox('Inspect key token',range(len(labels)),format_func=lambda i:labels[i],key=f'm10_key_{len(labels)}', persist_state='session')
                    q,k = trace['Q'][0,h,query],trace['K'][0,h,key]
                    st.write('Coordinate products:',(q*k).tolist(),'dot product:',float(q@k), 'scaled:',float(trace['scaled_scores'][0,h,query,key]))
                    for name in ('scores','scaled_scores','masked_logits','probabilities'):
                        _tensor(trace[name][0,h],name+', selected head',labels)
                    _tensor(mask.float(),'Visibility: 1 permits, 0 blocks',labels)
                    unmasked = model.mha(x)['probabilities'][0,h]
                    _plot(head_figure(unmasked,labels,'Unmasked selected head'),'m10_unmasked','Unmasked probabilities')
                    _plot(head_figure(trace['probabilities'][0,h],labels,'Masked selected head'),'m10_masked','Masked probabilities')
                    st.write('Row sums:',trace['probabilities'][0,h].sum(-1).tolist())
                if index == 4:
                    for i in range(spec.num_heads):
                        _tensor(trace['heads'][0,i],f'Head {i+1} output',labels)
                    _tensor(trace['concat'][0],'Concatenated heads',labels)
                    _tensor(model.mha.Wo,'Output projection Wo')
                    _tensor(trace['output'][0],'MHA output',labels)
                if index == 5:
                    single = MultiHeadAttention(replace(spec,num_heads=1))
                    with torch.no_grad():
                        one = single(x,mask)
                        m9 = self_attention(x[0],dict(Wq=single.Wq,Wk=single.Wk,Wv=single.Wv),mask)
                    st.caption('Both models use identical seeded packed weights and input. Only head splitting and scaling change. M9 context is shown before the added Wo projection.')
                    for name,result,heads in (('Single head',one,1),('Multi-head',trace,spec.num_heads)):
                        _tensor(result['output'][0],name+' projected output',labels)
                        st.write(name,dict(heads=heads,parameters=sum(p.numel() for p in single.parameters()),
                            **complexity(replace(spec,num_heads=heads),len(labels))))
                    _tensor(m9['output'],'M9 single-head context before Wo',labels)
                    _plot(head_figure(one['probabilities'][0,0],labels,'Single head'),'m10_single','Single-head attention')
                if index == 6:
                    use = st.checkbox('Include residual connection',True,key='m10_residual', persist_state='session')
                    _tensor(trace['output'][0],'Isolated MHA sublayer output',labels)
                    _tensor((residual(x,trace['output']) if use else trace['output'])[0],'Result',labels)
                    st.caption('Element-wise: first token X + sublayer = '+str((x[0,0]+trace['output'][0,0]).tolist()))
                if index == 7:
                    gamma = st.slider('Demonstration gamma',.0,3.,1.,key='m10_gamma', persist_state='session')
                    beta = st.slider('Demonstration beta',-2.,2.,.0,key='m10_beta', persist_state='session')
                    with torch.no_grad():
                        model.ln1.weight.fill_(gamma)
                        model.ln1.bias.fill_(beta)
                        norm = layer_norm_trace(x,model.ln1)
                    for name in ('mean','variance','normalized','output'):
                        _tensor(norm[name][0],name,labels)
                    _tensor(norm['gamma'][None],'Learnable gamma')
                    _tensor(norm['beta'][None],'Learnable beta')
                    worked = torch.arange(1,spec.d_model+1,dtype=torch.float32).reshape(1,1,-1)
                    example = layer_norm_trace(worked,model.ln1)
                    st.write('Worked row:',worked.flatten().tolist(),'mean:',float(example['mean'].item()),
                             'variance (correction=0):',float(example['variance'].item()))
                    _tensor(example['output'][0],'Worked normalized and affine output')
                if index == 8:
                    with torch.no_grad(): ffn = model.ffn(x)
                    for name in ('linear1','activation','output'):
                        _tensor(ffn[name][0],'FFN '+name,labels)
                    for name,p in model.ffn.named_parameters():
                        _tensor(p[None] if p.ndim == 1 else p,'FFN '+name)
                if index == 9:
                    st.graphviz_chart(block_diagram(spec.norm),alt=spec.norm+' Transformer block with dashed residual paths')
                    stage = st.selectbox('Forward-pass stage',list(block['stages']),key='m10_stage', persist_state='session')
                    if stage is None:
                        return
                    operations = dict(X='Input embeddings',LN1='LN1(X)',MHA='Multi-head self-attention',
                        residual1='X + MHA(X)',U='First residual result (normalized in post-norm)',
                        LN2='LN2(U)',FFN='Linear2(Activation(Linear1(input)))',
                        residual2='U + FFN(U)',Y='Final residual result (normalized in post-norm)')
                    st.write('Operation:',operations[stage])
                    st.caption(f'All block stages consume and return [batch, tokens, d_model] = {list(x.shape)}. FFN internal width is {spec.d_ff}.')
                    _tensor(block['stages'][stage][0],stage+' numerical output',labels)
                    st.dataframe(pd.DataFrame([dict(stage=k,operation=operations[k],shape=str(list(v.shape)),
                        sample=str(v.flatten()[:4].tolist())) for k,v in block['stages'].items()]),alt='Transformer stage operations, shapes and samples')
                    _plot(all_heads_figure(block['mha']['probabilities'][0],labels),'m10_block_heads','Actual Transformer block attention')
                _parameters(model,len(labels))
            except (ValueError,TypeError,KeyError,IndexError,RuntimeError) as exc:
                st.warning(f'Check Transformer inputs: {exc}')
