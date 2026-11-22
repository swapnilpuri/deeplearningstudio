"""Embedding lesson presentation; numerical work lives in adapter/training modules."""
import json
import numpy as np
import pandas as pd
import streamlit as st
from .navigation import render_learning_section_navigation
import torch
from .embedding_spec import EmbeddingModelSpec
from .embedding_adapter import (TOY_CORPUS, tokenize, build_vocabulary, encode, embedding_table,
    lookup_demo, EmbeddingAdapter, gradient_experiment, similarity, normalize_vectors,
    nearest_neighbors, project_embeddings, pool_embeddings)
from .embedding_training import (EmbeddingTrainingConfig, train_skipgram, train_classifier,
    evaluate_classifier_test)
from .embedding_content import TOPICS, QUESTIONS, EXPERIMENTS
from .progress import load_progress, export_progress, record_quiz, mark_complete
from visualization.embedding_graphs import matrix_figure, space_figure, curves_figure, ranking_figure, architecture_graph

SECTIONS = ['Concepts & mathematics', 'Tokenization & vocabulary', 'Matrix & one-hot',
            'Gradients & updates', 'Similarity & retrieval', 'Word embedding training',
            'Sentence pooling', 'Positions & context', 'Architecture builder',
            'Classification laboratory', 'Evaluation & experiments', 'Quiz & progress']


def _plot(figure, key, alt):
    st.plotly_chart(figure, key=key, width='stretch', alt=alt)


def _tensor(values, name, labels=None):
    values = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    st.caption(f'{name} — shape {values.shape}')
    if values.ndim > 2:
        st.json(values.tolist())
    else:
        frame = pd.DataFrame(values[None,:] if values.ndim == 1 else values, index=labels)
        st.dataframe(frame, alt=name)


def _seed(prefix):
    return int(st.number_input('Random seed', 0, 2147483647, 42, key=prefix+'_seed',persist_state='session'))


def _concepts():
    index = st.selectbox('Embedding concept', range(len(TOPICS)), format_func=lambda i:TOPICS[i][0], key='emb_topic', persist_state='session')
    title, intuition, formula, shapes, example, application, explorer = TOPICS[index]
    st.subheader(title); st.write(intuition); st.latex(formula)
    st.code(shapes); st.info('Worked example: '+example)
    st.write('Application: '+application)
    st.caption('Explore this live in '+explorer+'. '+EXPERIMENTS[index%len(EXPERIMENTS)][0])
    # Real PyTorch lookup is available for every concept, not a static placeholder.
    table = embedding_table(4, 2, values=[[0,0],[.1,.2],[1,2],[3,4]])
    ids = torch.tensor([[2,3,2,0]])
    r = lookup_demo(table, ids)
    st.code('table = torch.nn.Embedding(4, 2, padding_idx=0)\nids = torch.tensor([[2, 3, 2, 0]])\nvectors = table(ids)  # [1,4,2]', language='python')
    _tensor(r['lookup'], 'Live lookup [B,T,D]')
    _plot(matrix_figure(table.weight.detach().numpy(),'Example embedding table'), 'emb_concept_plot','Live embedding rows')
    st.caption('Embeddings supply numerical token representations for future attention, Transformers and LLMs. M8 does not implement attention.')


def _tokenization():
    text = st.text_area('Raw text', 'cat dog mystery', key='emb_text', max_chars=2000, persist_state='session')
    mode = st.selectbox('Tokenizer', ['word','character'], key='emb_tokenizer', persist_state='session')
    source = st.selectbox('Vocabulary source', ['Toy corpus','Custom corpus'], key='emb_vocab_source', persist_state='session')
    corpus_text = st.text_area('Corpus (one sentence per line)', 'cat dog pet\ndog puppy pet\napple orange fruit',key='emb_corpus',max_chars=20000, persist_state='session') if source == 'Custom corpus' else '\n'.join(TOY_CORPUS)
    length = st.slider('Padded / truncated length', 1, 50, 8, key='emb_pad_length', persist_state='session')
    dim = st.slider('Token embedding width', 1, 12, 3, key='emb_token_dim', persist_state='session')
    vocab, counts = build_vocabulary(corpus_text.splitlines(), mode)
    if len(vocab) > 500: st.warning('Use a corpus with at most 500 vocabulary entries.'); return
    tokens = tokenize(text, mode)
    ids = encode(tokens, vocab, length)
    st.graphviz_chart('digraph {rankdir=LR; "Raw text" -> "Tokens" -> "Vocabulary" -> "Token IDs" -> "Embedding lookup";}')
    st.write('Raw input:', text); st.write('Token sequence:', tokens); st.write('Token IDs:', ids.tolist())
    st.metric('Vocabulary size including PAD and UNK', len(vocab))
    st.dataframe(pd.DataFrame([dict(token=repr(token),ID=index,frequency=counts.get(token,0)) for token,index in vocab.items()]),alt='Vocabulary IDs and corpus frequencies')
    st.write('Unknown tokens:', [token for token in tokens if token not in vocab])
    st.caption(f'PAD=0, UNK=1. {max(0,length-len(tokens))} padding positions; {max(0,len(tokens)-length)} truncated tokens. Token IDs are categorical labels, not continuous quantities.')
    if counts:
        top = counts.most_common(40)
        _plot(ranking_figure([dict(token=repr(token),score=count) for token,count in top],'Token frequency'), 'emb_frequency','Corpus token frequency distribution')
    table = embedding_table(len(vocab), dim)
    _tensor(table(ids), 'Actual nn.Embedding output')
    _plot(matrix_figure(table(ids).detach().numpy(),'Token-to-vector sequence'), 'emb_token_vectors','Selected token vectors including padding')
    st.info('Word mode splits whitespace and retains punctuation; character mode retains spaces and usually creates longer sequences with a smaller alphabet. Real LLMs often use subwords to balance coverage and length. This offline lab is not a production tokenizer.')


def _matrix():
    v = st.slider('Vocabulary size', 2, 30, 8, key='emb_matrix_v', persist_state='session')
    d = st.slider('Embedding dimension', 1, 16, 3, key='emb_matrix_d', persist_state='session')
    seed = _seed('emb_matrix')
    init = st.selectbox('Weight initialization', ['normal','uniform','zeros'], key='emb_init', persist_state='session')
    padding = st.selectbox('Padding index', [None,*range(v)], index=1, key=f'emb_padding_{v}', persist_state='session')
    values = None
    if st.checkbox('Custom embedding values (JSON)',key='emb_custom', persist_state='session'):
        values = json.loads(st.text_area('Embedding matrix JSON',json.dumps(embedding_table(v,d,seed,init,padding).weight.detach().tolist()),key=f'emb_values_{v}_{d}', persist_state='session'))
    table = embedding_table(v,d,seed,init,padding,values)
    token = st.selectbox('Selected token ID', range(v), key=f'emb_selected_{v}', persist_state='session')
    raw = st.text_input('Sequence IDs (comma-separated)', f'{token},{token},{(token+1)%v}',key=f'emb_ids_{v}_{token}', persist_state='session')
    ids = torch.tensor([[int(part.strip()) for part in raw.split(',')]])
    r = lookup_demo(table, ids)
    st.latex(r'e=xE=E[\mathrm{id},:],\quad E:[V,D],\quad X:[B,T]\to[B,T,D]')
    st.metric('Embedding parameters V × D',table.weight.numel())
    _tensor(table.weight,'Embedding matrix E', [f'token_{i}' for i in range(v)])
    _tensor(table(torch.tensor([token])),'Selected row')
    _tensor(r['one_hot'],'One-hot input'); _tensor(r['lookup'],'Direct lookup'); _tensor(r['multiplication'],'One-hot × E')
    st.success(f'Max absolute lookup/multiplication difference: {(r["lookup"]-r["multiplication"]).abs().max().item():.8f}')
    _plot(matrix_figure(table.weight.detach().numpy(),'Embedding weights',[f'token_{i}' for i in range(v)]),'emb_matrix_heatmap','Embedding matrix heatmap')
    _plot(matrix_figure(r['one_hot'][0].numpy(),'One-hot sequence'),'emb_one_hot','Sparse one-hot rows')
    st.dataframe(pd.DataFrame([dict(representation='Dense float32 one-hot per token',bytes=4*v),dict(representation='Dense float32 vector per token',bytes=4*d),dict(representation='int64 ID per token',bytes=8),dict(representation='Full float32 embedding table',bytes=4*v*d)]),alt='Representation storage versus full model parameter storage')
    st.caption('Repeated IDs select identical rows, independent of surrounding tokens. PAD lookup gradients are suppressed; a supplied custom PAD value need not be zero. Sparse one-hot storage differs from the dense comparison above. Direct lookup avoids materializing large one-hot tensors.')


def _gradients():
    optimizer = st.selectbox('Update optimizer',['SGD','Adam'],key='emb_grad_optimizer', persist_state='session')
    lr = st.number_input('Update learning rate',.0001,1.,.1,.01,key='emb_grad_lr', persist_state='session')
    result = gradient_experiment(optimizer,lr,_seed('emb_grad'))
    st.code('optimizer.zero_grad()\nvectors = model.embedding(ids)\nlogits = model(ids)\nloss = torch.nn.functional.cross_entropy(logits, target)\nloss.backward()\noptimizer.step()',language='python')
    for key in ('ids','vectors','prediction','target','before','gradient','occurrence_gradients','after'):_tensor(result[key],key)
    st.write(f'Cross-entropy {result["loss"]:.8f} → {result["loss_after"]:.8f}. Used IDs: [2,2,3,0].')
    st.write(f'Central difference for E[2,0]: {result["numerical_gradient"]:.10f}; autograd: {result["gradient"][2,0].item():.10f}.')
    st.dataframe(pd.DataFrame(dict(row=range(6),gradient_norm=result['gradient'].norm(dim=1).numpy(),updated=(result['before']!=result['after']).any(dim=1).numpy())),alt='Embedding rows receiving gradients and updates')
    _plot(matrix_figure(result['gradient'].numpy(),'Embedding gradients'),'emb_gradient_heatmap','Used rows and zero unused/padding gradients')
    st.latex(r'E_{new}=E_{old}-\eta\nabla_E L' if optimizer=='SGD' else r'E_{new}=E_{old}-\eta\hat m/(\sqrt{\hat v}+\epsilon)')
    st.info('Row 2 adds two occurrence contributions. PAD row 0 and unused rows receive no table gradient in this lookup-only, no-weight-decay experiment. Adam is adaptive. Dense gradients allocate [V,D]; sparse=True is an alternative requiring compatible optimizers such as SparseAdam or supported SGD configurations, rather than ordinary Adam.')


def _available_vectors(prefix):
    available = ['Random toy table']
    if 'emb_word_run' in st.session_state: available.append('Trained skip-gram')
    source = st.selectbox('Vector source', available, key=prefix+'_source', persist_state='session')
    if source == 'Trained skip-gram':
        run = st.session_state.emb_word_run
        return run['vocabulary'], run['trained'].numpy(), 'Trained toy co-occurrence vectors; semantic quality is not established.'
    vocab, _ = build_vocabulary(TOY_CORPUS)
    table = embedding_table(len(vocab),4)
    return vocab, table.weight.detach().numpy(), 'Random untrained vectors; neighbors do not establish semantic relationships.'


def _similarity():
    vocab, vectors, message = _available_vectors('emb_sim')
    st.info(message)
    labels = list(vocab)
    a = st.selectbox('Vector A token',labels,index=2,key='emb_sim_a', persist_state='session')
    b = st.selectbox('Vector B token',labels,index=3,key='emb_sim_b', persist_state='session')
    raw_a = st.text_input('Edit vector A as JSON',json.dumps(vectors[vocab[a]].tolist()),key=f'emb_a_{a}_{vectors.shape[1]}_{message}', persist_state='session')
    raw_b = st.text_input('Edit vector B as JSON',json.dumps(vectors[vocab[b]].tolist()),key=f'emb_b_{b}_{vectors.shape[1]}_{message}', persist_state='session')
    va,vb = np.asarray(json.loads(raw_a),dtype=float),np.asarray(json.loads(raw_b),dtype=float)
    if st.checkbox('Normalize comparison vectors',key='emb_sim_normalize', persist_state='session'): va,vb = normalize_vectors(va),normalize_vectors(vb)
    results = similarity(va,vb)
    st.write('Vector norms:', float(np.linalg.norm(va)),float(np.linalg.norm(vb))); st.json(results)
    st.latex(r'a^Tb=\sum_i a_ib_i,\quad \cos(a,b)=\frac{a^Tb}{\|a\|\|b\|}')
    st.latex(r'd_2=\sqrt{\sum_i(a_i-b_i)^2},\quad d_1=\sum_i|a_i-b_i|')
    _plot(matrix_figure(np.stack([va,vb]),'Comparison vectors',['A','B']),'emb_compare_vectors','Editable vectors for metric comparison')
    st.caption('Cosine at zero is undefined mathematically; this lab uses 0 as an explicit safe convention. Smaller distance means closer; larger similarity means a better match under the chosen model and normalization assumptions.')
    st.subheader('Exact nearest-neighbor retrieval')
    st.graphviz_chart('digraph {rankdir=LR; "Query vector" -> "Compatible embeddings" -> "Similarity scores" -> "Ranked neighbors";}')
    query = st.selectbox('Query token',labels[2:],key='emb_query', persist_state='session')
    metric = st.selectbox('Retrieval metric',['cosine','dot'],key='emb_metric', persist_state='session')
    normalize = st.checkbox('Normalize retrieval vectors',key='emb_retrieval_normalize', persist_state='session')
    k = st.slider('Top K',1,min(15,len(vocab)-3),5,key='emb_topk', persist_state='session')
    ranking = nearest_neighbors(vectors[vocab[query]],vectors,k,metric,normalize,exclude=(0,1,vocab[query]))
    rows = [dict(token=labels[i],ID=i,score=score) for i,score in ranking]
    st.dataframe(pd.DataFrame(rows),alt='Exact nearest neighbors in original embedding space')
    _plot(ranking_figure(rows),'emb_ranking','Similarity score ranking')
    normalized = normalize_vectors(vectors)
    _plot(matrix_figure(normalized@normalized.T,'Cosine matrix',labels),'emb_similarity_matrix','All token cosine similarities')
    st.caption('Exact search scores every candidate. Approximate indexes trade recall for scale. Query and document embeddings must come from compatible models and dimensions. Dot and cosine can rank differently without normalization. No external vector database or RAG is used.')


def _config(prefix, word=False):
    dim = st.slider('Training embedding dimension',2,32,8,key=prefix+'_dim',persist_state='session')
    window = st.slider('Context window',1,4,2,key=prefix+'_window',persist_state='session') if word else 2
    negatives = st.slider('Negative samples per positive pair',1,10,3,key=prefix+'_negatives',persist_state='session') if word else 3
    lr = st.number_input('Training learning rate',.0001,1.,.03,.001,key=prefix+'_lr',persist_state='session')
    batch = st.selectbox('Mini-batch size',[16,32,64,128,256],index=1,key=prefix+'_batch',persist_state='session')
    epochs = st.slider('Training epochs',1,40,5,key=prefix+'_epochs',persist_state='session')
    seed = _seed(prefix)
    optimizer = st.selectbox('Training optimizer',['Adam','SGD'],key=prefix+'_optimizer',persist_state='session')
    return EmbeddingTrainingConfig(dim,window,negatives,lr,batch,epochs,seed,optimizer)


def _invalidate(prefix, signature):
    if st.session_state.get(prefix+'_signature') != signature:
        for suffix in ('_run','_test'): st.session_state.pop(prefix+suffix,None)
        st.session_state[prefix+'_signature'] = signature


def _spaces(run):
    labels = list(run['vocabulary'])[2:]
    initial,trained = run['initial'].numpy()[2:],run['trained'].numpy()[2:]
    selected = st.selectbox('Inspect word',labels,key='emb_space_token', persist_state='session')
    index = labels.index(selected)
    axes = st.selectbox('Projection dimensions',[2,3] if trained.shape[1]>=3 else [2],key='emb_space_axes', persist_state='session')
    methods = ['PCA','t-SNE']
    if trained.shape[1] == axes: methods.insert(0,'Direct')
    method = st.selectbox('Projection method',methods,key='emb_projection', persist_state='session')
    perplexity = st.slider('t-SNE perplexity',1,max(1,len(labels)-1),min(5,len(labels)-1),key='emb_perplexity', persist_state='session') if method=='t-SNE' else 5
    initial_ranking = nearest_neighbors(initial[index],initial,5,exclude=(index,))
    trained_ranking = nearest_neighbors(trained[index],trained,5,exclude=(index,))
    st.dataframe(pd.DataFrame([dict(stage=stage,token=labels[i],cosine=score) for stage,ranked in [('Initial',initial_ranking),('Trained',trained_ranking)] for i,score in ranked]),alt='Initial and trained nearest-neighbor comparison')
    # Project together so before/after coordinates share axes, including common PCA basis.
    coordinates = project_embeddings(np.concatenate([initial,trained]),axes,method,run['config'].seed,perplexity)
    for stage,points,ranking in [('Initial',coordinates[:len(labels)],initial_ranking),('Trained',coordinates[len(labels):],trained_ranking)]:
        _plot(space_figure(points,labels,index,[i for i,_ in ranking],stage+' '+method),'emb_space_'+stage,'Token coordinates and original-space neighbor links')
        _tensor(points,stage+' projected coordinates',labels)
    st.caption('Initial and trained tables use a joint projection for comparable axes. PCA retains maximum variance; t-SNE favors local neighborhoods and can distort global distances. Neighbor links and rankings use original vectors, not projected distances. t-SNE needs enough samples and perplexity below sample count.')


def _word_training():
    config = _config('emb_word',word=True)
    _invalidate('emb_word',config)
    st.write('Deterministic offline corpus with repeated co-occurrence patterns:')
    st.dataframe(pd.DataFrame({'sentence':list(dict.fromkeys(TOY_CORPUS))}),alt='Toy skip-gram corpus')
    st.latex(r'L=-\log\sigma(v_w^Tu_c)-\sum_{k=1}^K\log\sigma(-v_w^Tu_{n_k})')
    st.code('positive = (input(target)*context(word)).sum(-1)\nnegative = (input(target)[:,None,:]*context(negatives)).sum(-1)\nloss = -(F.logsigmoid(positive)+F.logsigmoid(-negative).sum(-1)).mean()',language='python')
    if st.button('Train word embeddings',key='emb_word_train'):
        with st.spinner('Training offline skip-gram...'): st.session_state.emb_word_run = train_skipgram(config)
    if 'emb_word_run' not in st.session_state: st.info('Train to compare initial and learned embeddings.');return
    run = st.session_state.emb_word_run
    st.metric('Two-table parameters 2 × V × D',run['parameter_count'])
    labels = list(run['vocabulary'])
    examples = [dict(target=labels[t],context=labels[c],negatives=[labels[n] for n in ns]) for (t,c),ns in zip(run['pairs'][:8].tolist(),run['negative_examples'].tolist())]
    st.dataframe(pd.DataFrame(examples),alt='Positive word/context pairs and sampled negatives')
    _plot(curves_figure(run['history'],['train_loss','fixed_negative_loss'],'Skip-gram objective'),'emb_word_loss','Observed epoch losses and fixed-negative evaluation')
    _tensor(run['initial'],'Initial input embeddings',labels);_tensor(run['trained'],'Trained input embeddings',labels)
    _tensor(run['model'].context.weight,'Trained output/context embedding table',labels)
    _plot(matrix_figure((run['trained']-run['initial']).numpy(),'Training changes',labels),'emb_word_delta','Initial to trained embedding coordinate changes')
    _spaces(run)
    st.info('Each target predicts nearby context words. Uniform negative samples exclude PAD, UNK, the target and every observed positive context for that target. Fixed seeds reproduce samples and runs. Neighbors reflect this small corpus, and falling loss alone does not prove semantic quality.')


def _sentence():
    vocab,vectors,message = _available_vectors('emb_sentence'); st.info(message)
    first = st.text_input('Sentence A','cat dog pet',key='emb_sentence_a', persist_state='session')
    second = st.text_input('Sentence B','pet dog cat',key='emb_sentence_b', persist_state='session')
    mode = st.selectbox('Pooling operation',['mean','sum','max'],key='emb_pooling', persist_state='session')
    length = st.slider('Sentence padded length',1,20,8,key='emb_sentence_length', persist_state='session')
    table = torch.tensor(vectors,dtype=torch.float32)
    pooled = []
    for name,text in [('A',first),('B',second)]:
        tokens = tokenize(text);ids = encode(tokens,vocab,length);embedded = table[ids]
        result = pool_embeddings(embedded[None],(ids!=0)[None],mode)[0]
        st.write(name+' tokens:',tokens);st.write(name+' padded IDs:',ids.tolist())
        _tensor(embedded,name+' individual token embeddings');_tensor(result,name+' sentence embedding')
        _plot(matrix_figure(embedded.numpy(),name+' token vectors'), 'emb_sentence_'+name,'Sentence tokens before masked pooling')
        pooled.append(result.numpy())
    st.json(similarity(*pooled))
    st.latex(r'\mathrm{mean}=\frac{\sum_t m_te_t}{\max(1,\sum_tm_t)},\quad \mathrm{sum}=\sum_tm_te_t,\quad \mathrm{max}_d=\max_{t:m_t=1}e_{td}')
    st.caption('PAD is masked for all operations. Empty/all-padding input returns a zero vector (including max); UNK counts as a real token. Mean, sum and max lose token order: reversing the same multiset yields identical pooled vectors. This is not context-sensitive encoding.')


def _positions():
    vocab,_ = build_vocabulary(['the weather is cool.','that movie is cool.'])
    text = st.text_input('Positional sentence','the weather is cool.',key='emb_pos_sentence', persist_state='session')
    reverse = st.checkbox('Reverse token order',key='emb_reverse', persist_state='session')
    length = st.slider('Position sequence length',1,16,6,key='emb_pos_length', persist_state='session')
    dim = st.slider('Position embedding dimension',1,16,5,key='emb_pos_dim', persist_state='session')
    positional = st.selectbox('Position representation',['learned','sinusoidal'],key='emb_pos_type', persist_state='session')
    seed = _seed('emb_pos')
    tokens = tokenize(text)
    if reverse:tokens = tokens[::-1]
    ids = encode(tokens,vocab,length)[None]
    spec = EmbeddingModelSpec(len(vocab),dim,positional=positional,max_length=length,seed=seed)
    model = EmbeddingAdapter().build(spec)
    r = model.representations(ids)
    st.write('Token IDs:',ids.tolist());st.write('Position IDs:',list(range(length)))
    st.latex(r'z_t=e_{w_t}+p_t')
    if positional == 'sinusoidal': st.latex(r'PE_{p,2i}=\sin(p/10000^{2i/D}),\quad PE_{p,2i+1}=\cos(p/10000^{2i/D})')
    for name in ('token','position','combined'):
        _tensor(r[name][0],name+' vectors')
        _plot(matrix_figure(r[name][0].detach().numpy(),name+' representations'),'emb_pos_'+name,name+' positional heatmap')
    left = st.selectbox('Compare position A',range(length),key=f'emb_position_a_{length}', persist_state='session')
    right = st.selectbox('Compare position B',range(length),index=min(1,length-1),key=f'emb_position_b_{length}', persist_state='session')
    st.json(similarity(r['position'][0,left].detach().numpy(),r['position'][0,right].detach().numpy()))
    st.subheader('Static versus contextual: cool')
    for sentence in ('the weather is cool.','that movie is cool.'):
        selected = st.text_input('Context sentence '+str(len(sentence)),sentence,key='emb_static_'+str(len(sentence)), persist_state='session')
        token_ids = encode(tokenize(selected),vocab)
        st.write('Tokens:',tokenize(selected))
        if vocab['cool.'] in token_ids.tolist(): _tensor(model.embedding(torch.tensor([vocab['cool.']])),'Static cool vector in '+selected)
        else: st.caption('Include cool. to compare the shared lookup row.')
    st.info('The same cool. ID selects the same static vector in both contexts. Position addition introduces local position information, with no cross-token mixing. Attention later enables context-dependent representations. Even mean-pooling token-plus-position sums can remain invariant to token permutation over fixed valid positions; position addition alone does not solve sentence order modeling.')


def _architecture():
    v = st.slider('Architecture vocabulary size',2,40,12,key='emb_arch_v', persist_state='session')
    dim = st.slider('Architecture embedding dimension',1,16,4,key='emb_arch_d', persist_state='session')
    classes = st.slider('Output classes',2,8,2,key='emb_arch_c', persist_state='session')
    batch = st.slider('Batch size B',1,4,2,key='emb_arch_b', persist_state='session')
    length = st.slider('Sequence length T',1,12,5,key='emb_arch_t', persist_state='session')
    maximum = st.slider('Maximum positional length',1,20,12,key='emb_arch_max', persist_state='session')
    padding = st.selectbox('Architecture padding index',[None,*range(v)],index=1,key=f'emb_arch_pad_{v}', persist_state='session')
    pooling = st.selectbox('Architecture pooling',['mean','sum','max'],key='emb_arch_pool', persist_state='session')
    positional = st.selectbox('Architecture position mode',['none','learned','sinusoidal'],key='emb_arch_pos', persist_state='session')
    spec = EmbeddingModelSpec(v,dim,classes,padding,pooling,positional,maximum,_seed('emb_arch'))
    rows = spec.infer_shapes(batch,length)
    st.graphviz_chart(architecture_graph(rows))
    st.dataframe(pd.DataFrame([{k:str(value) for k,value in row.items()} for row in rows]),alt='Inferred embedding architecture stage shapes, weights and counts')
    st.metric('All allocated trainable parameters',spec.parameter_count)
    st.caption('Embedding V×D + linear C×D+C + learned positions max_length×D when enabled. PAD occupies allocated parameters although lookup gradients are suppressed. Sinusoidal positions add no parameters.')
    model = EmbeddingAdapter().build(spec)
    ids = torch.arange(batch*length).reshape(batch,length)%v
    _tensor(ids,'Token-ID input')
    for name,values in model.representations(ids).items():_tensor(values,name)
    for name,param in model.named_parameters():st.write(f'{name}: {tuple(param.shape)}, {param.numel()} weights')
    st.download_button('Export embedding specification',json.dumps(spec.as_dict(),indent=2),file_name='embedding-spec.json',key='emb_spec_export')


def _classification():
    config = _config('emb_class')
    _invalidate('emb_class',config)
    st.caption('Unique offline adjective/noun sentences split first: 64 train, 17 validation, 17 test. Vocabulary fits train only. PAD=0, UNK=1, mini-batches, masked mean, linear logits and CrossEntropyLoss. Two held-out novel terms explicitly exercise UNK.')
    if st.button('Train embedding classifier',key='emb_class_train'):
        with st.spinner('Training classifier on CPU...'):st.session_state.emb_class_run = train_classifier(config)
    if 'emb_class_run' not in st.session_state:return
    run = st.session_state.emb_class_run
    _plot(curves_figure(run['history'],['train_loss','validation_loss'],'Classification loss'),'emb_class_loss','Training and validation cross-entropy')
    _plot(curves_figure(run['history'],['train_accuracy','validation_accuracy'],'Classification accuracy'),'emb_class_accuracy','Training and validation classification accuracy')
    st.dataframe(pd.DataFrame(run['history']),alt='Actual per-epoch train and validation metrics')
    st.metric('Classifier parameters',run['model'].spec.parameter_count)
    st.json(run['data']['vocabulary'])
    for name in ('initial','trained','gradient'):
        _tensor(run[name],name+' embedding table')
        _plot(matrix_figure(run[name].numpy(),name+' classifier embeddings'), 'emb_class_'+name,name+' classifier embedding heatmap')
    st.caption('Gradient is the last training mini-batch gradient. Hyperparameters should be chosen with validation, never by repeatedly selecting against the final test split.')
    if st.button('Evaluate held-out classifier test',key='emb_class_evaluate'):
        st.session_state.emb_class_test = evaluate_classifier_test(run)
    if 'emb_class_test' in st.session_state:
        result = st.session_state.emb_class_test
        st.write(f'Held-out test CE={result["loss"]:.6f}; accuracy={result["accuracy"]:.6f}')
        _tensor(result['confusion'],'Confusion matrix (rows=true, columns=predicted)')
        _plot(matrix_figure(result['confusion'].numpy(),'Held-out confusion matrix',['negative','positive']),'emb_class_confusion','Held-out classification confusion counts')
        st.dataframe(pd.DataFrame([dict(text=text,true=label,predicted=int(pred)) for (text,label),pred in zip(run['data']['raw']['test'],result['predictions'])]),alt='Held-out sample predictions')
    st.info('These metrics describe a tiny generated dataset, not general sentiment quality. Vocabulary never includes validation/test-only terms. OOV words use UNK; test results are not read during training. Changing any training control clears stale results.')


def _evaluation():
    st.write('Evaluate embeddings with multiple kinds of evidence. The displays below use completed experiments from this session.')
    st.dataframe(pd.DataFrame([
        dict(evidence='Training objective',interpretation='Optimization behavior; decreasing loss does not prove semantics'),
        dict(evidence='Nearest neighbors and retrieval',interpretation='Inspect original-space ranking and compatible model assumptions'),
        dict(evidence='Different random seeds',interpretation='Repeat runs and compare cosine neighbors; raw axes can rotate'),
        dict(evidence='PCA / t-SNE',interpretation='Partial projection, not a faithful global distance map'),
        dict(evidence='Held-out classification',interpretation='Task-specific generalization on this split only')]),alt='Embedding evaluation evidence and interpretation')
    if 'emb_word_run' in st.session_state:
        run = st.session_state.emb_word_run
        _spaces(run)
        st.write('Initial to trained Frobenius change:',float((run['trained']-run['initial']).norm()))
    if 'emb_class_test' in st.session_state:
        st.write('Last explicit test accuracy:',st.session_state.emb_class_test['accuracy'])
    st.info('Small corpora, rare tokens, OOV inputs, biased distributions, polysemy, domain shift, language coverage and dimension choices limit conclusions. Static embeddings collapse multiple senses. Analogy patterns are contingent, never guaranteed or hardcoded here.')
    for task,expected in EXPERIMENTS:st.write(task);st.caption('Expected observation: '+expected)
    st.caption('For sensitivity, repeat skip-gram with another seed using the same budget and compare neighbors. For retrieval, train first, select Trained skip-gram in Similarity & retrieval, then inspect rankings. Exact vector search is a foundation, not a complete semantic search/RAG system.')


def _progress():
    progress = st.session_state.learning_progress_v1
    answers = [st.radio(question,options,index=None,key=f'emb_quiz_{i}', persist_state='session') for i,(question,options,_,_) in enumerate(QUESTIONS)]
    if st.button('Grade embedding quiz',key='emb_grade'):
        if any(answer is None for answer in answers):st.warning('Answer every question first.')
        else:
            score = sum(answer==options[correct] for answer,(_,options,correct,_) in zip(answers,QUESTIONS))
            st.session_state.learning_progress_v1 = record_quiz(progress,'embedding',score,len(QUESTIONS))
            st.success(f'{score}/{len(QUESTIONS)} correct')
            for question,_,_,explanation in QUESTIONS:st.caption(question+' — '+explanation)
    notes = st.text_area('Embedding experiment notes',value=progress.get('exercise_notes',{}).get('embedding',''),key='emb_notes', persist_state='session')
    if st.button('Save embedding notes',key='emb_save_notes'):
        current = st.session_state.learning_progress_v1
        st.session_state.learning_progress_v1 = {**current,'exercise_notes':{**current.get('exercise_notes',{}),'embedding':notes}}
    if st.button('Mark embedding lesson complete',key='emb_complete'):
        st.session_state.learning_progress_v1 = mark_complete(st.session_state.learning_progress_v1,'embedding')
    st.download_button('Export learning progress JSON',export_progress(st.session_state.learning_progress_v1),file_name='learning-progress.json',key='emb_progress_export')
    upload = st.file_uploader('Import learning progress JSON',type='json',key='emb_progress_import')
    if upload and st.button('Load embedding progress',key='emb_load_progress'):
        st.session_state.learning_progress_v1 = load_progress(upload.getvalue());st.rerun()
    st.caption('Progress schema v1 preserves M4–M7 completion, scores and notes. Export to retain progress beyond this session.')


def render_embedding_learning_studio():
    st.subheader('Embedding Network')
    st.caption('Discrete tokens → real learned vectors → similarity, retrieval and classification.')
    for section, tab in render_learning_section_navigation(SECTIONS, 'emb').items():
        if not tab.open:
            continue
        with tab:
            renderers = [_concepts,_tokenization,_matrix,_gradients,_similarity,_word_training,_sentence,_positions,_architecture,_classification,_evaluation,_progress]
            try:renderers[SECTIONS.index(section)]()
            except (ValueError,TypeError,KeyError,IndexError,RuntimeError) as exc:
                st.warning(f'Check the experiment inputs: {exc}')
