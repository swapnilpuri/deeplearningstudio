"""Presentation for LSTM; computations live in adapters and training modules."""
import json
import pandas as pd
import streamlit as st
from .navigation import render_learning_section_navigation
import torch
from .lstm_spec import LSTMModelSpec
from .lstm_adapter import GATES, LSTMAdapter, cell_trace, demo_cell, controlled_memory, gradient_experiment, variable_length_demo
from .lstm_training import LSTMTrainingConfig, lstm_splits, train_lstm, compare_models, dependency_sweep, evaluate_sequence_test
from .lstm_content import TOPICS, QUESTIONS, EXPERIMENTS
from .progress import mark_complete, record_quiz, export_progress, load_progress
from .rnn_ui import _tensor, _plot
from visualization.rnn_graphs import state_heatmap, unrolled_figure, recurrent_architecture, prediction_figure
from visualization.lstm_graphs import state_panels, gate_diagram

SECTIONS = ['Concepts & mathematics','Cell & unrolling','Gate & state laboratory','LSTM architecture',
            'Training laboratory','RNN versus LSTM','BPTT & clipping','Variable lengths','Quiz & progress']


def _concepts():
    index = st.selectbox('LSTM concept',range(len(TOPICS)),format_func=lambda i:TOPICS[i][0],key='lstm_topic', persist_state='session')
    title,explanation,formula,shapes,example,use,explorer = TOPICS[index]
    st.subheader(title);st.write(explanation);st.latex(formula)
    st.write('Dimensions: '+shapes);st.write('Numerical example: '+example)
    st.write('Practical use: '+use);st.info('Explore this in: '+explorer)
    st.graphviz_chart(gate_diagram())
    length = st.slider('Preview sequence length',2,50,12,key='lstm_preview_t', persist_state='session')
    forget = st.slider('Preview controlled forget value',0.,1.,.95,.01,key='lstm_preview_f', persist_state='session')
    _plot(state_panels(controlled_memory(length,forget)),'lstm_preview','Fixed-gate memory and state preview')
    st.caption('Fixed gate interventions isolate memory effects. The Cell explorer computes learned gates from x and previous h. ⊙ means element-wise multiplication.')


def _move_step(delta,maximum):
    st.session_state.lstm_step = max(0,min(maximum,st.session_state.get('lstm_step',0)+delta))


def _cell():
    a,b,c = st.columns(3)
    i = a.slider('Input features',1,5,2,key='lstm_cell_i')
    h = b.slider('Cell hidden size',1,8,3,key='lstm_cell_h')
    t = c.slider('Cell sequence length',2,20,6,key='lstm_cell_t')
    seed = int(st.number_input('Cell seed',0,2147483647,42,key='lstm_cell_seed', persist_state='session'))
    scale = st.slider('Weight scale',0.,1.5,.5,.05,key='lstm_cell_scale', persist_state='session')
    h0 = st.number_input('Initial hidden value',value=0.,key='lstm_cell_h0', persist_state='session')
    c0 = st.number_input('Initial cell value',value=0.,key='lstm_cell_c0', persist_state='session')
    x,wi,wh,bi,bh = demo_cell(i,h,t,seed,scale)
    x = torch.tensor(st.data_editor(pd.DataFrame(x.numpy()),key=f'lstm_x_{i}_{t}_{seed}',alt='Editable LSTM input sequence').to_numpy(),dtype=torch.float32)
    if st.checkbox('Edit LSTM weights, biases and initial vectors as JSON',key='lstm_edit', persist_state='session'):
        default = dict(w_ih=wi.tolist(),w_hh=wh.tolist(),b_ih=bi.tolist(),b_hh=bh.tolist(),h0=[h0]*h,c0=[c0]*h)
        raw = st.text_area('LSTM cell JSON',json.dumps(default),key=f'lstm_json_{i}_{h}_{seed}_{scale}', persist_state='session')
        try:
            parsed = json.loads(raw)
            wi,wh,bi,bh,initial_h,initial_c = [torch.tensor(parsed[k],dtype=torch.float32) for k in ('w_ih','w_hh','b_ih','b_hh','h0','c0')]
        except (ValueError,KeyError,TypeError) as exc:st.warning(f'Invalid JSON: {exc}');return
    else:initial_h,initial_c = torch.full((h,),h0),torch.full((h,),c0)
    try:values,steps = cell_trace(x,wi,wh,bi,bh,initial_h,initial_c)
    except (ValueError,RuntimeError,IndexError) as exc:st.warning(str(exc));return
    st.session_state.lstm_step = min(st.session_state.get('lstm_step',0),t-1)
    with st.container(horizontal=True):
        st.button('Previous step',on_click=_move_step,args=(-1,t-1),key='lstm_previous')
        st.button('Next step',on_click=_move_step,args=(1,t-1),key='lstm_next')
    selected = st.slider('Time step (zero-based)',0,t-1,key='lstm_step', persist_state='session')
    unit = st.selectbox('Selected unit',range(h),key=f'lstm_unit_{h}', persist_state='session')
    step = steps[selected]
    _plot(unrolled_figure(x.numpy(),values['hidden'].detach().numpy(),selected),'lstm_unroll','Unrolled hidden states sharing gate parameters')
    _plot(state_panels({**values,'input_vector':x},unit),'lstm_states','Computed inputs, four gates, cell and hidden states')
    st.caption('PyTorch packed order: i,f,g,o. W_ih=[4H,I], W_hh=[4H,H], b_ih=b_hh=[4H]. Both biases are trainable. All parameters are shared over time.')
    for name in ('input_vector','previous_h','previous_c'):_tensor(step[name],name)
    for j,gate in enumerate(GATES):
        st.subheader(gate.capitalize());lo,hi = j*h,(j+1)*h
        for name,value in [('W_ih',wi[lo:hi]),('W_hh',wh[lo:hi]),('b_ih',bi[lo:hi]),('b_hh',bh[lo:hi]),
                           ('input contribution',step['input_contribution'][lo:hi]),('recurrent contribution',step['recurrent_contribution'][lo:hi]),
                           ('preactivation',step['preactivation'][lo:hi]),('activation',step[gate])]:_tensor(value,gate+' '+name)
        parts = [step['input_contribution'][lo+unit],bi[lo+unit],step['recurrent_contribution'][lo+unit],bh[lo+unit]]
        activation = 'tanh' if gate=='candidate' else 'sigmoid'
        st.code(f'unit {unit}: '+' + '.join(f'({p.item():.6f})' for p in parts)+f' = {step["preactivation"][lo+unit].item():.6f}; {activation} → {step[gate][unit].item():.6f}')
    for name in ('retained','written','cell','hidden'):_tensor(step[name],name)
    st.code(f'c[{unit}] = {step["forget"][unit].item():.6f} × {step["previous_c"][unit].item():.6f} + '
            f'{step["input"][unit].item():.6f} × {step["candidate"][unit].item():.6f} = {step["cell"][unit].item():.6f}\n'
            f'h[{unit}] = {step["output"][unit].item():.6f} × tanh({step["cell"][unit].item():.6f}) = {step["hidden"][unit].item():.6f}')
    gate = st.selectbox('Gate heatmap',GATES,key='lstm_heatmap_gate', persist_state='session')
    _plot(state_heatmap(values[gate].detach().numpy(),gate),'lstm_gate_heatmap','Gate activations across units and time')
    st.line_chart(pd.DataFrame({gate:values[gate][:,unit].detach().numpy() for gate in GATES}),alt='Unit gate comparison over time')


def _gates():
    st.info('Controlled gate interventions: sliders replace learned functions to isolate retention, writing and exposure. Normal gates depend on x_t and h_previous.')
    t = st.slider('Memory sequence length',2,80,20,key='lstm_memory_t', persist_state='session')
    f = st.slider('Forget value',0.,1.,.95,.01,key='lstm_memory_f', persist_state='session')
    i = st.slider('Input gate value',0.,1.,.5,.01,key='lstm_memory_i', persist_state='session')
    o = st.slider('Output gate value',0.,1.,.8,.01,key='lstm_memory_o', persist_state='session')
    initial = st.number_input('Initial memory',value=1.,key='lstm_memory_initial', persist_state='session')
    reverse = st.checkbox('Reverse input order',key='lstm_memory_reverse', persist_state='session')
    result = controlled_memory(t,f,i,o,initial,reverse)
    st.caption('Known sequence: +1 first, -0.5 at the middle, zeros elsewhere; g=tanh(x). Reversing changes when writes occur.')
    _plot(state_panels(result),'lstm_memory_panels','Controlled inputs, gates, cell and hidden evolution')
    left,right = st.columns(2)
    with left:_plot(state_heatmap(result['cell'].numpy(),'Cell state'),'lstm_c_heatmap','Internal memory pathway')
    with right:_plot(state_heatmap(result['hidden'].numpy(),'Hidden state'),'lstm_h_heatmap','Exposed representation')
    st.line_chart(pd.DataFrame({'autograd dL/dc_t':result['cell_gradients'],'direct forget product':result['direct_path']}),alt='Direct cell pathway gradients with fixed gates')
    st.latex(r'\left.\frac{\partial c_T}{\partial c_t}\right|_{direct}=\prod_{k=t+1}^T f_k')
    st.write('Terminal loss L=c_T isolates the direct path with fixed gates. Learned LSTMs add hidden-state and gate paths. Both c and h propagate; they are generally unequal. Memory can accumulate; retention is not guaranteed.')
    st.caption('Try f=0 versus 0.99, i=0 versus 1, o=0 versus 1. Output changes h while leaving this controlled c unchanged. Clipping addresses exploding gradients, not vanishing gradients.')


def _architecture():
    b = st.slider('Batch size',1,8,2,key='lstm_arch_b', persist_state='session')
    t = st.slider('Tensor sequence length',2,20,5,key='lstm_arch_t', persist_state='session')
    i = st.slider('Feature count',1,5,2,key='lstm_arch_i', persist_state='session')
    h = st.slider('Hidden dimensions',1,24,4,key='lstm_arch_h', persist_state='session')
    layers = st.slider('Number of LSTM layers',1,4,2,key='lstm_arch_layers', persist_state='session')
    outputs = st.slider('Head outputs',1,10,2,key='lstm_arch_outputs', persist_state='session')
    first = st.checkbox('batch_first',True,key='lstm_arch_first', persist_state='session')
    selection = st.selectbox('Output selection',['final','sequence'],key='lstm_arch_selection', persist_state='session')
    bias = st.checkbox('Use biases',True,key='lstm_arch_bias', persist_state='session')
    dropout = st.slider('Between-layer dropout',0.,.8,0.,.05,key='lstm_arch_dropout', persist_state='session')
    try:spec = LSTMModelSpec(input_size=i,hidden_size=h,num_layers=layers,output_size=outputs,batch_first=first,selection=selection,bias=bias,dropout=dropout)
    except ValueError as exc:st.warning(str(exc));return
    x = torch.randn((b,t,i) if first else (t,b,i),generator=torch.Generator().manual_seed(spec.seed))
    model,trace = LSTMAdapter().trace(spec,x);rows = spec.infer_shapes(b,t)
    st.graphviz_chart(recurrent_architecture(rows))
    st.dataframe(pd.DataFrame([{k:str(v) for k,v in row.items()} for row in rows]),alt='Inferred shapes and parameter counts')
    st.metric('Trainable parameters',spec.parameter_count)
    st.latex(r'P_l=4HI_l+4H^2+8H;\quad P_{head}=CH+C')
    st.caption('Bias enabled: I_l=I at layer 1, H afterwards; b_ih and b_hh each [4H]. Without bias remove 8H per layer and C from the head. I=2,H=3 gives 84 recurrent parameters plus 8 for a two-class head. Sequence length adds no parameters.')
    with torch.no_grad():output,(hn,cn) = model.encode(x)
    st.write(f'Full output: {tuple(output.shape)}; final hidden: {tuple(hn.shape)}; final cell: {tuple(cn.shape)}')
    st.caption('batch_first swaps only input/output batch and time axes. Final states always use [layers,batch,hidden]. Output contains every top-layer step; final selection uses h_n[-1], while c_n stays internal. Traces use eval mode: dropout inactive. Sequence heads predict each position; training uses final-state heads.')
    for row in trace:_tensor(row['tensor'],row['name'])
    for name,p in model.named_parameters():st.caption(f'{name}: {tuple(p.shape)}; {p.numel()} parameters')
    gate = st.selectbox('Inspect packed gate slice',GATES,key='lstm_arch_gate', persist_state='session');index = GATES.index(gate)
    _tensor(model.rnn.weight_ih_l0[index*h:(index+1)*h],gate+' input matrix')
    _tensor(model.rnn.weight_hh_l0[index*h:(index+1)*h],gate+' recurrent matrix')
    st.download_button('Export LSTM specification',json.dumps(spec.as_dict(),indent=2),file_name='lstm-spec.json',key='lstm_spec_export')


def _config(prefix):
    task = st.selectbox('Sequence task',['memory','forecasting'],key=prefix+'_task', persist_state='session')
    samples = st.slider('Samples',60,1200,240,60,key=prefix+'_samples', persist_state='session')
    length = st.slider('Data sequence length',2,40,12,key=prefix+'_length', persist_state='session')
    distance = st.slider('Dependency distance',1,length-1,min(7,length-1),key=prefix+'_distance', persist_state='session') if length>2 else 1
    noise = st.slider('Noise',0.,.5,.05,.01,key=prefix+'_noise', persist_state='session')
    horizon = st.slider('Forecast horizon',1,10,1,key=prefix+'_horizon', persist_state='session')
    seed = int(st.number_input('Random seed',0,2147483647,42,key=prefix+'_seed', persist_state='session'))
    epochs = st.slider('Epochs',1,40,5,key=prefix+'_epochs', persist_state='session')
    batch = st.selectbox('Batch size',[16,32,64,128],index=1,key=prefix+'_batch', persist_state='session')
    lr = st.number_input('Learning rate',.0001,1.,.01,.001,format='%.4f',key=prefix+'_lr', persist_state='session')
    hidden = st.slider('Training hidden size',2,32,12,key=prefix+'_hidden', persist_state='session')
    layers = st.slider('Training layers',1,4,1,key=prefix+'_layers', persist_state='session')
    dropout = st.slider('Training inter-layer dropout',0.,.8,0.,.05,key=prefix+'_dropout', persist_state='session')
    optimizer = st.selectbox('Optimizer',['Adam','SGD'],key=prefix+'_optimizer', persist_state='session')
    clip = st.number_input('Clip threshold (0 disables)',0.,100.,1.,.1,key=prefix+'_clip', persist_state='session')
    return LSTMTrainingConfig(task=task,samples=samples,sequence_length=length,dependency_distance=distance,
        noise=noise,horizon=horizon,seed=seed,epochs=epochs,batch_size=batch,learning_rate=lr,
        hidden_size=hidden,num_layers=layers,dropout=dropout,optimizer=optimizer,clip=clip)


@st.cache_data(max_entries=8)
def _data(config):
    return lstm_splits(config)


def _run_summary(run,label):
    st.subheader(label);frame = pd.DataFrame(run['history']).set_index('epoch')
    st.line_chart(frame[['train_loss','validation_loss']],alt=label+' train and validation loss')
    st.line_chart(frame[['train_metric','validation_metric']],alt=label+' accuracy or forecasting MAE')
    st.line_chart(frame[['gradient_norm','clipped_norm']],alt=label+' gradients before and after clipping')
    st.write(f'Parameters: {run["spec"].parameter_count}; training seconds: {run["seconds"]:.4f}')
    st.dataframe(pd.DataFrame({'parameter':list(run['parameter_changes']),'update norm':list(run['parameter_changes'].values())}),alt=label+' parameter changes')
    x,y = run['data']['validation']
    with torch.no_grad():pred = run['model'](x)
    st.caption('Validation examples; metric is accuracy for memory, MAE for forecasting.')
    prediction = pred.argmax(-1) if run['config'].task=='memory' else pred
    st.dataframe(pd.DataFrame({'target':y.tolist(),'prediction':prediction.tolist()}),alt=label+' validation predictions')
    if run['config'].task=='memory':
        ids = torch.where(prediction!=y)[0]
        st.dataframe(pd.DataFrame({'example':ids.tolist(),'target':y[ids].tolist(),'prediction':prediction[ids].tolist()}),alt=label+' incorrect validation classifications')


def _test_summary(result,label):
    st.write(f'{label} held-out loss: {result["loss"]:.6f}; {result["metric_name"]}: {result["metric"]:.6f}')
    if result['metric_name']=='MAE':
        _plot(prediction_figure(result['target'][0],result['prediction'][0]),'lstm_test_'+label,label+' forecast versus target')
        pred = result['prediction']
    else:pred = result['prediction'].argmax(-1)
    st.dataframe(pd.DataFrame({'target':result['target'].tolist(),'prediction':pred.tolist()}),alt=label+' held-out predictions')
    ids = result['mistakes']
    st.dataframe(pd.DataFrame({'example':ids.tolist(),'target':result['target'][ids].tolist(),'prediction':pred[ids].tolist()}),alt=label+' mistakes or largest forecast errors')


def _training(comparison=False):
    prefix = 'lstm_compare' if comparison else 'lstm_train'
    # Invalidate old results before validation so invalid controls cannot revive them.
    try:config = _config(prefix)
    except ValueError as exc:
        for suffix in ('_run','_test','_sweep'):st.session_state.pop(prefix+suffix,None)
        st.warning(str(exc));return
    if st.session_state.get(prefix+'_config') != config:
        st.session_state[prefix+'_config'] = config
        for suffix in ('_run','_test','_sweep'):st.session_state.pop(prefix+suffix,None)
    data = _data(config)
    st.caption('Memory: recall the signal d steps before terminal prediction, amid independent distractors. Forecasting: independent sine sources prevent overlapping-window leakage across splits. 60/20/20 split; raw values need no fitted scaling. Noise affects the signal/series. Dependency distance is memory-only; horizon is forecast-only.')
    st.table({part:len(data[part][0]) for part in ('train','validation','test')})
    st.line_chart(pd.DataFrame({'input':data['train'][0][0,:,0].numpy()}),alt='First training input')
    st.write('First training target:',data['train'][1][0].tolist());st.json(config.__dict__)
    if comparison:st.info('Shared data/splits, epochs, batches, optimizer, learning rate, clipping, hidden width and layers. Equal width is not equal capacity: LSTM has more parameters and computation. RNN uses tanh; dropout is LSTM-only and reported. Outcomes depend on seed and budget; LSTM is not guaranteed to win.')
    if st.button('Train RNN and LSTM' if comparison else 'Train LSTM',key=prefix+'_button'):
        try:
            with st.spinner('Training sequence models…'):
                st.session_state[prefix+'_run'] = compare_models(config) if comparison else {'LSTM':train_lstm(config,data)}
            st.session_state.pop(prefix+'_test',None)
        except (ValueError,RuntimeError) as exc:st.error(str(exc))
    runs = st.session_state.get(prefix+'_run')
    if runs:
        if comparison:
            st.line_chart(pd.DataFrame({name+' '+metric:[r[metric] for r in run['history']] for name,run in runs.items() for metric in ('train_loss','validation_loss')}),alt='Measured RNN and LSTM loss comparison')
        for name,run in runs.items():_run_summary(run,name)
        if st.button('Evaluate held-out test',key=prefix+'_evaluate'):
            st.session_state[prefix+'_test'] = {name:evaluate_sequence_test(run) for name,run in runs.items()}
        for name,result in st.session_state.get(prefix+'_test',{}).items():_test_summary(result,name)
    if comparison and config.task=='memory':
        st.caption('Distance sweep retrains both models at three distances with this same budget; validation accuracy only.')
        if st.button('Measure dependency distances',key=prefix+'_sweep_button'):
            with st.spinner('Measuring dependency distances…'):
                st.session_state[prefix+'_sweep'] = dependency_sweep(config,sorted(set([1,config.sequence_length//2,config.sequence_length-1])))
        if prefix+'_sweep' in st.session_state:
            frame = pd.DataFrame(st.session_state[prefix+'_sweep']);st.dataframe(frame,alt='Measured metrics by dependency distance')
            st.line_chart(frame.pivot(index='distance',columns='model',values='validation_metric'),alt='Validation accuracy by dependency distance')
    st.caption('No test evaluation during fitting. Repeated test-based tuning compromises held-out interpretation. Tiny smoke results establish workflows, not model-quality guarantees.')


def _bptt():
    length = st.slider('BPTT sequence length',2,60,12,key='lstm_bptt_t', persist_state='session')
    bias = st.slider('Forget bias initialization',-6.,6.,2.,.25,key='lstm_bptt_f', persist_state='session')
    scale = st.slider('BPTT weight scale',0.,1.5,.3,.05,key='lstm_bptt_scale', persist_state='session')
    clip = st.number_input('Gradient clipping threshold',.0001,100.,1.,.1,key='lstm_bptt_clip', persist_state='session')
    lr = st.number_input('Update learning rate',.0001,1.,.01,.001,key='lstm_bptt_lr', persist_state='session')
    optimizer = st.selectbox('Update optimizer',['SGD','Adam'],key='lstm_bptt_optimizer', persist_state='session')
    result = gradient_experiment(length,bias,scale,clip,lr,optimizer)
    st.code('optimizer.zero_grad()\nvalues, steps = cell_trace(x, *parameters)\n# retain_grad() on cell and hidden states\nprediction = steps[-1]["hidden"].sum()\nloss = (prediction - 1).square()\nloss.backward()\nclip_grad_norm_(parameters, threshold)\noptimizer.step()',language='python')
    st.write(f'Regression output: {result["prediction"]:.6f}; target: 1. Loss before: {result["loss"]:.6f}; after update: {result["loss_after"]:.6f}.')
    st.line_chart(pd.DataFrame(result['gradients']),alt='Cell and hidden state gradient norms before clipping')
    st.dataframe(pd.DataFrame(result['gate_norms']),alt='Per-gate gradient norms for packed weights and biases')
    st.write(f'Gradient norm before: {result["norm_before"]:.6f}; after: {result["norm_after"]:.6f}; threshold: {clip}.')
    _plot(state_panels({**result['values'],'input_vector':demo_cell(1,2,length,scale=scale)[0]}),'lstm_bptt_states','Forward gates and state updates')
    selected = st.selectbox('Parameter tensor',range(4),format_func=lambda k:('W_ih','W_hh','b_ih','b_hh')[k],key='lstm_bptt_parameter', persist_state='session')
    for name,value in [('Before',result['before'][selected]),('Raw gradient',result['raw_gradients'][selected]),('Clipped gradient',result['clipped_gradients'][selected]),('After',result['after'][selected])]:_tensor(value,name)
    if optimizer=='SGD':
        st.latex(r'\theta_{new}=\theta_{old}-\eta\,g_{clipped}')
        st.caption('Exact plain-SGD update, no momentum/weight decay; verified numerically in tests.')
    else:
        st.latex(r'\theta_{new}=\theta_{old}-\eta\frac{\hat m}{\sqrt{\hat v}+\epsilon}')
        st.caption('Adam uses bias-corrected first/second gradient moments. The after tensor is its actual adaptive update.')
    _tensor(result['input_gradients'],'Input gradients across time')
    st.info('Direct cell gradients multiply forget values; full autograd adds hidden and gate paths. Compare short/long sequences and different initializations. Gates can saturate and memory can vanish. Clipping limits exploding parameter gradients and cannot restore vanished signals. State gradients precede clipping.')


def _variable():
    raw = st.text_input('Valid lengths (comma-separated)','5,2,4',key='lstm_lengths', persist_state='session')
    h = st.slider('Packed hidden size',1,12,3,key='lstm_packed_h', persist_state='session')
    try:result = variable_length_demo(tuple(int(v.strip()) for v in raw.split(',')),h)
    except (ValueError,RuntimeError) as exc:st.warning(str(exc));return
    st.code('packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)\npacked_output, (h_n, c_n) = lstm(packed)\noutput, lengths = pad_packed_sequence(packed_output, batch_first=True)\nfinal_valid_hidden = h_n[-1]\n# masks exclude padded targets from supervised loss',language='python')
    for name in ('input','mask','output','hidden','cell','final_valid','wrong_padded_last'):_tensor(result[name],name)
    st.caption('Packed final h_n and c_n: [layers,batch,hidden]. Final valid output equals h_n[-1]; c_n is separate memory. Ordinary zero padding still updates gates and states. Masked MSE reuses M6; tests poison padding and check zero gradients there. Training datasets are fixed-length.')


def _progress():
    progress = st.session_state.learning_progress_v1
    answers = [st.radio(q,options,index=None,key=f'lstm_quiz_{i}', persist_state='session') for i,(q,options,_,_) in enumerate(QUESTIONS)]
    if st.button('Grade LSTM quiz',key='lstm_grade'):
        if any(a is None for a in answers):st.warning('Answer every question first.')
        else:
            score = sum(a==options[correct] for a,(_,options,correct,_) in zip(answers,QUESTIONS))
            st.session_state.learning_progress_v1 = record_quiz(progress,'lstm',score,len(QUESTIONS))
            st.success(f'{score}/{len(QUESTIONS)} correct')
            for question,_,_,explanation in QUESTIONS:st.caption(question+' — '+explanation)
    for task,expected in EXPERIMENTS:st.write(task);st.caption('Expected: '+expected)
    notes = st.text_area('LSTM experiment notes',value=progress.get('exercise_notes',{}).get('lstm',''),key='lstm_notes', persist_state='session')
    if st.button('Save LSTM notes',key='lstm_save_notes'):
        current = st.session_state.learning_progress_v1
        st.session_state.learning_progress_v1 = {**current,'exercise_notes':{**current.get('exercise_notes',{}),'lstm':notes}}
    if st.button('Mark LSTM complete',key='lstm_complete'):
        st.session_state.learning_progress_v1 = mark_complete(st.session_state.learning_progress_v1,'lstm')
    st.download_button('Export progress JSON',export_progress(st.session_state.learning_progress_v1),file_name='learning-progress.json',key='lstm_progress_export')
    upload = st.file_uploader('Import learning progress JSON',type='json',key='lstm_progress_import')
    if upload and st.button('Load LSTM progress',key='lstm_load_progress'):
        try:st.session_state.learning_progress_v1 = load_progress(upload.getvalue());st.rerun()
        except (ValueError,TypeError) as exc:st.error(str(exc))
    st.caption('Schema v1 preserves M4/M5/M6 records. Export to retain progress after the session.')


def render_lstm_learning_studio():
    st.subheader('Long Short-Term Memory (LSTM)')
    st.caption('Learn gated memory through PyTorch calculations, training and measured comparisons.')
    for section, tab in render_learning_section_navigation(SECTIONS, 'lstm').items():
        if not tab.open:
            continue
        with tab:
            renderers = [_concepts,_cell,_gates,_architecture,_training,lambda:_training(True),_bptt,_variable,_progress]
            renderers[SECTIONS.index(section)]()
