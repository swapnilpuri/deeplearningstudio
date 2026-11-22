"""Presentation only for the recurrent lesson; all mathematics is in adapters."""
import json
import pandas as pd
import streamlit as st
from .navigation import render_learning_section_navigation
import torch
from .rnn_spec import RNNModelSpec
from .rnn_adapter import RNNAdapter, cell_trace, demo_cell, gradient_experiment, variable_length_demo
from .rnn_training import RNNTrainingConfig, sequence_splits, train_sequences, evaluate_sequence_test
from .rnn_content import TOPICS, QUESTIONS, EXPERIMENTS
from .progress import mark_complete, record_quiz, export_progress, load_progress
from visualization.rnn_graphs import state_heatmap, unrolled_figure, gradient_figure, prediction_figure, recurrent_architecture

SECTIONS = ['Concepts & mathematics','Cell & unrolling','Sequence tensors','Recurrent architecture',
            'Sequence datasets','Training laboratory','BPTT & clipping','Variable lengths','Quiz & progress']


def _tensor(value, title):
    st.caption(f'{title} — shape {tuple(value.shape)}')
    array = value.detach().cpu().numpy()
    st.dataframe(array.reshape(-1,array.shape[-1]) if array.ndim > 1 else array.reshape(1,-1),alt=title)


def _plot(figure, key, alt):
    st.plotly_chart(figure,width='stretch',key=key,alt=alt)


def _concepts():
    index = st.selectbox('Concept',range(len(TOPICS)),format_func=lambda i:TOPICS[i][0],key='rnn_topic', persist_state='session')
    title,explanation,formula,shapes,use,experiment = TOPICS[index]
    st.subheader(title);st.write(explanation);st.latex(formula)
    st.write('Tensor walkthrough: '+shapes)
    st.write('Practical use: '+use);st.info('Guided experiment: '+experiment)
    st.caption('Use the dedicated explorers for editable weights, stacked execution, training and padding. This live preview uses the same real PyTorch computations.')
    length = st.slider('Concept sequence length',2,30,8,key='rnn_concept_length', persist_state='session')
    gain = st.slider('Concept recurrent gain',.1,1.8,.5,.1,key='rnn_concept_gain', persist_state='session')
    activation = st.selectbox('Concept activation',['tanh','relu'],key='rnn_concept_activation', persist_state='session')
    result = gradient_experiment(length,gain,activation)
    _plot(state_heatmap(result['states'].numpy()),'rnn_concept_states','Live recurrent states')
    _plot(gradient_figure(result['state_gradients']),'rnn_concept_grad','Live gradient magnitudes through time')


def _move_step(delta, maximum):
    st.session_state.rnn_step = max(0,min(maximum,st.session_state.get('rnn_step',0)+delta))


def _cell():
    a,b,c = st.columns(3)
    i = a.slider('Input size',1,5,2,key='rnn_cell_i')
    h = b.slider('Hidden size',1,8,3,key='rnn_cell_h')
    t = c.slider('Sequence length',2,20,5,key='rnn_cell_t')
    activation = st.selectbox('Cell activation',['tanh','relu'],key='rnn_cell_activation', persist_state='session')
    seed = int(st.number_input('Cell seed',0,2147483647,42,key='rnn_cell_seed', persist_state='session'))
    scale = st.slider('Weight initialization scale',.0,1.5,.5,.05,key='rnn_cell_scale', persist_state='session')
    initial = st.number_input('Initial hidden value',value=0.,key='rnn_cell_initial', persist_state='session')
    x,wi,wh,bi,bh = demo_cell(i,h,t,seed,scale)
    x = torch.tensor(st.data_editor(pd.DataFrame(x.numpy()),key=f'rnn_input_{i}_{t}_{seed}',alt='Editable input sequence').to_numpy(),dtype=torch.float32)
    custom = st.checkbox('Edit weights and biases as JSON',key='rnn_edit_weights', persist_state='session')
    if custom:
        raw = st.text_area('Cell parameters JSON',json.dumps({'w_ih':wi.tolist(),'w_hh':wh.tolist(),'b_ih':bi.tolist(),'b_hh':bh.tolist()}),key=f'rnn_weights_{i}_{h}_{seed}_{scale}', persist_state='session')
        try:
            values = json.loads(raw)
            wi,wh,bi,bh = [torch.tensor(values[k],dtype=torch.float32) for k in ('w_ih','w_hh','b_ih','b_hh')]
        except (ValueError,TypeError,KeyError) as exc:
            st.warning(f'Invalid parameter JSON: {exc}');return
    try:
        states,steps = cell_trace(x,wi,wh,bi,bh,torch.full((h,),float(initial)),activation)
        reverse,_ = cell_trace(x.flip(0),wi,wh,bi,bh,torch.full((h,),float(initial)),activation)
    except (ValueError,RuntimeError,IndexError) as exc:
        st.warning(str(exc));return
    st.session_state.rnn_step = min(st.session_state.get('rnn_step',0),t-1)
    left,right = st.columns(2)
    left.button('Previous step',on_click=_move_step,args=(-1,t-1),key='rnn_previous')
    right.button('Next step',on_click=_move_step,args=(1,t-1),key='rnn_next')
    selected = st.slider('Selected time step (zero-based)',0,t-1,key='rnn_step', persist_state='session')
    _plot(unrolled_figure(x.numpy(),states.detach().numpy(),selected),'rnn_unrolled','Inputs and recurrent states with current step highlighted')
    _plot(state_heatmap(states.detach().numpy()),'rnn_cell_states','Hidden values across time')
    st.info('W_ih, W_hh and both biases are shared at every time step. Orange marks the current step; its left neighbor is the previous state. Hidden states are the cell outputs here; a task head is added in Architecture.')
    st.latex(r'h_t=f(W_{ih}x_t+b_{ih}+W_{hh}h_{t-1}+b_{hh})')
    step = steps[selected]
    left,right = st.columns(2)
    with left:
        st.code('input_part = W_ih @ x_t\nrecurrent_part = W_hh @ h_previous\nz = input_part + b_ih + recurrent_part + b_hh\nh = torch.tanh(z)  # or torch.relu(z)',language='python')
        for name in ('input','previous','w_ih','w_hh'):_tensor(step[name],name)
    with right:
        for name in ('input_contribution','recurrent_contribution','b_ih','b_hh','preactivation','hidden'):_tensor(step[name],name)
    unit = st.selectbox('Hidden unit arithmetic',range(h),key=f'rnn_arithmetic_{h}', persist_state='session')
    parts = [step[k][unit].item() for k in ('input_contribution','b_ih','recurrent_contribution','b_hh')]
    st.write(f'Unit {unit}: '+' + '.join(f'({v:.6f})' for v in parts)+f' = {step["preactivation"][unit].item():.6f}; {activation} → {states[selected,unit].item():.6f}')
    _tensor(states[-1],'Original final state');_tensor(reverse[-1],'Reversed-order final state')
    st.caption('With W_hh=0 each state processes its own input independently; a nonzero recurrent path carries history.')


def _architecture(tensor_only=False):
    prefix = 'rnn_tensor' if tensor_only else 'rnn_arch'
    b = st.slider('Batch size',1,8,2,key=prefix+'_b', persist_state='session')
    t = st.slider('Tensor sequence length',2,20,5,key=prefix+'_t', persist_state='session')
    i = st.slider('Feature count',1,5,2,key=prefix+'_i', persist_state='session')
    h = st.slider('Hidden dimensions',1,24,4,key=prefix+'_h', persist_state='session')
    layers = st.slider('Number of RNN layers',1,4,2,key=prefix+'_layers', persist_state='session')
    outputs = st.slider('Head output features',1,8,2,key=prefix+'_outputs', persist_state='session')
    batch_first = st.checkbox('batch_first',True,key=prefix+'_batch_first', persist_state='session')
    activation = st.selectbox('Recurrent activation',['tanh','relu'],key=prefix+'_activation', persist_state='session')
    selection = st.selectbox('Output selection',['final','sequence'],key=prefix+'_selection', persist_state='session')
    bias = st.checkbox('Use recurrent and head biases',True,key=prefix+'_bias', persist_state='session')
    seed = int(st.number_input('Architecture seed',0,2147483647,42,key=prefix+'_seed', persist_state='session'))
    spec = RNNModelSpec(i,h,layers,outputs,activation,selection,bias,batch_first,seed)
    x = torch.randn((b,t,i) if batch_first else (t,b,i),generator=torch.Generator().manual_seed(seed))
    model,trace = RNNAdapter().trace(spec,x)
    rows = spec.infer_shapes(b,t)
    st.graphviz_chart(recurrent_architecture(rows))
    st.dataframe(pd.DataFrame([{**row,'shape':str(row['shape']),'weights':str(row['weights'])} for row in rows]),alt='Inferred shapes and parameter counts')
    st.metric('Trainable parameters',spec.parameter_count)
    st.latex(r'P_l=H I_l+H^2+2H;\quad P_{head}=CH+C')
    st.caption('With bias enabled, PyTorch has b_ih and b_hh separately. I_l=I for layer 1, H afterwards. Remove 2H per layer and C in the head when bias is disabled. Length and batch size add no parameters.')
    with torch.no_grad():
        full,hn = model.encode(x)
    full_batch = full if batch_first else full.transpose(0,1)
    x_batch = x if batch_first else x.transpose(0,1)
    with torch.no_grad():
        sequence_logits = model.head(full_batch[0]) if selection == 'sequence' else None
    _plot(unrolled_figure(x_batch[0].numpy(),full_batch[0].numpy(),t-1,
                          None if sequence_logits is None else sequence_logits.numpy()),prefix+'_unrolled',
          'Unrolled top-layer states with per-step head outputs when sequence selection is enabled')
    st.write(f'nn.RNN output: {tuple(full.shape)}; h_n: {tuple(hn.shape)}')
    st.caption('batch_first=False swaps time and batch in input/output only; h_n is always [layers,batch,hidden]. h_n[-1] is the top layer final state. Padded inputs require final VALID states; see Variable lengths.')
    for step in trace:
        _tensor(step['tensor'],step['name'])
    for name,p in model.named_parameters():st.caption(f'{name}: {tuple(p.shape)}, {p.numel()} parameters')
    st.download_button('Export recurrent specification',json.dumps(spec.as_dict(),indent=2),file_name='rnn-spec.json',key=prefix+'_download')


def _data_config(prefix):
    task = st.selectbox('Sequence task',['classification','forecasting','memory'],key=prefix+'_task', persist_state='session')
    samples = st.slider('Samples',60,1200,240,60,key=prefix+'_samples', persist_state='session')
    length = st.slider('Data sequence length',2,40,8,key=prefix+'_length', persist_state='session')
    noise = st.slider('Sine noise',.0,.5,.05,.01,key=prefix+'_noise', persist_state='session')
    horizon = st.slider('Forecast horizon',1,10,1,key=prefix+'_horizon', persist_state='session')
    seed = int(st.number_input('Dataset seed',0,2147483647,42,key=prefix+'_seed', persist_state='session'))
    return {'task':task,'samples':samples,'sequence_length':length,'noise':noise,'horizon':horizon,'seed':seed}


@st.cache_data(max_entries=8)
def _cached_data(settings):
    return sequence_splits(**settings)


def _datasets():
    settings = _data_config('rnn_data');data = _cached_data(settings)
    st.write('Classification: label 1 when first < last; reversing the sequence flips the label. Memory: recall the sign of the first symbol. Forecasting: predict the next horizon values of a noisy sine source.')
    st.caption('Sine noise/horizon apply only to forecasting. All splits use 60% train, 20% validation and the remainder test. Distinct source-series IDs prevent overlapping windows across splits. Raw generated values need no fitted preprocessing.')
    st.table({part:len(data[part][0]) for part in ('train','validation','test')})
    example = st.slider('Training example',0,len(data['train'][0])-1,0,key='rnn_data_example', persist_state='session')
    x,y = data['train'];st.line_chart(pd.DataFrame({'input':x[example,:,0].numpy()}),alt='Selected training input sequence')
    st.write('Target:',y[example].tolist())
    st.json({part:data['source_series'][part][:5].tolist() for part in ('train','validation','test')})


def _training():
    settings = _data_config('rnn_train')
    epochs = st.slider('Epochs',1,40,8,key='rnn_train_epochs', persist_state='session')
    batch = st.selectbox('Batch size',[16,32,64,128],index=1,key='rnn_train_batch', persist_state='session')
    lr = st.number_input('Learning rate',.0001,1.,.01,.001,format='%.4f',key='rnn_train_lr', persist_state='session')
    hidden = st.slider('Training hidden size',2,32,12,key='rnn_train_hidden', persist_state='session')
    layers = st.slider('Training layers',1,4,1,key='rnn_train_layers', persist_state='session')
    activation = st.selectbox('Training activation',['tanh','relu'],key='rnn_train_activation', persist_state='session')
    optimizer = st.selectbox('Optimizer',['Adam','SGD'],key='rnn_train_optimizer', persist_state='session')
    clip = st.number_input('Gradient clip threshold (0 disables)',0.,100.,1.,.1,key='rnn_train_clip', persist_state='session')
    config = RNNTrainingConfig(**settings,epochs=epochs,batch_size=batch,learning_rate=lr,hidden_size=hidden,
                               num_layers=layers,optimizer=optimizer,clip=clip,activation=activation)
    if st.session_state.get('rnn_training_config') != config:
        st.session_state.rnn_training_config = config
        st.session_state.pop('rnn_run',None);st.session_state.pop('rnn_test',None)
    st.caption('Training uses final-state selection and its own settings, independent of the architecture explorer. CE + raw logits for classification; MSE for the full future vector. No test data is evaluated during fitting.')
    if st.button('Train RNN',key='rnn_train_button'):
        try:
            with st.spinner('Training recurrent model…'):
                st.session_state.rnn_run = train_sequences(config)
            st.session_state.pop('rnn_test',None)
        except ValueError as exc:st.error(str(exc))
    run = st.session_state.get('rnn_run')
    if run is None:return
    frame = pd.DataFrame(run['history']).set_index('epoch')
    st.line_chart(frame[['train_loss','validation_loss']],alt='Training and validation loss')
    st.caption('Metric: '+('MAE (lower is better)' if config.task=='forecasting' else 'accuracy (higher is better)'))
    st.line_chart(frame[['train_metric','validation_metric']],alt='Training and validation metrics')
    st.line_chart(frame[['gradient_norm','clipped_norm']],alt='Global parameter gradient norms before and after clipping')
    st.dataframe(pd.DataFrame({'parameter':list(run['parameter_changes']),'change_norm':list(run['parameter_changes'].values())}),alt='Parameter changes from initialization')
    _plot(state_heatmap(run['hidden'].numpy()),'rnn_train_hidden_states','Trained model hidden state progression for first training example')
    if st.button('Evaluate RNN held-out test',key='rnn_evaluate'):
        st.session_state.rnn_test = evaluate_sequence_test(run)
    result = st.session_state.get('rnn_test')
    if result is None:return
    st.metric('Held-out test loss',f'{result["loss"]:.5f}')
    st.metric('Held-out '+result['metric_name'],f'{result["metric"]:.5f}')
    if config.task=='forecasting':
        selected = st.slider('Held-out forecast example',0,len(result['target'])-1,0,key='rnn_forecast_example', persist_state='session')
        _plot(prediction_figure(result['target'][selected],result['prediction'][selected]),'rnn_test_prediction','Forecast and target over future horizon')
        st.caption('Largest forecast errors (regression has continuous errors, rather than discrete wrong labels).')
        st.dataframe(pd.DataFrame({'example':result['mistakes'].tolist(),'target':result['target'][result['mistakes']].tolist(),'prediction':result['prediction'][result['mistakes']].tolist()}),alt='Largest forecast errors')
    else:
        predicted = result['prediction'].argmax(-1)
        st.dataframe(pd.DataFrame({'target':result['target'].tolist(),'prediction':predicted.tolist()}),alt='Held-out sequence classifications')
        st.caption('Incorrect predictions')
        ids = result['mistakes']
        st.dataframe(pd.DataFrame({'example':ids.tolist(),'target':result['target'][ids].tolist(),'prediction':predicted[ids].tolist()}),alt='Incorrect sequence predictions')
    st.caption('Use validation to choose settings. Repeatedly tuning against test results invalidates their held-out interpretation.')


def _bptt():
    t = st.slider('BPTT sequence length',2,40,12,key='rnn_bptt_t', persist_state='session')
    gain = st.slider('Recurrent gain',.0,1.8,.5,.1,key='rnn_bptt_gain', persist_state='session')
    activation = st.selectbox('BPTT activation',['tanh','relu'],key='rnn_bptt_activation', persist_state='session')
    chunk = st.slider('Detach chunk size (0 = full BPTT)',0,t,0,key='rnn_bptt_chunk', persist_state='session')
    clip = st.number_input('Clipping threshold',.001,100.,1.,.1,key='rnn_bptt_clip', persist_state='session')
    lr = st.number_input('BPTT SGD learning rate',.0001,1.,.01,.001,key='rnn_bptt_lr', persist_state='session')
    result = gradient_experiment(t,gain,activation,chunk,clip,lr)
    st.write('Controlled scalar recurrence: h0=0.01, x_t=0.01, W_ih=1, no bias; the trainable recurrent gain is shared. Terminal loss L=h_T isolates the chain product. This diagnostic linear loss is intentionally different from the training task losses.')
    st.latex(r'\frac{\partial L}{\partial h_t}=\frac{\partial L}{\partial h_T}\prod_{k=t+1}^{T}f^{\prime}(z_k)W_{hh};\quad\frac{\partial L}{\partial W_{hh}}=\sum_t\frac{\partial L}{\partial z_t}h_{t-1}^{T}')
    st.code('optimizer.zero_grad()\n# unroll: retain_grad() on each state\n# at chunk boundaries: h = h.detach()\nloss = states[-1].sum()\nloss.backward()\ntorch.nn.utils.clip_grad_norm_(parameters, threshold)\noptimizer.step()',language='python')
    _plot(gradient_figure(result['state_gradients'],result['boundaries']),'rnn_bptt_gradients','State gradients with detach boundaries')
    _plot(state_heatmap(result['states'].numpy()),'rnn_bptt_states','Scalar recurrent state magnitudes')
    st.write(f'Loss: {result["loss"]:.6g}; norm before: {result["norm_before"]:.6g}; threshold: {clip}; norm after: {result["norm_after"]:.6g}')
    st.write(f'Parameter update: {result["before"].item():.6g} − {lr} × {result["gradient_after"].item():.6g} = {result["after"].item():.6g}')
    _tensor(result['input_gradients'],'dL/dx_t')
    st.dataframe(pd.DataFrame({'time_step':range(1,t+1),'local_shared_weight_gradient':result['weight_contributions']}),alt='Per-time-step contributions to shared recurrent weight gradient')
    st.caption(f'Sum of contributions: {sum(result["weight_contributions"]):.6g} = autograd dL/dW_hh: {result["gradient_before"].item():.6g} before clipping.')
    st.info('Try ReLU with gain 0.3 for vanishing and 1.5 for exploding gradients. Positive states keep ReLU derivatives at 1. Tanh can saturate even with gain >1. Clipping scales large parameter gradients; it does not repair vanished or detached paths. The state-gradient plot shows gradients before parameter clipping.')
    if result['boundaries']:
        last = result['boundaries'][-1]
        st.warning(f'Detach BEFORE zero-based steps {result["boundaries"]}. Final-only loss reaches inputs {last}..{t-1}; earlier inputs and states have zero/disconnected gradients. Forward values still carry context, and shared parameters still receive contributions from the last chunk.')


def _variable():
    raw = st.text_input('Valid lengths (comma-separated)','5,2,4',key='rnn_lengths', persist_state='session')
    hidden = st.slider('Packed hidden size',1,12,3,key='rnn_packed_hidden', persist_state='session')
    try:result = variable_length_demo(tuple(int(v.strip()) for v in raw.split(',')),hidden)
    except (ValueError,RuntimeError) as exc:st.warning(str(exc));return
    st.code('packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)\npacked_output, h_n = rnn(packed)\noutput, lengths = pad_packed_sequence(packed_output, batch_first=True)\nfinal_valid = h_n[-1]',language='python')
    _tensor(result['input'],'Zero-padded input [B,T,I]')
    st.dataframe(result['mask'].numpy(),alt='True means a valid sequence position')
    for key in ('wrong_padded_last','final_valid','packed_hidden'):_tensor(result[key],key)
    st.info('Zero padding still triggers recurrent updates and biases in ordinary execution. Packed sequences skip padding. h_n[-1] from the packed run equals the gathered final valid state; using padded output[:,-1] can be wrong.')
    st.caption('For a many-to-many supervised loss, select only mask=True positions before computing MSE or CE. Padding must not contribute to either the loss numerator or denominator. Training laboratory datasets are fixed-length and contain no padding.')


def _progress():
    progress = st.session_state.learning_progress_v1
    answers = [st.radio(question,options,index=None,key=f'rnn_quiz_{i}', persist_state='session') for i,(question,options,_,_) in enumerate(QUESTIONS)]
    if st.button('Grade RNN quiz',key='rnn_grade'):
        if any(a is None for a in answers):st.warning('Answer every question first.')
        else:
            score = sum(a==options[correct] for a,(_,options,correct,_) in zip(answers,QUESTIONS))
            st.session_state.learning_progress_v1 = record_quiz(progress,'rnn',score,len(QUESTIONS))
            st.success(f'{score}/{len(QUESTIONS)} correct')
            for question,_,_,explanation in QUESTIONS:st.caption(question+' — '+explanation)
    for task,expected in EXPERIMENTS:
        st.write(task);st.caption('Expected: '+expected)
    notes = st.text_area('RNN experiment notes',value=progress.get('exercise_notes',{}).get('rnn',''),key='rnn_notes', persist_state='session')
    if st.button('Save RNN notes',key='rnn_save_notes'):
        current = st.session_state.learning_progress_v1
        st.session_state.learning_progress_v1 = {**current,'exercise_notes':{**current.get('exercise_notes',{}),'rnn':notes}}
    if st.button('Mark RNN complete',key='rnn_complete'):
        st.session_state.learning_progress_v1 = mark_complete(st.session_state.learning_progress_v1,'rnn')
    st.download_button('Export progress JSON',export_progress(st.session_state.learning_progress_v1),file_name='learning-progress.json',key='rnn_progress_export')
    upload = st.file_uploader('Import M4/M5/M6 progress JSON',type='json',key='rnn_progress_import')
    if upload and st.button('Load RNN progress',key='rnn_load_progress'):
        try:st.session_state.learning_progress_v1 = load_progress(upload.getvalue());st.rerun()
        except (ValueError,TypeError) as exc:st.error(str(exc))
    st.caption('Progress is stored in this session. Export JSON to retain it across restarts; earlier lesson records are preserved.')


def render_rnn_learning_studio():
    st.subheader('Recurrent Neural Network (RNN)')
    st.caption('Explore time, shared parameters, hidden states and learning through real PyTorch execution.')
    for section, tab in render_learning_section_navigation(SECTIONS, 'rnn').items():
        if not tab.open:
            continue
        with tab:
            renderers = dict(zip(SECTIONS,[_concepts,_cell,lambda:_architecture(True),_architecture,_datasets,_training,_bptt,_variable,_progress]))
            renderers[section]()
