"""CNN Learning Studio tabs, powered only by the computation modules."""
import json
import numpy as np
import plotly.express as px
import streamlit as st
import torch
from .cnn_spec import CNNLayerSpec, CNNModelSpec, digits_spec, output_axis
from .cnn_adapter import CNNAdapter, convolution_explorer, pooling_explorer, filter_presets, single_sgd_step, convolution_gradient_check
from .cnn_content import TOPICS, EXPERIMENTS
from .cnn_training import CNNTrainingConfig, train_digits, evaluate_test
from .progress import empty_progress, mark_complete, record_quiz, export_progress, load_progress
from visualization.cnn_graphs import heatmap, architecture_diagram

QUESTIONS = [
 ('A 3-channel input and 4 Conv2d 3×3 filters with bias have how many parameters?', ['108','112','36'],1,'4 × (3 × 3 × 3 + 1) = 112.'),
 ('Which dimensions are used for PyTorch images?', ['NHWC','NCHW','CHWN'],1,'Batch, channels, height, width.'),
 ('Does Conv2d flip its kernel?', ['Yes','No, it uses cross-correlation'],1,'The weights are used as written.'),
 ('8×8 input, 3×3 kernel, stride=1, padding=1 produces?', ['6×6','8×8','10×10'],1,'floor((8+2−3)/1+1)=8.'),
 ('2×2 max pooling with stride 2 on 8×8 produces?', ['4×4','8×8','2×2'],0,'A nonoverlapping 2×2 reduction halves each spatial axis.'),
 ('What does backward() do?', ['Updates weights','Computes derivatives','Resets gradients'],1,'The optimizer updates weights after autograd computes gradients.'),
 ('Three input channels and eight filters produce how many output channels?', ['3','8','24'],1,'Each filter sums across all input channels and emits one channel.'),
 ('Increasing stride usually does what?', ['Adds weights','Reduces spatial output','Adds channels'],1,'Larger steps give fewer window positions, without changing weights.'),
 ('Flatten [N,16,2,2] produces?', ['[N,16]','[N,64]','[64,N]'],1,'16×2×2=64 features per sample.'),
 ('CrossEntropyLoss expects?', ['Raw logits and integer labels','Softmax probabilities only','One-hot labels only'],0,'Use [N,C] logits and [N] integer class indices.'),
 ('A 3×3 kernel with dilation 2 spans?', ['3×3','5×5','6×6'],1,'D(K−1)+1 = 5, with nine weights per channel.'),
 ('Padding 1, stride 1, 3×3 convolution on 8×8 gives?', ['6×6','8×8','9×9'],1,'The zero border preserves size in this case.'),
]


def _position(label, count, key):
    if count == 1:
        st.caption(f'{label}: 0 (only position)')
        return 0
    return st.slider(label,0,count-1,0,key=f'{key}_{count}')


def _progress_panel(progress):
    st.subheader('Quiz & guided experiments')
    answers=[]
    for i,(question,options,_,_) in enumerate(QUESTIONS):
        answers.append(st.radio(question,options,index=None,key=f'cnn_question_{i}'))
    if st.button('Grade CNN quiz'):
        if any(a is None for a in answers): st.warning('Answer all questions first.')
        else:
            score=sum(answer==options[correct] for answer,(_,options,correct,_) in zip(answers,QUESTIONS))
            st.session_state.learning_progress_v1=record_quiz(progress,'cnn',score,len(QUESTIONS))
            st.success(f'Score: {score}/{len(QUESTIONS)}')
            for question,options,correct,explanation in QUESTIONS:st.caption(f'{question} — {explanation}')
    for task,expected in EXPERIMENTS:
        st.write(task)
        st.caption('Expected observation: '+expected)
    notes=st.text_area('Your observations',value=progress.get('exercise_notes',{}).get('cnn',''),key='cnn_notes')
    if st.button('Save CNN notes'):
        st.session_state.learning_progress_v1={**progress,'exercise_notes':{**progress.get('exercise_notes',{}),'cnn':notes}}
        st.success('Saved in session; export to persist.')
    if st.button('Mark CNN lesson complete'):
        st.session_state.learning_progress_v1=mark_complete(st.session_state.learning_progress_v1,'cnn')
    st.download_button('Export progress JSON',export_progress(st.session_state.learning_progress_v1),file_name='learning-progress.json')
    upload=st.file_uploader('Import v4 or v5 progress',type='json',key='cnn_progress_upload')
    if upload and st.button('Import CNN progress'):
        try:
            st.session_state.learning_progress_v1=load_progress(upload.getvalue());st.rerun()
        except (ValueError,TypeError,json.JSONDecodeError) as exc:st.error(f'Invalid progress: {exc}')


def _convolution_tab():
    st.markdown('**Conv2d uses cross-correlation (kernel not flipped).** Move the output position to see the exact window and its products.')
    c1,c2,c3,c4=st.columns(4)
    with c1:height=st.slider('Image height',4,12,6,key='cnn_img_height')
    with c2:channels=st.selectbox('Input channels',[1,3],key='cnn_in_channels')
    with c3:k=st.slider('Kernel size',1,5,3,key='cnn_kernel')
    with c4:nfilters=st.slider('Output filters',1,8,2,key='cnn_filters')
    width=st.slider('Image width',4,12,6,key='cnn_img_width')
    kw=st.slider('Kernel width',1,5,k,key='cnn_kernel_width')
    dilation=st.slider('Dilation',1,3,1,key='cnn_dilation')
    c1,c2,c3,c4=st.columns(4)
    with c1:stride=st.slider('Stride',1,3,1,key='cnn_stride')
    with c2:padding=st.slider('Padding',0,3,1,key='cnn_padding')
    with c3:use_bias=st.checkbox('Use bias',True,key='cnn_use_bias')
    with c4:activation=st.selectbox('Activation',['None','ReLU'],key='cnn_activation')
    bias_value=st.number_input('Bias per filter',value=0.0,key='cnn_bias_value')
    preset=st.selectbox('Kernel', ['Vertical edge','Horizontal edge','Identity','Blur','Sharpen','Custom seeded'],key='cnn_filter_preset')
    seed=st.number_input('Explorer seed',min_value=0,max_value=100000,value=42,key='cnn_explorer_seed')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        x=torch.arange(1,height*width*channels+1,dtype=torch.float32).reshape(channels,height,width)/height
        weights=torch.randn(nfilters,channels,k,kw)*.25
        if k==kw==3 and preset!='Custom seeded':
            weights[:]=filter_presets(preset,channels)
        bias=torch.full((nfilters,),float(bias_value)) if use_bias else None
    signature=(int(seed),preset,k,kw,channels,nfilters)
    if st.session_state.get('cnn_weight_signature')!=signature:
        st.session_state.cnn_weight_signature=signature
        st.session_state.cnn_weight_edits={}
    for (f,c),kernel in st.session_state.cnn_weight_edits.items():
        weights[f,c]=kernel
    if preset!='Custom seeded' and (k,kw)!=(3,3):
        st.caption('Named presets require 3×3 kernels; seeded weights are shown for other sizes.')
    edit_filter=st.selectbox('Filter weights to edit',list(range(nfilters)),key='cnn_edit_filter')
    edit_channel=st.selectbox('Kernel input channel to edit',list(range(channels)),key='cnn_edit_channel')
    edited=st.data_editor(weights[edit_filter,edit_channel].numpy(),key=f'cnn_weights_{seed}_{preset}_{k}_{kw}_{channels}_{nfilters}_{edit_filter}_{edit_channel}')
    if not np.isfinite(edited).all():
        st.warning('Kernel weights must be finite.')
        return
    weights[edit_filter,edit_channel]=torch.as_tensor(np.asarray(edited),dtype=torch.float32)
    st.session_state.cnn_weight_edits[(edit_filter,edit_channel)]=weights[edit_filter,edit_channel].clone()
    try:
        out_shape=output_axis(height,k,stride,padding,dilation)
        out_width=output_axis(width,kw,stride,padding,dilation)
        if out_shape<1:raise ValueError('Kernel larger than padded input')
        ci,cf=st.columns(2)
        with ci:fil=st.selectbox('Output filter index',list(range(nfilters)),key='cnn_filter_index')
        with cf:channel=st.selectbox('Input channel to inspect',list(range(channels)),key='cnn_channel_index')
        row=_position('Output row',out_shape,'cnn_conv_row')
        col=_position('Output column',out_width,'cnn_conv_col')
        result=convolution_explorer(x,weights,bias,stride,padding,activation,row,col,fil,dilation)
        st.latex(r'H_{out}=\lfloor(H+2P-D(K-1)-1)/S+1\rfloor')
        st.write(f'Input {tuple(x[None].shape)} · Weights {tuple(weights.shape)} · Output {tuple(result["output"][None].shape)}')
        st.caption(f'Window begins at unpadded coordinates {result["window_origin"]}; green highlight corresponds to the patch (including padding when outside image).')
        left,right=st.columns(2)
        with left:
            st.plotly_chart(heatmap(x[channel].numpy(),'Input channel (un-padded)',highlight=(*result['window_origin'],*result['window_extent']),positions=result['sample_positions']),width='stretch')
            st.plotly_chart(heatmap(result['patch'][channel].numpy(),'Selected padded window'),width="stretch")
        with right:
            st.plotly_chart(heatmap(weights[fil,channel].numpy(),'Filter weights'),width="stretch")
            st.plotly_chart(heatmap(result['products'][channel].numpy(),'Window × kernel'),width="stretch")
        st.write('Per-channel contributions',result['channel_contributions'].tolist())
        st.write(f'Sum of products {result["sum_products"]:.5g} + bias {result["bias"]:.5g} = {result["linear_value"]:.5g}; after {activation}: {result["activated_value"]:.5g}')
        views=st.columns(min(4,nfilters))
        for f in range(nfilters):
            with views[f%len(views)]:
                st.plotly_chart(heatmap(result['output'][f].numpy(),f'Filter {f} pre-activation',highlight=(row,col,1,1)),width="stretch")
                if activation=='ReLU':st.plotly_chart(heatmap(result['activated'][f].numpy(),f'Filter {f} post-ReLU'),width="stretch")
    except (ValueError,RuntimeError) as exc:st.warning(str(exc))


def _pool_tab():
    st.write('Pooling operates independently on each feature-map channel. This example uses one 2D feature map.')
    kind=st.selectbox('Pooling mode',['max','average'],key='cnn_pool_kind')
    size=st.slider('Feature-map width/height',4,12,6,key='cnn_pool_size')
    k=st.slider('Pool kernel',1,4,2,key='cnn_pool_k')
    stride=st.slider('Pool stride',1,4,2,key='cnn_pool_stride')
    fmap=torch.arange(1,size*size+1,dtype=torch.float32).reshape(size,size)
    try:
        dims=output_axis(size,k,stride)
        if dims<1:raise ValueError('Pooling kernel too large')
        row=_position('Pool output row',dims,'cnn_pool_row')
        col=_position('Pool output column',dims,'cnn_pool_col')
        result=pooling_explorer(fmap,kind,k,stride,row,col)
        a,b=st.columns(2)
        with a:st.plotly_chart(heatmap(fmap.numpy(),'Feature map',highlight=(*result['origin'],k,k)),width="stretch")
        with b:st.plotly_chart(heatmap(result['output'].numpy(),f'{kind.title()} pooling output',highlight=(row,col,1,1)),width="stretch")
        st.write(f'Window {result["window"].tolist()} → {kind} = {result["value"]:.4g}')
        st.latex(r'H_{out}=\lfloor(H-K)/S+1\rfloor')
        st.caption('For this formula: zero padding, dilation=1 and ceil_mode=False. Pooling has no trainable weights.')
    except (ValueError,RuntimeError) as exc:st.warning(str(exc))


def _architecture_tab():
    st.write('Construct convolution blocks and a classification head. The dense input size is inferred from tensor shapes.')
    size=st.selectbox('Square image size',[8,12,16,28],key='cnn_arch_size')
    channels=st.selectbox('Input channels',[1,3],key='cnn_arch_channels')
    blocks=st.slider('Convolution blocks',1,3,2,key='cnn_arch_blocks')
    filters=[st.slider(f'Block {i+1} output filters',1,32,8*(i+1),key=f'cnn_arch_f{i}') for i in range(blocks)]
    hidden=st.multiselect('Dense hidden widths (in selected order)',[16,32,64,128],key='cnn_arch_hidden')
    k=st.slider('Convolution kernel',1,5,3,key='cnn_arch_kernel')
    pad=st.slider('Conv padding',0,2,1,key='cnn_arch_pad')
    pool='maxpool2d' if st.selectbox('Pooling',['maxpool2d','avgpool2d'],key='cnn_arch_pool')=='maxpool2d' else 'avgpool2d'
    classes=st.slider('Class logits',2,20,10,key='cnn_arch_classes')
    try:
        layers=[]
        for filters_out in filters:
            layers.extend([CNNLayerSpec('conv2d',out_channels=filters_out,kernel_size=(k,k),padding=(pad,pad)),CNNLayerSpec('relu'),CNNLayerSpec(pool,kernel_size=(2,2),stride=(2,2))])
        layers.append(CNNLayerSpec('flatten'))
        for units in hidden:
            layers.extend([CNNLayerSpec('linear',out_features=units),CNNLayerSpec('relu')])
        layers.append(CNNLayerSpec('linear',out_features=classes))
        spec=CNNModelSpec(channels,size,size,tuple(layers))
        rows=spec.infer_shapes()
        st.plotly_chart(architecture_diagram(rows),width="stretch")
        st.dataframe(rows,hide_index=True,width="stretch")
        st.metric('Trainable parameters',sum(row['parameters'] for row in rows))
        _,steps=CNNAdapter().trace(spec)
        st.write('Actual PyTorch layer outputs')
        st.dataframe([{'operator':s['operator'],'shape':s['shape']} for s in steps],hide_index=True)
        selected=st.selectbox('Inspect forward tensor',list(range(len(steps))),format_func=lambda i:f'{i}: {steps[i]["operator"]}',key='cnn_trace_step')
        tensor=steps[selected]['tensor'][0]
        if tensor.ndim==3:
            channel=st.selectbox('Trace channel',list(range(tensor.shape[0])),key='cnn_trace_channel')
            st.plotly_chart(heatmap(tensor[channel].numpy(),'Actual forward values'),width='stretch')
        else:
            st.write('Forward values',tensor.tolist())
        st.write('Final classification logits',steps[-1]['tensor'].tolist())
        st.download_button('Download CNN specification JSON',json.dumps(spec.as_dict(),indent=2),file_name='cnn-spec-v2.json')
    except (ValueError,RuntimeError) as exc:st.warning(f'Invalid architecture: {exc}')


def _backprop_tab():
    st.write('Follow logits → CrossEntropyLoss → backward → parameter gradients → optimizer.step → new loss.')
    optimizer=st.selectbox('One-step optimizer',['SGD','Adam'],key='cnn_back_optimizer')
    lr=st.select_slider('One-step learning rate',options=[.0001,.001,.01,.05,.1],value=.01,key='cnn_back_lr')
    spec=digits_spec()
    x=torch.arange(64,dtype=torch.float32).reshape(1,1,8,8)/63
    result=single_sgd_step(spec,x,torch.tensor([3]),lr,optimizer)
    st.plotly_chart(heatmap(x[0,0].numpy(),'Backpropagation input; target class 3'),width='stretch')
    st.code('optimizer.zero_grad()\nlogits = model(images)\nloss = cross_entropy(logits, targets)\nloss.backward()\noptimizer.step()',language='python')
    st.write('Logits',result['logits'].flatten().tolist())
    a,b=st.columns(2)
    a.metric('Loss before',f"{result['loss_before']:.6f}")
    b.metric('Loss after',f"{result['loss_after']:.6f}")
    name=st.selectbox('Inspect parameter',list(result['parameters']),key='cnn_backward_parameter')
    p=result['parameters'][name]
    st.metric('Gradient L2 norm',f"{p['gradient_norm']:.6f}")
    data=[{'before':float(old),'dLoss/dW':float(grad),'after update':float(new)} for old,grad,new in zip(
        p['before'].flatten()[:24],p['gradient'].flatten()[:24],p['after'].flatten()[:24])]
    st.dataframe(data,hide_index=True)
    st.latex(r'W_{new}=W_{old}-\eta\nabla_W L')
    st.caption('Only first 24 entries shown. SGD follows the equation above; Adam uses adaptive first and second moment estimates.')
    if st.button('Run deterministic numerical gradient check'):
        st.success(f'Autograd agrees with finite differences: {convolution_gradient_check()}')


def _training_tab():
    st.write('Train a real CNN using Digits (8×8); validation is used for monitoring, test is evaluated only when explicitly requested.')
    a,b,c,d=st.columns(4)
    with a:epochs=st.slider('Epochs',1,30,5,key='cnn_epochs')
    with b:batch=st.selectbox('Batch size',[16,32,64,128],index=1,key='cnn_batch')
    with c:opt=st.selectbox('Optimizer',['Adam','SGD'],key='cnn_opt')
    with d:lr=st.select_slider('Learning rate',options=[.0001,.0003,.001,.003,.01,.03,.1],value=.01,key='cnn_train_lr')
    a,b,c=st.columns(3)
    with a:filters1=st.slider('First filters',2,16,8,key='cnn_train_f1')
    with b:filters2=st.slider('Second filters',2,32,16,key='cnn_train_f2')
    with c:pool=st.selectbox('Pooling style',['maxpool2d','avgpool2d'],key='cnn_train_pool')
    seed=st.number_input('Training seed',min_value=0,max_value=2147483647,value=42,key='cnn_training_seed')
    cfg=CNNTrainingConfig(epochs,batch,lr,opt,int(seed),filters1,filters2,pool)
    if st.session_state.get('cnn_experiment_config')!=cfg:
        st.session_state.pop('cnn_experiment',None)
        st.session_state.pop('cnn_test_result',None)
        st.session_state.cnn_experiment_config=cfg
    if st.button('Train CNN',type='primary'):
        with st.spinner('Running CPU training...'):
            st.session_state.cnn_experiment=train_digits(cfg)
            st.session_state.pop('cnn_test_result',None)
    exp=st.session_state.get('cnn_experiment')
    if not exp:return
    hist=exp['history']
    st.dataframe(hist,hide_index=True)
    st.plotly_chart(px.line(hist,x='epoch',y=['train_loss','val_loss'],title='Train / validation loss'),width="stretch")
    st.plotly_chart(px.line(hist,x='epoch',y=['train_accuracy','val_accuracy'],title='Train / validation accuracy'),width="stretch")
    st.plotly_chart(px.line(hist,x='epoch',y='conv_gradient_norm',title='Convolution gradient norms'),width="stretch")
    p=st.selectbox('Compare convolution filters',list(exp['initial_filters']),key='cnn_compare_filter')
    old=exp['initial_filters'][p];new=exp['trained_filters'][p]
    st.write(f'Parameter: {p} — {tuple(old.shape)}. Showing output filter 0 and input channel 0.')
    a,b=st.columns(2)
    with a:st.plotly_chart(heatmap(old[0,0].numpy(),'Initial weights'),width="stretch")
    with b:st.plotly_chart(heatmap(new[0,0].numpy(),'Trained weights'),width="stretch")
    st.caption('The test set stays untouched throughout training. Evaluate once after deciding the configuration.')
    if st.button('Evaluate held-out test set'):
        st.session_state.cnn_test_result=evaluate_test(exp)
    result=st.session_state.get('cnn_test_result')
    if result is not None:
        st.metric('Held-out test accuracy',f"{result['accuracy']*100:.2f}%")
        st.plotly_chart(heatmap(result['confusion_matrix'],'Confusion matrix (rows actual, columns predicted)'),width="stretch")
        bad=result['incorrect_indices'][:12]
        if len(bad):
            st.write('Incorrect predictions')
            for i in bad:
                st.write(f'Actual {result["actual"][i]}, predicted {result["predicted"][i]}, confidence {result["confidence"][i]:.2f}')
                st.plotly_chart(heatmap(result['images'][i],f'Misclassified digit {i}'),width="stretch")
        st.write('Sample predictions (first 12 test images)')
        st.dataframe([{'actual':int(a),'prediction':int(b),'confidence':float(c)} for a,b,c in zip(result['actual'][:12],result['predicted'][:12],result['confidence'][:12])],hide_index=True)


def render_cnn_learning_studio():
    st.header('🖼️ CNN Learning Studio')
    if 'learning_progress_v1' not in st.session_state:
        st.session_state.learning_progress_v1=empty_progress()
    progress=st.session_state.learning_progress_v1
    st.info('Explore real PyTorch image tensors, cross-correlation, pooling, CNN shape propagation, Digits training and backpropagation.')
    section=st.selectbox('CNN section',['Concepts & mathematics','Convolution explorer','Pooling explorer','CNN architecture','Backpropagation','Training laboratory','Quiz & progress'],key='cnn_section')
    if section=='Concepts & mathematics':
        st.latex(r'Y[n,o,i,j]=b[o]+\sum_c\sum_u\sum_v W[o,c,u,v]X_{padded}[n,c,iS_h+uD_h,jS_w+vD_w]')
        st.latex(r'H_{out}=\lfloor(H_{in}+2P-D(K-1)-1)/S+1\rfloor')
        st.latex(r'\mathrm{params}=C_{out}(C_{in}K_hK_w+\mathrm{bias\_enabled})')
        st.caption('H_in/W_in: input height/width; P: padding; D: dilation; K: kernel size; S: stride. Parameter formula assumes standard ungrouped convolution.')
        st.latex(r'W_{out}=\lfloor(W_{in}+2P_w-D_w(K_w-1)-1)/S_w+1\rfloor')
        st.write('Example: RGB → 4 filters, 3×3, bias: 4×(3×3×3+1)=112 parameters. [1,3,8,8] → [1,4,6,6] at stride 1, padding 0.')
        for title,body in TOPICS:
            with st.expander(title,expanded=title=='Image tensors'):st.write(body)
    elif section=='Convolution explorer':_convolution_tab()
    elif section=='Pooling explorer':_pool_tab()
    elif section=='CNN architecture':_architecture_tab()
    elif section=='Backpropagation':_backprop_tab()
    elif section=='Training laboratory':_training_tab()
    else:_progress_panel(progress)
