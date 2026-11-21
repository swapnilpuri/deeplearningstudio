"""CNN Learning Studio tabs, powered only by the computation modules."""
import json
import numpy as np
import plotly.express as px
import streamlit as st
import torch
from .cnn_spec import CNNLayerSpec, CNNModelSpec, digits_spec
from .cnn_adapter import CNNAdapter, convolution_explorer, pooling_explorer, filter_presets, single_sgd_step
from .cnn_training import CNNTrainingConfig, train_digits, evaluate_test
from .progress import empty_progress, mark_complete, record_quiz, export_progress, load_progress
from visualization.cnn_graphs import heatmap, architecture_diagram

CONCEPTS = [
 ('Image tensors', 'Images are numerical tensors. PyTorch uses NCHW: batch, channels, height, width. Grayscale has one channel; RGB has three.'),
 ('Kernels & filters', 'A Conv2d filter has one 2D kernel per input channel. Filter weights and optional bias are learned via gradient descent.'),
 ('Sliding windows', 'PyTorch Conv2d performs cross-correlation: multiply aligned window and kernel values without flipping the kernel, then sum.'),
 ('Stride, padding, dilation', 'Stride determines how far the window moves, padding adds a border, and dilation spaces kernel positions.'),
 ('Feature maps', 'Each output filter generates one output channel. Several filters discover different patterns. ReLU keeps positive responses.'),
 ('Pooling', 'Max or average pooling summarizes local windows. It reduces downstream spatial computation and provides some local shift tolerance.'),
 ('Flatten and dense head', 'Flatten changes [N,C,H,W] to [N,C×H×W]. A Linear layer maps these features to ten class logits.'),
 ('Loss & gradients', 'CrossEntropyLoss combines log-softmax and negative log likelihood. Autograd differentiates through the convolution and pooling path.'),
 ('Optimization', 'SGD applies W_new = W_old − learning_rate × dLoss/dW. Adam uses momentum-like first and second moment estimates.'),
]
QUESTIONS = [
 ('A 3-channel input and 4 Conv2d 3×3 filters with bias have how many parameters?', ['108','112','36'],1,'4 × (3 × 3 × 3 + 1) = 112.'),
 ('Which dimensions are used for PyTorch images?', ['NHWC','NCHW','CHWN'],1,'Batch, channels, height, width.'),
 ('Does Conv2d flip its kernel?', ['Yes','No, it uses cross-correlation'],1,'The weights are used as written.'),
 ('8×8 input, 3×3 kernel, stride=1, padding=1 produces?', ['6×6','8×8','10×10'],1,'floor((8+2−3)/1+1)=8.'),
 ('2×2 max pooling with stride 2 on 8×8 produces?', ['4×4','8×8','2×2'],0,'A nonoverlapping 2×2 reduction halves each spatial axis.'),
 ('What does backward() do?', ['Updates weights','Computes derivatives','Resets gradients'],1,'The optimizer updates weights after autograd computes gradients.'),
]


def _progress_panel(progress):
    st.subheader('Quiz & guided experiments')
    answers=[]
    for i,(question,options,_,_) in enumerate(QUESTIONS):
        answers.append(st.radio(question,options,index=None,key=f'cnn_question_{i}'))
    if st.button('Grade CNN quiz'):
        if any(a is None for a in answers): st.warning('Answer all questions first.')
        else:
            score=sum(options.index(answer)==correct for answer,(_,options,correct,_) in zip(answers,QUESTIONS))
            st.session_state.learning_progress_v1=record_quiz(progress,'cnn',score,len(QUESTIONS))
            st.success(f'Score: {score}/{len(QUESTIONS)}')
            for question,options,correct,explanation in QUESTIONS:st.caption(f'{question} — {explanation}')
    for item in ['Change kernel from vertical to horizontal edge and compare feature maps.',
                 'Increase stride while fixing input and padding: predict output sizes.',
                 'Increase filter count: predict parameter counts before rebuilding.',
                 'Compare max vs average pooling and explain which details disappear.',
                 'Compare filters before and after 5 training epochs.']:
        st.write('• '+item)
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
    with c1:size=st.slider('Image height/width',4,12,6,key='cnn_img_size')
    with c2:channels=st.selectbox('Input channels',[1,3],key='cnn_in_channels')
    with c3:k=st.slider('Kernel size',1,5,3,key='cnn_kernel')
    with c4:nfilters=st.slider('Output filters',1,8,2,key='cnn_filters')
    c1,c2,c3,c4=st.columns(4)
    with c1:stride=st.slider('Stride',1,3,1,key='cnn_stride')
    with c2:padding=st.slider('Padding',0,3,1,key='cnn_padding')
    with c3:use_bias=st.checkbox('Use bias',True,key='cnn_use_bias')
    with c4:activation=st.selectbox('Activation',['None','ReLU'],key='cnn_activation')
    preset=st.selectbox('Kernel', ['Vertical edge','Horizontal edge','Identity','Blur','Sharpen','Custom seeded'],key='cnn_filter_preset')
    seed=st.number_input('Explorer seed',min_value=0,max_value=100000,value=42,key='cnn_explorer_seed')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        x=torch.arange(1,size*size*channels+1,dtype=torch.float32).reshape(channels,size,size)/size
        weights=torch.randn(nfilters,channels,k,k)*.25
        if k==3 and preset!='Custom seeded':
            weights[:]=filter_presets(preset,channels)
        bias=torch.zeros(nfilters) if use_bias else None
    if preset=='Custom seeded':
        raw=st.text_area(f'Optional weights for filter 0, channel 0 ({k*k} comma-separated)',value='',key=f'cnn_custom_{k}')
        if raw.strip():
            try:
                numbers=[float(v.strip()) for v in raw.split(',')]
                if len(numbers)!=k*k or not np.isfinite(numbers).all(): raise ValueError('Expected finite weights')
                weights[0,0]=torch.tensor(numbers).reshape(k,k)
            except ValueError as exc:st.warning(f'Invalid custom filter: {exc}')
    try:
        out_shape=(size+2*padding-k)//stride+1
        if out_shape<1:raise ValueError('Kernel larger than padded input')
        ci,cf=st.columns(2)
        with ci:fil=st.selectbox('Output filter index',list(range(nfilters)),key='cnn_filter_index')
        with cf:channel=st.selectbox('Input channel to inspect',list(range(channels)),key='cnn_channel_index')
        row=st.slider('Output row',0,out_shape-1,0,key='cnn_conv_row')
        col=st.slider('Output column',0,out_shape-1,0,key='cnn_conv_col')
        result=convolution_explorer(x,weights,bias,stride,padding,activation,row,col,fil)
        st.latex(r'H_{out}=\lfloor(H+2P-D(K-1)-1)/S+1\rfloor')
        st.write(f'Input {tuple(x[None].shape)} · Weights {tuple(weights.shape)} · Output {tuple(result["output"][None].shape)}')
        st.caption(f'Window begins at unpadded coordinates {result["window_origin"]}; green highlight corresponds to the patch (including padding when outside image).')
        left,right=st.columns(2)
        with left:
            st.plotly_chart(heatmap(x[channel].numpy(),'Input channel (un-padded)'),use_container_width=True)
            st.plotly_chart(heatmap(result['patch'][channel].numpy(),'Selected padded window'),use_container_width=True)
        with right:
            st.plotly_chart(heatmap(weights[fil,channel].numpy(),'Filter weights'),use_container_width=True)
            st.plotly_chart(heatmap(result['products'][channel].numpy(),'Window × kernel'),use_container_width=True)
        st.write('Per-channel contributions',result['channel_contributions'].tolist())
        st.write(f'Sum of products {result["sum_products"]:.5g} + bias {result["bias"]:.5g} = {result["linear_value"]:.5g}; after {activation}: {result["activated_value"]:.5g}')
        views=st.columns(min(4,nfilters))
        for f in range(nfilters):
            with views[f%len(views)]:
                st.plotly_chart(heatmap(result['output'][f].numpy(),f'Filter {f} pre-activation',highlight=(row,col,1,1)),use_container_width=True)
                if activation=='ReLU':st.plotly_chart(heatmap(result['activated'][f].numpy(),f'Filter {f} post-ReLU'),use_container_width=True)
    except (ValueError,RuntimeError) as exc:st.warning(str(exc))


def _pool_tab():
    st.write('Pooling operates independently on each feature-map channel. This example uses one 2D feature map.')
    kind=st.selectbox('Pooling mode',['max','average'],key='cnn_pool_kind')
    size=st.slider('Feature-map width/height',4,12,6,key='cnn_pool_size')
    k=st.slider('Pool kernel',1,4,2,key='cnn_pool_k')
    stride=st.slider('Pool stride',1,4,2,key='cnn_pool_stride')
    fmap=torch.arange(1,size*size+1,dtype=torch.float32).reshape(size,size)
    try:
        dims=(size-k)//stride+1
        if dims<1:raise ValueError('Pooling kernel too large')
        row=st.slider('Pool output row',0,dims-1,0,key='cnn_pool_row')
        col=st.slider('Pool output column',0,dims-1,0,key='cnn_pool_col')
        result=pooling_explorer(fmap,kind,k,stride,row,col)
        a,b=st.columns(2)
        with a:st.plotly_chart(heatmap(fmap.numpy(),'Feature map',highlight=(*result['origin'],k,k)),use_container_width=True)
        with b:st.plotly_chart(heatmap(result['output'].numpy(),f'{kind.title()} pooling output',highlight=(row,col,1,1)),use_container_width=True)
        st.write(f'Window {result["window"].tolist()} → {kind} = {result["value"]:.4g}')
        st.latex(r'H_{out}=\lfloor(H-K)/S+1\rfloor')
        st.caption('For this formula: zero padding, dilation=1 and ceil_mode=False. Pooling has no trainable weights.')
    except (ValueError,RuntimeError) as exc:st.warning(str(exc))


def _architecture_tab():
    st.write('Construct a two-block CNN. The dense input size is derived from tensor shapes, never hardcoded.')
    size=st.selectbox('Square image size',[8,12,16,28],key='cnn_arch_size')
    channels=st.selectbox('Input channels',[1,3],key='cnn_arch_channels')
    f1=st.slider('Block 1 output filters',1,32,8,key='cnn_arch_f1')
    f2=st.slider('Block 2 output filters',1,32,16,key='cnn_arch_f2')
    k=st.slider('Convolution kernel',1,5,3,key='cnn_arch_kernel')
    pad=st.slider('Conv padding',0,2,1,key='cnn_arch_pad')
    pool=st.selectbox('Pooling',['maxpool2d','avgpool2d'],key='cnn_arch_pool')
    classes=st.slider('Class logits',2,20,10,key='cnn_arch_classes')
    try:
        spec=CNNModelSpec(channels,size,size,(
            CNNLayerSpec('conv2d',out_channels=f1,kernel_size=(k,k),padding=(pad,pad)),CNNLayerSpec('relu'),
            CNNLayerSpec(pool,kernel_size=(2,2),stride=(2,2)),
            CNNLayerSpec('conv2d',out_channels=f2,kernel_size=(k,k),padding=(pad,pad)),CNNLayerSpec('relu'),
            CNNLayerSpec(pool,kernel_size=(2,2),stride=(2,2)),
            CNNLayerSpec('flatten'),CNNLayerSpec('linear',out_features=classes)))
        rows=spec.infer_shapes()
        st.plotly_chart(architecture_diagram(rows),use_container_width=True)
        st.dataframe(rows,hide_index=True,use_container_width=True)
        st.metric('Trainable parameters',sum(row['parameters'] for row in rows))
        _,steps=CNNAdapter().trace(spec)
        st.write('Actual PyTorch layer outputs')
        st.dataframe([{'operator':s['operator'],'shape':s['shape']} for s in steps],hide_index=True)
        st.download_button('Download CNN specification JSON',json.dumps(spec.as_dict(),indent=2),file_name='cnn-spec-v2.json')
    except (ValueError,RuntimeError) as exc:st.warning(f'Invalid architecture: {exc}')


def _backprop_tab():
    st.write('Follow logits → CrossEntropyLoss → backward → parameter gradients → SGD.step → new loss.')
    lr=st.select_slider('One-step SGD learning rate',options=[.0001,.001,.01,.05,.1],value=.01,key='cnn_back_lr')
    spec=digits_spec()
    x=torch.arange(64,dtype=torch.float32).reshape(1,1,8,8)/63
    result=single_sgd_step(spec,x,torch.tensor([3]),lr)
    st.write('Logits',result['logits'].flatten().tolist())
    a,b=st.columns(2)
    a.metric('Loss before',f"{result['loss_before']:.6f}")
    b.metric('Loss after',f"{result['loss_after']:.6f}")
    name=st.selectbox('Inspect parameter',list(result['parameters']),key='cnn_backward_parameter')
    p=result['parameters'][name]
    st.metric('Gradient L2 norm',f"{p['gradient_norm']:.6f}")
    data=[{'before':float(old),'dLoss/dW':float(grad),'after SGD':float(new)} for old,grad,new in zip(
        p['before'].flatten()[:24],p['gradient'].flatten()[:24],p['after'].flatten()[:24])]
    st.dataframe(data,hide_index=True)
    st.latex(r'W_{new}=W_{old}-\eta\nabla_W L')
    st.caption('Only first 24 entries shown; all updates are actual PyTorch optimizer steps. Adam would use adaptive moments instead.')


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
    cfg=CNNTrainingConfig(epochs,batch,lr,opt,42,filters1,filters2,pool)
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
    st.plotly_chart(px.line(hist,x='epoch',y=['train_loss','val_loss'],title='Train / validation loss'),use_container_width=True)
    st.plotly_chart(px.line(hist,x='epoch',y=['train_accuracy','val_accuracy'],title='Train / validation accuracy'),use_container_width=True)
    st.plotly_chart(px.line(hist,x='epoch',y='conv_gradient_norm',title='Convolution gradient norms'),use_container_width=True)
    p=st.selectbox('Compare convolution filters',list(exp['initial_filters']),key='cnn_compare_filter')
    old=exp['initial_filters'][p];new=exp['trained_filters'][p]
    st.write(f'Parameter: {p} — {tuple(old.shape)}. Showing output filter 0 and input channel 0.')
    a,b=st.columns(2)
    with a:st.plotly_chart(heatmap(old[0,0].numpy(),'Initial weights'),use_container_width=True)
    with b:st.plotly_chart(heatmap(new[0,0].numpy(),'Trained weights'),use_container_width=True)
    st.caption('The test set stays untouched throughout training. Evaluate once after deciding the configuration.')
    if st.button('Evaluate held-out test set'):
        st.session_state.cnn_test_result=evaluate_test(exp)
    result=st.session_state.get('cnn_test_result')
    if result is not None:
        st.metric('Held-out test accuracy',f"{result['accuracy']*100:.2f}%")
        st.plotly_chart(heatmap(result['confusion_matrix'],'Confusion matrix (rows actual, columns predicted)'),use_container_width=True)
        bad=result['incorrect_indices'][:12]
        if len(bad):
            st.write('Incorrect predictions')
            for i in bad:
                st.write(f'Actual {result["actual"][i]}, predicted {result["predicted"][i]}, confidence {result["confidence"][i]:.2f}')
                st.plotly_chart(heatmap(result['images'][i],f'Misclassified digit {i}'),use_container_width=True)
        st.write('Sample predictions (first 12 test images)')
        st.dataframe([{'actual':int(a),'prediction':int(b),'confidence':float(c)} for a,b,c in zip(result['actual'][:12],result['predicted'][:12],result['confidence'][:12])],hide_index=True)


def render_cnn_learning_studio():
    st.header('🖼️ CNN Learning Studio')
    if 'learning_progress_v1' not in st.session_state:
        st.session_state.learning_progress_v1=empty_progress()
    progress=st.session_state.learning_progress_v1
    st.info('Explore real PyTorch image tensors, cross-correlation, pooling, CNN shape propagation, Digits training and backpropagation.')
    tabs=st.tabs(['Concepts & mathematics','Convolution explorer','Pooling explorer','CNN architecture','Backpropagation','Training laboratory','Quiz & progress'])
    with tabs[0]:
        st.latex(r'Y[n,o,i,j]=b[o]+\sum_c\sum_u\sum_v W[o,c,u,v]X[n,c,iS+u-P,jS+v-P]')
        st.latex(r'H_{out}=\lfloor(H_{in}+2P-D(K-1)-1)/S+1\rfloor')
        st.latex(r'\mathrm{params}=C_{out}(C_{in}K_hK_w+\mathrm{bias\_enabled})')
        for title,body in CONCEPTS:
            with st.expander(title,expanded=title=='Image tensors'):st.write(body)
    with tabs[1]:_convolution_tab()
    with tabs[2]:_pool_tab()
    with tabs[3]:_architecture_tab()
    with tabs[4]:_backprop_tab()
    with tabs[5]:_training_tab()
    with tabs[6]:_progress_panel(progress)
