"""PyTorch Neural Network Studio + Training Studio (Streamlit application)."""
import io
import json
import math
import tempfile
from pathlib import Path

import streamlit as st
import torch

from models.feed_forward import ACTIVATIONS, build_model, generate_code
from training.exports import history_csv
from training.engine import DATASETS, OPTIMIZERS, Experiment, ExperimentConfig, is_classification
from training.datasets import built_in_dataset, prepare_dataset, read_uploaded_dataset
from visualization.network_graph import network_figure
from visualization.training_graphs import dataset_figure, decision_figure, history_figure

st.set_page_config(page_title='PyTorch Neural Network Studio', page_icon='🧠', layout='wide')
st.title('🧠 PyTorch Neural Network Studio')
st.caption('Learn by building: network architectures, forward propagation, training, losses and gradients')

with st.sidebar:
    studio = st.radio('Choose workspace', ['Neural Network Studio', 'Dataset Management', 'Training Studio', 'Learning Studio'], key='workspace')
    st.divider()


def architecture_controls(prefix: str, input_size=4, output_size=2, max_width=32):
    """Sidebar controls with unique keys across both studios."""
    inputs = st.slider('Input neurons', 1, 16, input_size, key=f'{prefix}_inputs', disabled=(prefix == 'train'))
    count = st.slider('Hidden layers', 0, 5, 2, key=f'{prefix}_count')
    hidden = [st.slider(f'Hidden layer {i+1} neurons', 1, max_width, 16 if prefix == 'train' else (6 if i == 0 else 4), key=f'{prefix}_hidden_{i}') for i in range(count)]
    outputs = st.slider('Output neurons', 1, 16, output_size, key=f'{prefix}_outputs', disabled=(prefix == 'train'))
    activation = st.selectbox('Hidden activation', list(ACTIVATIONS), index=2 if prefix == 'train' else 0, key=f'{prefix}_activation')
    bias = st.checkbox('Include bias', value=True, key=f'{prefix}_bias')
    seed = int(st.number_input('Random seed', 0, 2147483647, 42, key=f'{prefix}_seed'))
    return [inputs, *hidden, outputs], activation, bias, seed


def parameter_table(model):
    return [{'Parameter': name, 'Shape': str(tuple(p.shape)), 'Count': p.numel(), 'Gradient required': p.requires_grad} for name, p in model.named_parameters()]


def export_controls(model, signature, sizes, activation, bias, seed, prefix='design'):
    code = generate_code(sizes, activation, bias, seed)
    st.code(code, language='python')
    st.download_button('Download model Python code', data=code, file_name='generated_model.py', mime='text/x-python', key=f'{prefix}_code_download')
    st.caption('ONNX export captures the current model weights (trained weights when using Training Studio).')
    if st.button('Generate ONNX', key=f'{prefix}_onnx_generate'):
        try:
            with st.spinner('Exporting model…'):
                with tempfile.TemporaryDirectory() as d:
                    path = Path(d) / 'network.onnx'
                    previous = model.training
                    model.eval()
                    try:
                        torch.onnx.export(model, (torch.zeros(1, sizes[0]),), str(path), input_names=['input'], output_names=['output'], dynamo=True, external_data=False)
                    finally:
                        model.train(previous)
                    import onnx
                    onnx.checker.check_model(str(path))
                    st.session_state[f'{prefix}_onnx_bytes'] = path.read_bytes()
                    st.session_state[f'{prefix}_onnx_signature'] = signature
            st.success('ONNX model exported and validated.')
        except Exception as exc:
            st.error(f'ONNX export failed: {exc}')
    if st.session_state.get(f'{prefix}_onnx_signature') == signature:
        st.download_button('Download ONNX model', st.session_state[f'{prefix}_onnx_bytes'], file_name='network.onnx', mime='application/octet-stream', key=f'{prefix}_onnx_download')
    st.markdown('Open the downloaded ONNX file using [Netron](https://netron.app).')
    st.code('uv run python -c "import netron; netron.start(\'network.onnx\', browse=True)"', language='powershell')


if studio == 'Learning Studio':
    from learning.ui import render_learning_studio
    render_learning_studio()

elif studio == 'Neural Network Studio':
    with st.sidebar:
        st.header('Architecture configuration')
        sizes, activation, bias, seed = architecture_controls('design')
    signature = (tuple(sizes), activation, bias, seed)
    model = build_model(sizes, activation, bias, seed)
    linear = [m for m in model if isinstance(m, torch.nn.Linear)]
    a, b, c = st.columns(3)
    a.metric('Neurons', sum(sizes))
    b.metric('Weight connections', sum(m.weight.numel() for m in linear))
    c.metric('Trainable parameters', sum(p.numel() for p in model.parameters()))
    graph, inspect, forward, export = st.tabs(['Architecture', 'Weights & biases', 'Forward pass', 'Code & ONNX'])
    with graph:
        st.plotly_chart(network_figure(sizes), width="stretch")
        st.caption('Each line is one learnable connection; the displayed graph is an architecture diagram, not a computational-operator graph.')
    with inspect:
        st.dataframe(parameter_table(model), hide_index=True, width="stretch")
        selected = st.selectbox('Inspect linear layer', list(range(1, len(linear)+1)))
        layer = linear[selected-1]
        st.write('Weights [output neuron, input neuron]')
        st.dataframe(layer.weight.detach().numpy(), width="stretch")
        if layer.bias is not None:
            st.write('Bias vector')
            st.dataframe(layer.bias.detach().numpy().reshape(1, -1), width="stretch")
    with forward:
        st.write('Enter one number per input feature to trace each PyTorch layer.')
        raw = st.text_input('Input values (comma-separated)', ', '.join(['1']*sizes[0]), key=f'design_input_{sizes[0]}')
        try:
            values = [float(part.strip()) for part in raw.split(',')]
            if len(values) != sizes[0] or not all(math.isfinite(value) for value in values):
                raise ValueError(f'Enter exactly {sizes[0]} finite numbers.')
            x = torch.tensor([values], dtype=torch.float32)
            with torch.no_grad():
                for i, layer in enumerate(model):
                    x = layer(x)
                    st.write(f'{i+1}. {layer.__class__.__name__} — tensor shape {tuple(x.shape)}')
                    st.dataframe(x.numpy(), width="stretch")
        except (ValueError, OverflowError) as exc:
            st.warning(str(exc))
    with export:
        export_controls(model, signature, sizes, activation, bias, seed)

elif studio == 'Dataset Management':
    st.header('📁 Dataset Management')
    st.caption('Choose a generated/built-in dataset or upload CSV/XLSX. Apply it to make it the active dataset in Training Studio.')
    with st.sidebar:
        st.header('Dataset source')
        source = st.radio('Source', ['Built-in / synthetic', 'Upload CSV / Excel'])
        if source == 'Built-in / synthetic':
            name = st.selectbox('Dataset', [*DATASETS, 'Iris', 'Digits (8×8)', 'MNIST (28×28)'])
            dseed = int(st.number_input('Dataset random seed', 0, 2147483647, 42))
            dsamples = st.slider('Samples (synthetic / MNIST)', 100, 3000, 320, 20)
            dnoise = st.slider('Noise (synthetic only)', 0.0, 0.5, 0.12, .01)
            uploaded = None
        else:
            uploaded = st.file_uploader('Choose CSV or XLSX', type=['csv', 'xlsx'])
            name = uploaded.name if uploaded else 'Uploaded data'
            dseed, dsamples, dnoise = 42, 320, .12
    try:
        if source == 'Built-in / synthetic':
            frame, target_default, task_default = built_in_dataset(name, seed=dseed, samples=dsamples, noise=dnoise)
        elif uploaded:
            frame = read_uploaded_dataset(uploaded.getvalue(), uploaded.name)
            target_default = str(frame.columns[-1]); task_default = 'classification'
        else:
            frame = None
        if frame is not None:
            st.subheader('Raw dataset preview')
            st.dataframe(frame.head(40), hide_index=True, width="stretch")
            c1,c2,c3 = st.columns(3)
            c1.metric('Rows', f'{len(frame):,}'); c2.metric('Columns', len(frame.columns)); c3.metric('Missing cells', int(frame.isna().sum().sum()))
            col1, col2 = st.columns(2)
            with col1:
                target = st.selectbox('Target / label column', list(frame.columns), index=list(frame.columns).index(target_default))
                task = st.radio('Task type', ['classification', 'regression'], index=0 if task_default=='classification' else 1)
                candidates = [c for c in frame.columns if c != target]
                chosen = st.multiselect('Numeric feature columns (select up to 784)', candidates, default=candidates[:784])
            with col2:
                norm = st.checkbox('Standardize numerical features (train-only fit)', value=True)
                impute = st.selectbox('Missing feature imputation', ['median', 'mean'])
                train_pct = st.slider('Training percentage', 50, 90, 70, 5)
                val_pct = st.slider('Validation percentage', 5, min(30,95-train_pct), 15 if train_pct <= 80 else 5, 5)
                st.write(f'Test percentage: **{100-train_pct-val_pct}%**')
                split_seed = int(st.number_input('Split seed', min_value=0, max_value=2147483647, value=42))
            st.subheader('Explore features')
            numeric_columns = frame.select_dtypes(include='number').columns.tolist()
            available_features = [c for c in numeric_columns if c != target]
            if available_features:
                chosen_feature = st.selectbox('Feature distribution', available_features)
                import plotly.express as px
                st.plotly_chart(px.histogram(frame, x=chosen_feature, nbins=30, title=f'{chosen_feature} distribution'), width="stretch")
            if task == 'classification':
                st.caption('Class counts (raw dataset)')
                st.bar_chart(frame[target].astype(str).value_counts())
            else:
                st.caption('Target distribution (raw dataset)')
                if target in numeric_columns:
                    st.plotly_chart(px.histogram(frame, x=target, nbins=30), width="stretch")
            if st.button('Apply dataset to Training Studio', type='primary'):
                try:
                    bundle = prepare_dataset(frame, name=name, target=target, task=task,
                        features=chosen, seed=split_seed, train_fraction=train_pct/100,
                        val_fraction=val_pct/100, normalize=norm, impute=impute)
                    st.session_state['active_bundle'] = bundle
                    st.session_state.pop('experiment_signature', None)
                    st.session_state.pop('experiment', None)
                    st.success(f'Active dataset: {name}. Open Training Studio to train on {bundle.summary["train"]} training samples and monitor {bundle.summary["validation"]} validation samples. {bundle.summary["test"]} test samples held out.')
                except (ValueError, TypeError) as exc:
                    st.error(str(exc))
        else:
            st.info('Upload a CSV or XLSX file to preview columns and configure preprocessing.')
    except Exception as exc:
        st.error(f'Unable to prepare dataset: {exc}')
    if 'active_bundle' in st.session_state:
        active = st.session_state['active_bundle']
        st.divider()
        st.subheader(f'Active dataset: {active.name}')
        st.json({'task': active.task, 'features': active.input_size, 'outputs': active.output_size, **active.summary})
        st.caption('Scaling and imputation are fitted exclusively on training samples. Validation/test samples never contribute to preprocessing fit or optimizer updates.')

else:
    with st.sidebar:
        st.header('Training experiment')
        bundle = st.session_state.get('active_bundle')
        st.caption('Dataset Management controls the active dataset. If none is selected, Two Moons is used with held-out splits.')
        if bundle is None:
            fallback, tgt, task = built_in_dataset('Two moons')
            bundle = prepare_dataset(fallback, name='Two moons', target=tgt, task=task)
            st.session_state['active_bundle'] = bundle
        st.write(f'**Dataset:** {bundle.name}')
        st.caption(f'{bundle.summary["train"]} train / {bundle.summary["validation"]} validation / {bundle.summary["test"]} test')
        classification = bundle.classification
        dataset = bundle.name
        samples = bundle.summary['train']
        noise = 0.0
        st.divider()
        st.header('Network configuration')
        # Classification datasets have 2 features and 2 output logits; regression datasets have 1 input/output.
        sizes, activation, bias, seed = architecture_controls('train', min(16,bundle.input_size), min(16,bundle.output_size))
        # A disabled slider's state can persist across dataset changes; derive the true sizes explicitly.
        sizes[0] = bundle.input_size
        sizes[-1] = bundle.output_size
        st.divider()
        st.header('Optimization')
        optimizer_name = st.selectbox('Optimizer', OPTIMIZERS, index=1)
        lr = st.select_slider('Learning rate', options=[0.0001,0.0003,0.001,0.003,0.01,0.02,0.03,0.05,0.1,0.2,0.5], value=0.02)
        batch_size = st.select_slider('Batch size', options=[8,16,32,64,128,256,512,1024], value=32)
        momentum = st.slider('SGD momentum', 0.0, 0.99, 0.9, 0.01, disabled=optimizer_name != 'SGD')
        decay = st.select_slider('Weight decay', options=[0.0, 0.0001, 0.001, 0.01, 0.1], value=0.0)
        epochs_per_run = st.slider('Epochs per click', 1, 200, 25)

    cfg = ExperimentConfig(dataset=dataset, hidden=tuple(sizes[1:-1]), activation=activation, bias=bias, seed=seed, samples=samples, noise=noise, optimizer=optimizer_name, lr=lr, batch_size=batch_size, momentum=momentum if optimizer_name == 'SGD' else 0.0, weight_decay=decay)
    signature = (cfg, id(bundle))
    if st.session_state.get('experiment_signature') != signature:
        st.session_state['experiment'] = Experiment(cfg, bundle)
        st.session_state['experiment_signature'] = signature
        st.session_state.pop('test_result', None)
        st.session_state.pop('train_onnx_bytes', None)
        st.session_state.pop('train_onnx_signature', None)
        st.session_state.pop('test_result', None)
    exp = st.session_state['experiment']
    a, b, c, d = st.columns(4)
    a.metric('Epochs', exp.epoch)
    b.metric('Loss', f"{exp.history[-1]['loss']:.5f}")
    c.metric('Accuracy' if exp.classification else 'RMSE', f"{exp.history[-1]['score']*100:.1f}%" if exp.classification else f"{exp.history[-1]['score']:.5f}")
    d.metric('Trainable parameters', sum(p.numel() for p in exp.model.parameters()))
    st.caption(f"Objective: {'CrossEntropyLoss (class logits)' if exp.classification else 'MSELoss'} · {cfg.optimizer} · learning rate {cfg.lr:g} · CPU")
    col_train, col_reset = st.columns([3, 1])
    with col_train:
        if st.button(f'▶ Train {epochs_per_run} additional epochs', type='primary', width="stretch"):
            try:
                with st.spinner('Training on CPU…'):
                    exp.train_epochs(epochs_per_run)
                st.session_state.pop('train_onnx_bytes', None)
                st.session_state.pop('train_onnx_signature', None)
                st.session_state.pop('test_result', None)
                st.rerun()
            except (FloatingPointError, ValueError) as exc:
                st.error(str(exc))
    with col_reset:
        if st.button('↺ Reset', width="stretch"):
            st.session_state['experiment'] = Experiment(cfg, bundle)
            st.session_state.pop('train_onnx_bytes', None)
            st.session_state.pop('train_onnx_signature', None)
            st.session_state.pop('test_result', None)
            st.rerun()
    st.info('Metrics above are TRAIN metrics. Validation metrics are measured separately after every epoch; TEST is held out for final evaluation.', icon='ℹ️')
    if exp.val_x is not None:
        v1, v2 = st.columns(2)
        v1.metric('Validation loss', f"{exp.history[-1]['val_loss']:.5f}")
        v2.metric('Validation accuracy' if exp.classification else 'Validation RMSE', f"{exp.history[-1]['val_score']*100:.1f}%" if exp.classification else f"{exp.history[-1]['val_score']:.5f}")
    overview, curves, gradients, params, export = st.tabs(['Data & predictions', 'Learning curves', 'Gradients', 'Model parameters', 'Export trained model'])
    with overview:
        if exp.x.shape[1] == 2 and exp.classification and exp.sizes[-1] == 2:
            st.plotly_chart(decision_figure(exp), width="stretch")
        elif exp.x.shape[1] == 1 and not exp.classification:
            st.plotly_chart(dataset_figure(exp), width="stretch")
        else:
            st.caption('Decision heatmaps are available for two-feature binary classification. For higher dimensions, inspect features in Dataset Management.')
            st.dataframe(bundle.raw_frame.head(30), hide_index=True, width="stretch")
        st.caption('Classification: background color represents the model\'s current class-1 probability. Regression: line represents predictions.')
        st.plotly_chart(network_figure(exp.sizes), width="stretch")
    with curves:
        left, right = st.columns(2)
        with left:
            st.subheader('Loss vs epoch')
            st.plotly_chart(history_figure(exp, 'loss'), width="stretch")
        with right:
            st.subheader('Accuracy vs epoch' if exp.classification else 'Root mean squared error')
            st.plotly_chart(history_figure(exp, 'score'), width="stretch")
        if exp.val_x is not None:
            st.subheader('Held-out validation loss')
            st.plotly_chart(history_figure(exp, 'val_loss'), width="stretch")
            st.subheader('Held-out validation performance')
            st.plotly_chart(history_figure(exp, 'val_score'), width="stretch")
        st.subheader('Average mini-batch gradient norm')
        st.plotly_chart(history_figure(exp, 'grad_norm'), width="stretch")
        st.caption('Gradient norms are measured after backward() and before optimizer.step(). Epoch 0 has no gradients.')
        st.dataframe(exp.history[-15:][::-1], hide_index=True, width="stretch")
    with gradients:
        st.markdown('**Backpropagation diagnostics**')
        st.write('Each gradient measures how changing a parameter affects the loss. Values below come from the last mini-batch processed.')
        if exp.last_gradients:
            st.dataframe(exp.last_gradients, hide_index=True, width="stretch")
            st.bar_chart({item['parameter']: item['grad_norm'] for item in exp.last_gradients}, horizontal=True)
        else:
            st.warning('Train at least one epoch to see gradient values.')
        st.latex(r'\theta_{t+1} = \theta_t - \eta\,\nabla_{\theta}L(\theta_t)')
        st.caption('This equation illustrates basic gradient descent; Adam and RMSprop use modified update rules.')
    with params:
        st.dataframe(parameter_table(exp.model), hide_index=True, width="stretch")
        names = dict(exp.model.named_parameters())
        selected_name = st.selectbox('Inspect parameter tensor', list(names))
        values = names[selected_name].detach().cpu().numpy()
        st.dataframe(values.reshape(1, -1) if values.ndim == 1 else values, width="stretch")
        initial = exp.initial_parameters[selected_name]
        change = (names[selected_name].detach() - initial).norm().item()
        st.metric('L2 change from initialization', f'{change:.6f}')
    with export:
        st.subheader('Held-out test evaluation')
        st.caption('Inspect this after selecting the model using validation data. Repeated tuning using test results can bias final evaluation.')
        if st.button('Evaluate test split'):
            st.session_state['test_result'] = exp.test_metrics()
        if 'test_result' in st.session_state:
            st.json(st.session_state['test_result'])
        st.write('Export the actual currently trained weights. ONNX export will use these parameters, not a fresh untrained instance.')
        buffer = io.BytesIO()
        torch.save({'model_state_dict': exp.model.state_dict(), 'sizes': exp.sizes, 'activation': cfg.activation, 'bias': cfg.bias, 'config': cfg.__dict__, 'epoch': exp.epoch}, buffer)
        st.download_button('Download PyTorch checkpoint (.pt)', buffer.getvalue(), file_name='trained_network.pt', mime='application/octet-stream')
        st.download_button('Download training metrics (.csv)', history_csv(exp.history), 'training_metrics.csv', 'text/csv')
        st.download_button('Download experiment configuration (.json)', json.dumps(cfg.__dict__, indent=2), 'experiment_config.json', 'application/json')
        st.divider()
        export_controls(exp.model, (signature, exp.epoch), exp.sizes, activation, bias, seed, prefix='train')
        st.warning('The generated Python code builds the architecture with newly initialized weights. For learned parameters, use the exported .pt checkpoint or ONNX file.')

st.divider()
st.caption('Neural Network Studio · Educational experiments · Preprocessing fitted on train only · Validate before testing.')
