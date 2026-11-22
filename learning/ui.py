"""Streamlit presentation layer only; computation and content live in separate modules."""
import json
import streamlit as st
from .spec import ModelSpec
from .pytorch_adapter import get_adapter
from .content import LESSONS, EXERCISES
from .progress import empty_progress, load_progress, export_progress, mark_complete, record_quiz
from visualization.learning_graphs import tensor_heatmap, gradient_bar
from visualization.network_graph import network_figure


def render_learning_studio():
    st.header('📚 Learning Studio — Foundations')
    st.caption('Real PyTorch tensors, layer-by-layer explanations, autograd and one SGD update. TensorFlow is a future adapter, not yet implemented.')
    if 'learning_progress_v1' not in st.session_state:
        st.session_state.learning_progress_v1 = empty_progress()
    progress = st.session_state.learning_progress_v1
    lesson = st.selectbox('Learning path', [*LESSONS, 'cnn', 'rnn', 'lstm', 'embedding', 'attention'], format_func=lambda k: LESSONS[k]['title'] if k in LESSONS else {'cnn':'Convolutional Neural Network (CNN)', 'rnn':'Recurrent Neural Network (RNN)', 'lstm':'Long Short-Term Memory (LSTM)', 'embedding':'Embedding Network', 'attention':'Single-Head Self-Attention Learning Studio'}[k], key='learn_lesson')
    if lesson == 'attention':
        from .attention_ui import render_attention_learning_studio
        render_attention_learning_studio()
        return
    if lesson == 'embedding':
        from .embedding_ui import render_embedding_learning_studio
        render_embedding_learning_studio()
        return
    if lesson == 'lstm':
        from .lstm_ui import render_lstm_learning_studio
        render_lstm_learning_studio()
        return
    if lesson == 'rnn':
        from .rnn_ui import render_rnn_learning_studio
        render_rnn_learning_studio()
        return
    if lesson == 'cnn':
        from .cnn_ui import render_cnn_learning_studio
        render_cnn_learning_studio()
        return
    content = LESSONS[lesson]
    st.info(content['goal'])
    completed = len(set(progress.get('completed', [])) & set(LESSONS))
    st.progress(completed / len(LESSONS), text=f'{completed} of {len(LESSONS)} foundation lessons completed')
    left, right = st.columns([1, 2])
    with left:
        st.subheader('Model configuration')
        inputs = st.slider('Input features', 1, 8, 2, key='learn_inputs')
        hidden_count = 1 if lesson == 'feed_forward' else st.slider('Hidden layers', 2, 5, 2, key='learn_depth')
        width = st.slider('Neurons per hidden layer', 1, 16, 3, key='learn_width')
        outputs = st.slider('Outputs', 1, 4, 1, key='learn_outputs')
        activation = st.selectbox('Activation', ['Tanh', 'ReLU', 'Sigmoid', 'GELU'], key='learn_activation')
        bias = st.checkbox('Bias', True, key='learn_bias')
        seed = int(st.number_input('Seed', 0, 2147483647, 42, key='learn_seed'))
        lr = st.slider('SGD learning rate', 0.01, 1.0, 0.1, 0.01, key='learn_lr')
    spec = ModelSpec.for_lesson(lesson, inputs, (width,) * hidden_count, outputs, activation, bias, seed)
    adapter = get_adapter(spec.framework)
    st.metric('Trainable parameters', adapter.parameter_count(spec))
    tabs = st.tabs(['Concepts & math', 'Architecture & tensors', 'Forward propagation', 'Backward & SGD', 'Quiz', 'Exercises & progress'])
    with tabs[0]:
        st.latex(r'z = xW^T+b,\quad a=f(z)')
        st.latex(r'L=\frac{1}{N}\sum_i(\hat y_i-y_i)^2')
        st.latex(r'\frac{\partial L}{\partial W}=\frac{\partial L}{\partial a}\frac{\partial a}{\partial z}\frac{\partial z}{\partial W}')
        st.latex(r'W_{t+1}=W_t-\eta\nabla_W L')
        for title, explanation in content['concepts']:
            with st.expander(title, expanded=(title == content['concepts'][0][0])):
                st.write(explanation)
        st.caption('For educational regression examples, the final layer is linear and loss is MSE. Classification can use different output/loss configurations in Training Studio.')
    with tabs[1]:
        st.plotly_chart(network_figure(spec.sizes), width="stretch")
        model = adapter.build(spec)
        for name, parameter in model.named_parameters():
            st.write(f'{name} — {tuple(parameter.shape)}')
            st.plotly_chart(tensor_heatmap(parameter.detach().numpy(), f'{name}: shape {tuple(parameter.shape)}'), width="stretch")
        st.json(spec.as_dict())
    with tabs[2]:
        raw = st.text_input('Input vector (comma-separated)', ', '.join(['1'] * inputs), key=f'learn_x_{inputs}')
        try:
            x_values = [float(x.strip()) for x in raw.split(',')]
            _, steps = adapter.trace(spec, x_values)
            for step in steps:
                st.markdown(f'**{step.name}** — shape `{step.shape}`')
                st.caption(step.explanation)
                st.dataframe(step.values, width="stretch")
        except (ValueError, OverflowError) as exc:
            x_values = None
            st.warning(str(exc))
    with tabs[3]:
        raw_target = st.text_input('Target vector (comma-separated)', ', '.join(['0'] * outputs), key=f'learn_y_{outputs}')
        st.markdown('1. `zero_grad()` → 2. `forward()` → 3. MSE → 4. `backward()` → 5. `SGD.step()`')
        try:
            target = [float(t.strip()) for t in raw_target.split(',')]
            if x_values is None:
                raise ValueError('Correct the input vector in Forward propagation first')
            update = adapter.gradient_step(spec, x_values, target, lr)
            c1, c2 = st.columns(2)
            c1.metric('MSE before update', f"{update['loss_before']:.6f}")
            c2.metric('MSE after update', f"{update['loss_after']:.6f}")
            st.plotly_chart(gradient_bar(update['parameters']), width="stretch")
            name_options = [u['parameter'] for u in update['parameters']]
            selected = st.selectbox('Inspect one parameter tensor', name_options, key='learn_param')
            detail = next(u for u in update['parameters'] if u['parameter'] == selected)
            st.dataframe([{'index': i, 'before': old, 'dLoss/dParameter': g, 'after SGD': new}
                          for i, (old,g,new) in enumerate(zip(detail['before'], detail['gradient'], detail['after']))],
                         hide_index=True, width="stretch")
            st.caption('Compare after = before − learning_rate × gradient (within float32 rounding). Each control change recreates the same seeded model and demonstrates one independent update.')
        except (ValueError, OverflowError) as exc:
            st.warning(str(exc))
    with tabs[4]:
        questions = content['questions']
        answers = []
        for i, (question, options, correct, explanation) in enumerate(questions):
            st.write(f'**Q{i+1}. {question}**')
            answers.append(st.radio('Answer', list(range(len(options))), format_func=lambda j, opts=options: opts[j],
                                    index=None, key=f'quiz_{lesson}_{i}'))
        if st.button('Grade quiz', key=f'grade_{lesson}'):
            if any(a is None for a in answers):
                st.warning('Answer all questions before grading.')
            else:
                score = sum(a == q[2] for a, q in zip(answers, questions))
                st.session_state.learning_progress_v1 = record_quiz(progress, lesson, score, len(questions))
                st.success(f'Your score: {score}/{len(questions)}')
                for i, q in enumerate(questions):
                    st.write(f'Q{i+1}: {q[3]}')
        if lesson in progress.get('quiz_scores', {}):
            result = progress['quiz_scores'][lesson]
            st.caption(f"Last recorded score: {result['correct']}/{result['total']}")
    with tabs[5]:
        st.write('Try these experiments; explain what changed and why:')
        for i, task in enumerate(EXERCISES[lesson], 1):
            st.markdown(f'{i}. {task}')
        note = st.text_area('Your experiment notes', value=progress.get('exercise_notes', {}).get(lesson, ''), key=f'notes_{lesson}')
        if st.button('Save notes', key=f'save_notes_{lesson}'):
            updated = dict(progress)
            updated['exercise_notes'] = {**progress.get('exercise_notes', {}), lesson: note}
            st.session_state.learning_progress_v1 = updated
            st.success('Notes saved in this browser session. Export to retain them.')
        if st.button('Mark lesson complete', key=f'complete_{lesson}'):
            st.session_state.learning_progress_v1 = mark_complete(st.session_state.learning_progress_v1, lesson)
            st.success('Lesson marked complete.')
        st.download_button('Export progress JSON', export_progress(st.session_state.learning_progress_v1),
                           file_name='learning-progress.json', mime='application/json')
        upload = st.file_uploader('Restore previously exported progress', type=['json'], key='learn_upload')
        if upload and st.button('Import progress'):
            try:
                st.session_state.learning_progress_v1 = load_progress(upload.getvalue())
                st.success('Progress imported.')
                st.rerun()
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                st.error(f'Invalid progress file: {exc}')
