from pathlib import Path
import json
import numpy as np
import pytest
import torch
from streamlit.testing.v1 import AppTest
from learning.transformer_content import SECTIONS, QUESTIONS, evaluate_answers
from learning.transformer_training import TransformerTrainingConfig, train_transformer
from learning.transformer_spec import TransformerSpec
from learning.progress import export_progress, load_progress
from visualization.transformer_graphs import all_heads_figure, block_diagram


def start(section=SECTIONS[0]):
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=45).run()
    app.radio(key='workspace').set_value('Learning Studio').run()
    app.selectbox(key='learn_lesson').set_value('transformer').run()
    app.session_state['m10_section'] = section
    app.run()
    assert not app.exception and not app.warning
    return app


@pytest.mark.parametrize('section',SECTIONS)
def test_sections(section): start(section)


def test_configuration_and_masks():
    app = start(SECTIONS[3])
    app.slider(key='m10_d').set_value(7).run()
    assert app.warning and not app.exception
    app.slider(key='m10_d').set_value(8).run()
    app.selectbox(key='m10_mask').set_value('Causal').run()
    assert not app.warning and not app.exception
    app.selectbox(key='m10_mask').set_value('Padding').run()
    app.slider(key='m10_pad_6').set_value(6).run()
    assert app.warning and not app.exception
    app.selectbox(key='m10_mask').set_value('Bidirectional').run()
    app.selectbox(key='m10_source').set_value('Custom matrix JSON').run()
    app.text_area(key='m10_custom_6_8').set_value('bad').run()
    assert app.warning and not app.exception
    app.selectbox(key='m10_source').set_value('Synthetic demonstration').run()
    app.text_input(key='m10_text').set_value('').run()
    assert app.warning and not app.exception


def test_norm_stage_switch_and_components():
    app = start(SECTIONS[9])
    app.selectbox(key='m10_stage').set_value('LN2').run()
    app.selectbox(key='m10_norm').set_value('Post-LayerNorm').run()
    assert not app.exception and not app.warning
    assert 'residual1' in app.selectbox(key='m10_stage').options
    app.session_state['m10_section'] = SECTIONS[6]
    app.run()
    app.checkbox(key='m10_residual').uncheck().run()
    app.session_state['m10_section'] = SECTIONS[7]
    app.run()
    app.slider(key='m10_gamma').set_value(2.).run()
    app.session_state['m10_section'] = SECTIONS[8]
    app.run()
    app.selectbox(key='m10_activation').set_value('ReLU').run()
    assert not app.warning and not app.exception


def test_explicit_training_reset_and_staleness():
    app = start(SECTIONS[10])
    assert 'm10_run' not in app.session_state
    app.slider(key='m10_steps').set_value(2).run()
    assert 'm10_run' not in app.session_state
    app.button(key='m10_train').click().run()
    assert not app.exception and not app.warning
    initial = app.session_state.m10_run['initial']
    app.button(key='m10_reset').click().run()
    assert 'm10_run' not in app.session_state
    app.button(key='m10_train').click().run()
    for name,value in initial.items():
        assert torch.equal(value,app.session_state.m10_run['initial'][name])
    app.number_input(key='m10_lr').set_value(.02).run()
    assert 'm10_run' not in app.session_state
    app.selectbox(key='learn_lesson').set_value('attention').run()
    assert not app.warning and not app.exception


def test_exercises_and_legacy_progress():
    app = start(SECTIONS[11])
    app.button(key='m10_grade').click().run()
    assert app.warning and not app.exception
    old = dict(schema_version=1,completed=['attention'],quiz_scores={},exercise_notes={'embedding':'keep'})
    app.file_uploader(key='m10_import').upload('progress.json',json.dumps(old).encode()).run()
    app.button(key='m10_load').click().run()
    for i,(_,options,correct,_) in enumerate(QUESTIONS):
        app.radio(key=f'm10_quiz_{i}').set_value(options[correct])
    app.button(key='m10_grade').click().run()
    app.text_area(key='m10_notes').set_value('Heads mix features.')
    app.button(key='m10_save_notes').click().run()
    app.button(key='m10_complete').click().run()
    p = load_progress(export_progress(app.session_state.learning_progress_v1))
    assert set(p['completed']) == {'attention','transformer'}
    assert p['quiz_scores']['transformer']['correct'] == 12
    assert p['exercise_notes'] == {'embedding':'keep','transformer':'Heads mix features.'}
    assert not app.exception and not app.warning
    with pytest.raises(ValueError): evaluate_answers([])


def test_m8_embedding_reuse():
    app = start()
    app.selectbox(key='m10_source').set_value('M8 embedding table').run()
    assert not app.warning
    app.selectbox(key='m10_source').set_value('M8 trained skip-gram').run()
    assert app.warning and not app.exception
    trained = torch.arange(32,dtype=torch.float32).reshape(4,8) / 10
    app.session_state.emb_word_run = dict(trained=trained,vocabulary={'<PAD>':0,'<UNK>':1,'cat':2,'the':3})
    app.run()
    from learning.embedding_adapter import tokenize,encode
    expected = trained[encode(tokenize('The cat sat on the mat'),app.session_state.emb_word_run['vocabulary'])].numpy()
    assert not app.exception and not app.warning
    assert any(df.value.shape==expected.shape and np.allclose(df.value.to_numpy(),expected)
               for df in app.dataframe if df.value.shape == expected.shape)


def test_plot_data_consistency():
    values = torch.softmax(torch.randn(3,4,4),-1)
    fig = all_heads_figure(values,list('ABCD'))
    for i,trace in enumerate(fig.data):
        np.testing.assert_allclose(trace.z, values[i].numpy())
        assert trace.zmin == 0 and trace.zmax == 1
        assert list(trace.x) == list('ABCD') and list(trace.y) == list('ABCD')
    assert 'X -> N1 -> A' in block_diagram('Pre-LayerNorm')
    assert 'X -> A -> R1' in block_diagram('Post-LayerNorm')


@pytest.mark.parametrize('norm,optimizer,causal', [('Pre-LayerNorm','Adam',False),
    ('Post-LayerNorm','Adam',True),('Pre-LayerNorm','SGD',True)])
def test_training(norm,optimizer,causal):
    config = TransformerTrainingConfig(TransformerSpec(norm=norm),steps=20,optimizer=optimizer,causal=causal)
    a,b = train_transformer(config),train_transformer(config)
    assert a['losses'] == b['losses']
    assert a['losses'][-1] < a['losses'][0]
    for name,g in a['gradients'].items():
        assert torch.isfinite(g).all()
        assert not torch.equal(a['initial'][name],a['trained'][name])


def test_default_loss_reduction():
    run = train_transformer(TransformerTrainingConfig())
    assert run['losses'][-1] < .1 * run['losses'][0]


@pytest.mark.parametrize('kwargs',[dict(steps=0),dict(steps=True),dict(optimizer='bad'),
    dict(learning_rate=float('nan')),dict(learning_rate=0),dict(causal='yes')])
def test_training_invalid(kwargs):
    with pytest.raises(ValueError): TransformerTrainingConfig(**kwargs)
