from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from learning.attention_content import SECTIONS, QUESTIONS, evaluate_answers
from learning.progress import load_progress, export_progress


def start(section=SECTIONS[0]):
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=45).run()
    app.radio(key='workspace').set_value('Learning Studio').run()
    app.selectbox(key='learn_lesson').set_value('attention').run()
    app.selectbox(key='att_section').set_value(section).run()
    assert not app.exception and not app.warning
    return app


@pytest.mark.parametrize('section',SECTIONS)
def test_sections(section): start(section)


def test_inputs_and_mask_validation():
    app = start(SECTIONS[4])
    app.selectbox(key='att_mask').set_value('Causal').run()
    assert not app.exception and not app.warning
    app.selectbox(key='att_mask').set_value('Padding').run()
    app.slider(key='att_pad_6').set_value(6).run()
    assert app.warning and not app.exception
    app.selectbox(key='att_mask').set_value('Bidirectional').run()
    app.text_input(key='att_text').set_value('').run()
    assert app.warning and not app.exception
    app.text_input(key='att_text').set_value('cat bank').run()
    app.selectbox(key='att_source').set_value('Custom matrix JSON').run()
    app.text_area(key='att_custom_2_4').set_value('bad').run()
    assert app.warning and not app.exception


def test_training_explicit_reset_and_isolation():
    app = start(SECTIONS[7])
    assert 'att_run' not in app.session_state
    app.slider(key='att_steps').set_value(3).run()
    app.button(key='att_train').click().run()
    assert not app.exception and 'att_run' in app.session_state
    app.button(key='att_reset').click().run()
    assert 'att_run' not in app.session_state
    app.button(key='att_train').click().run()
    app.number_input(key='att_lr').set_value(.05).run()
    assert 'att_run' not in app.session_state
    app.selectbox(key='learn_lesson').set_value('embedding').run()
    assert not app.exception
    app.selectbox(key='learn_lesson').set_value('attention').run()
    assert not app.exception


def test_exercises_progress():
    app = start(SECTIONS[8])
    for i,(_,options,correct,_) in enumerate(QUESTIONS): app.radio(key=f'att_quiz_{i}').set_value(options[correct])
    app.button(key='att_grade').click().run()
    app.text_area(key='att_notes').set_value('Values are mixed.')
    app.button(key='att_save_notes').click().run()
    app.button(key='att_complete').click().run()
    p = load_progress(export_progress(app.session_state.learning_progress_v1))
    assert p['quiz_scores']['attention']['correct']==9
    assert 'attention' in p['completed'] and p['exercise_notes']['attention']=='Values are mixed.'
    assert not app.exception
    with pytest.raises(ValueError): evaluate_answers([])


def test_m8_learned_weights_and_manual_projections():
    app = start(SECTIONS[1])
    app.selectbox(key='att_source').set_value('M8 embedding table').run()
    app.checkbox(key='att_edit').check().run()
    assert not app.exception and not app.warning
    app.selectbox(key='learn_lesson').set_value('embedding').run()
    app.selectbox(key='emb_section').set_value('Word embedding training').run()
    app.slider(key='emb_word_epochs').set_value(1).run()
    app.button(key='emb_word_train').click().run()
    app.selectbox(key='learn_lesson').set_value('attention').run()
    app.checkbox(key='att_edit').uncheck().run()
    app.selectbox(key='att_source').set_value('M8 trained skip-gram').run()
    assert not app.exception and not app.warning
    source = app.session_state.emb_word_run
    from learning.embedding_adapter import encode, tokenize
    expected = source['trained'][encode(tokenize('The cat sat on the mat'),source['vocabulary'])].numpy()
    import numpy as np
    assert any(df.value.shape==expected.shape and np.allclose(df.value.to_numpy(),expected) for df in app.dataframe if df.value.shape==expected.shape)


def test_progress_legacy_import():
    import json
    old = dict(schema_version=1,completed=['embedding','rnn'],quiz_scores={'embedding':{'correct':2,'total':3}},exercise_notes={'rnn':'keep'})
    app = start(SECTIONS[8])
    app.file_uploader(key='att_import').upload('legacy.json',json.dumps(old).encode()).run()
    app.button(key='att_load').click().run()
    app.button(key='att_complete').click().run()
    p = app.session_state.learning_progress_v1
    assert set(p['completed'])=={'attention','embedding','rnn'}
    assert p['quiz_scores']==old['quiz_scores'] and p['exercise_notes']==old['exercise_notes']
