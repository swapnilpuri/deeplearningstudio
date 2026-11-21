"""Headless workflow checks, distinct from rendered browser acceptance."""
from pathlib import Path
import json
import pytest
from streamlit.testing.v1 import AppTest
from learning.lstm_ui import SECTIONS
from learning.lstm_content import TOPICS, QUESTIONS
from learning.progress import load_progress, export_progress

APP = Path(__file__).resolve().parents[1]/'app.py'


def start(section):
    app = AppTest.from_file(str(APP),default_timeout=40).run()
    app.radio(key='workspace').set_value('Learning Studio').run()
    app.selectbox(key='learn_lesson').set_value('lstm').run()
    app.selectbox(key='lstm_section').set_value(section).run()
    assert not app.exception
    return app


@pytest.mark.parametrize('section',SECTIONS)
def test_sections(section):
    assert not start(section).exception


def test_content_and_concepts():
    app = start(SECTIONS[0])
    assert len(TOPICS)==24 and len(QUESTIONS)>=14
    for i,topic in enumerate(TOPICS):
        assert len(topic)==7 and all(topic)
        app.selectbox(key='lstm_topic').set_value(i).run()
        assert not app.exception


def test_cell_navigation_and_json_validation():
    app = start('Cell & unrolling')
    app.button(key='lstm_next').click().run();assert app.slider(key='lstm_step').value==1
    app.button(key='lstm_previous').click().run();assert app.slider(key='lstm_step').value==0
    app.slider(key='lstm_cell_i').set_value(1)
    app.slider(key='lstm_cell_h').set_value(1)
    app.slider(key='lstm_cell_t').set_value(2).run()
    app.checkbox(key='lstm_edit').check().run()
    app.text_area[0].set_value(json.dumps(dict(w_ih=[[.2]]*4,w_hh=[[.1]]*4,b_ih=[0]*4,b_hh=[0]*4,h0=[.2],c0=[.4]))).run()
    assert not app.exception and not app.warning
    app.text_area[0].set_value('bad json').run();assert app.warning and not app.exception


@pytest.mark.parametrize('task',['memory','forecasting'])
def test_training_explicit_test_and_invalidation(task):
    app = start('Training laboratory')
    app.selectbox(key='lstm_train_task').set_value(task)
    app.slider(key='lstm_train_samples').set_value(60)
    app.slider(key='lstm_train_epochs').set_value(1).run()
    app.button(key='lstm_train_button').click().run()
    assert not app.exception and 'lstm_train_run' in app.session_state
    assert 'lstm_train_test' not in app.session_state
    app.button(key='lstm_train_evaluate').click().run()
    assert not app.exception and 'lstm_train_test' in app.session_state
    app.number_input(key='lstm_train_seed').set_value(9).run()
    assert 'lstm_train_run' not in app.session_state and 'lstm_train_test' not in app.session_state
    app.slider(key='lstm_train_length').set_value(2).run();assert not app.exception


def test_comparison_sweep_and_explicit_test():
    app = start('RNN versus LSTM')
    app.slider(key='lstm_compare_samples').set_value(60)
    app.slider(key='lstm_compare_epochs').set_value(1).run()
    app.button(key='lstm_compare_button').click().run()
    assert not app.exception and set(app.session_state['lstm_compare_run'])=={'RNN','LSTM'}
    assert 'lstm_compare_test' not in app.session_state
    app.button(key='lstm_compare_evaluate').click().run()
    assert not app.exception and len(app.session_state['lstm_compare_test'])==2
    app.button(key='lstm_compare_sweep_button').click().run()
    assert not app.exception and len(app.session_state['lstm_compare_sweep'])==6


def test_builder_and_experiments():
    app = start('LSTM architecture')
    app.checkbox(key='lstm_arch_first').uncheck()
    app.selectbox(key='lstm_arch_selection').set_value('sequence')
    app.slider(key='lstm_arch_layers').set_value(3)
    app.slider(key='lstm_arch_dropout').set_value(.2).run()
    assert not app.exception
    app.slider(key='lstm_arch_layers').set_value(1).run();assert app.warning and not app.exception
    app.selectbox(key='lstm_section').set_value('Gate & state laboratory').run()
    app.checkbox(key='lstm_memory_reverse').check()
    app.slider(key='lstm_memory_f').set_value(0.).run();assert not app.exception
    app.selectbox(key='lstm_section').set_value('BPTT & clipping').run()
    app.selectbox(key='lstm_bptt_optimizer').set_value('Adam').run();assert not app.exception
    app.selectbox(key='lstm_section').set_value('Variable lengths').run()
    app.text_input(key='lstm_lengths').set_value('0,2').run();assert app.warning and not app.exception


def test_progress_preserves_prior_lessons():
    app = start('Quiz & progress')
    app.session_state.learning_progress_v1 = load_progress(json.dumps(dict(schema_version=1,completed=['feed_forward','cnn','rnn'],quiz_scores={'rnn':{'correct':1,'total':2}},exercise_notes={'cnn':'keep'})))
    for i,(_,options,correct,_) in enumerate(QUESTIONS):app.radio(key=f'lstm_quiz_{i}').set_value(options[correct])
    app.button(key='lstm_grade').click().run()
    app.text_area(key='lstm_notes').set_value('Memory and exposure differ.')
    app.button(key='lstm_save_notes').click().run()
    app.button(key='lstm_complete').click().run()
    progress = load_progress(export_progress(app.session_state.learning_progress_v1))
    assert set(progress['completed'])=={'feed_forward','cnn','rnn','lstm'}
    assert progress['exercise_notes']=={'cnn':'keep','lstm':'Memory and exposure differ.'}
    assert progress['quiz_scores']['lstm']['correct']==len(QUESTIONS)
