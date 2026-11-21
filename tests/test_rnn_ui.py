"""Headless runtime integration; does not assert browser layout."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from learning.rnn_ui import SECTIONS
from learning.rnn_content import QUESTIONS, TOPICS

APP = Path(__file__).resolve().parents[1]/'app.py'


def start(section):
    app = AppTest.from_file(str(APP),default_timeout=30).run()
    app.radio(key='workspace').set_value('Learning Studio').run()
    app.selectbox(key='learn_lesson').set_value('rnn').run()
    app.selectbox(key='rnn_section').set_value(section).run()
    assert not app.exception
    return app


@pytest.mark.parametrize('section',SECTIONS)
def test_rnn_sections_render(section):
    assert not start(section).exception


def test_all_concepts_render_and_have_teaching_fields():
    app = start('Concepts & mathematics')
    for i,topic in enumerate(TOPICS):
        assert len(topic)==6 and all(topic)
        app.selectbox(key='rnn_topic').set_value(i).run()
        assert not app.exception


def test_cell_navigation_dimension_change_and_edit_validation():
    app = start('Cell & unrolling')
    app.button(key='rnn_next').click().run()
    assert app.slider(key='rnn_step').value == 1
    app.button(key='rnn_previous').click().run()
    assert app.slider(key='rnn_step').value == 0
    app.slider(key='rnn_cell_i').set_value(1)
    app.slider(key='rnn_cell_h').set_value(1)
    app.slider(key='rnn_cell_t').set_value(2).run()
    assert not app.exception
    app.checkbox(key='rnn_edit_weights').check().run()
    app.text_area[0].set_value('{"w_ih": [[1]], "w_hh": [[0.5]], "b_ih": [0], "b_hh": [0]}').run()
    assert not app.exception and not app.warning
    app.text_area[0].set_value('bad json').run()
    assert app.warning and not app.exception


@pytest.mark.parametrize('task',['classification','forecasting'])
def test_training_evaluation_and_invalidation(task):
    app = start('Training laboratory')
    app.selectbox(key='rnn_train_task').set_value(task)
    app.slider(key='rnn_train_epochs').set_value(1)
    app.slider(key='rnn_train_samples').set_value(60).run()
    app.button(key='rnn_train_button').click().run()
    assert not app.exception and 'rnn_run' in app.session_state
    assert 'rnn_test' not in app.session_state
    app.button(key='rnn_evaluate').click().run()
    assert not app.exception and 'rnn_test' in app.session_state
    app.number_input(key='rnn_train_seed').set_value(9).run()
    assert 'rnn_run' not in app.session_state and 'rnn_test' not in app.session_state


def test_quiz_notes_and_legacy_progress_preserved():
    app = start('Quiz & progress')
    app.session_state.learning_progress_v1['completed'] = ['feed_forward','cnn']
    for i,(_,options,correct,_) in enumerate(QUESTIONS):app.radio(key=f'rnn_quiz_{i}').set_value(options[correct])
    app.button(key='rnn_grade').click().run()
    assert app.session_state.learning_progress_v1['quiz_scores']['rnn']['correct'] == len(QUESTIONS)
    app.text_area(key='rnn_notes').set_value('Shared weights, different states.')
    app.button(key='rnn_save_notes').click().run()
    app.button(key='rnn_complete').click().run()
    p = app.session_state.learning_progress_v1
    assert set(p['completed']) == {'feed_forward','cnn','rnn'}
    assert p['exercise_notes']['rnn'] == 'Shared weights, different states.'


def test_time_first_stacked_sequence_head_and_truncation():
    app = start('Recurrent architecture')
    app.checkbox(key='rnn_arch_batch_first').uncheck()
    app.selectbox(key='rnn_arch_selection').set_value('sequence')
    app.slider(key='rnn_arch_layers').set_value(3).run()
    assert not app.exception
    app.selectbox(key='rnn_section').set_value('BPTT & clipping').run()
    app.slider(key='rnn_bptt_chunk').set_value(4).run()
    assert not app.exception and app.warning
    app.selectbox(key='rnn_section').set_value('Variable lengths').run()
    app.text_input(key='rnn_lengths').set_value('0,2').run()
    assert not app.exception and app.warning
