"""Navigation contracts and cross-section state, using the real app runtime."""
from copy import deepcopy
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from learning.attention_content import SECTIONS as ATTENTION
from learning.embedding_ui import SECTIONS as EMBEDDING
from learning.lstm_ui import SECTIONS as LSTM
from learning.rnn_ui import SECTIONS as RNN
from learning.transformer_content import SECTIONS as TRANSFORMER

APP = Path(__file__).resolve().parents[1] / 'app.py'
CNN = ['Concepts & mathematics', 'Convolution explorer', 'Pooling explorer',
       'CNN architecture', 'Backpropagation', 'Training laboratory', 'Quiz & progress']
FOUNDATIONS = ['Concepts & math', 'Architecture & tensors', 'Forward propagation',
               'Backward & SGD', 'Quiz', 'Exercises & progress']
MODULES = [('feed_forward', 'learn', FOUNDATIONS), ('deep_neural_network', 'learn', FOUNDATIONS),
           ('cnn', 'cnn', CNN), ('rnn', 'rnn', RNN), ('lstm', 'lstm', LSTM),
           ('embedding', 'emb', EMBEDDING), ('attention', 'att', ATTENTION),
           ('transformer', 'm10', TRANSFORMER)]


def start(lesson):
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    app.radio(key='workspace').set_value('Learning Studio').run()
    app.selectbox(key='learn_lesson').set_value(lesson).run()
    assert not app.exception
    return app


@pytest.mark.parametrize('lesson,prefix,sections', MODULES)
def test_all_learning_paths_have_ordered_native_tabs(lesson, prefix, sections, monkeypatch):
    import importlib

    def unexpected_training(*args, **kwargs):
        raise AssertionError('Navigation must never start training')

    for name, functions in [('cnn', ('train_digits',)), ('rnn', ('train_sequences',)),
                            ('lstm', ('train_lstm',)),
                            ('embedding', ('train_skipgram', 'train_classifier')),
                            ('attention', ('train_attention',)),
                            ('transformer', ('train_transformer',))]:
        module = importlib.import_module(f'learning.{name}_ui')
        for function in functions:
            monkeypatch.setattr(module, function, unexpected_training)
    app = start(lesson)
    assert [tab.label for tab in app.tabs] == list(sections)
    assert not any(widget.key == f'{prefix}_section' for widget in app.selectbox)
    if prefix == 'learn':
        return  # Eager foundations keep their original cross-panel dependencies.
    for section in sections:
        app.session_state[f'{prefix}_section'] = section
        app.run()
        assert not app.exception
        assert app.session_state[f'{prefix}_section'] == section
        active = [tab for tab in app.tabs if len(tab.children)]
        assert [tab.label for tab in active] == [section]


@pytest.mark.parametrize('lesson,prefix,sections', MODULES[2:])
def test_progress_and_draft_answers_survive_navigation(lesson, prefix, sections):
    app = start(lesson)
    progress = dict(schema_version=1, completed=[lesson],
                    quiz_scores={lesson: {'correct': 1, 'total': 2}},
                    exercise_notes={lesson: 'Saved experiment'})
    app.session_state.learning_progress_v1 = deepcopy(progress)
    app.session_state[f'{prefix}_section'] = sections[-1]
    app.run()
    answer = app.radio[0]
    answer.set_value(answer.options[0]).run()
    answer_key = answer.key
    saved_answer = app.session_state[answer_key]
    notes = app.text_area[0]
    notes.set_value('Unsubmitted draft').run()
    notes_key = notes.key
    app.session_state[f'{prefix}_section'] = sections[0]
    app.run()
    app.session_state[f'{prefix}_section'] = sections[-1]
    app.run()
    assert not app.exception
    assert app.session_state[answer_key] == saved_answer
    assert app.text_area(key=notes_key).value == 'Unsubmitted draft'
    assert app.session_state.learning_progress_v1 == progress


def test_training_result_and_configuration_survive_tabs_without_retraining(monkeypatch):
    from learning import transformer_ui

    calls = []
    train = transformer_ui.train_transformer

    def counted(config):
        calls.append(config)
        return train(config)

    monkeypatch.setattr(transformer_ui, 'train_transformer', counted)
    app = start('transformer')
    app.session_state.m10_section = TRANSFORMER[10]
    app.run()
    app.slider(key='m10_steps').set_value(2).run()
    assert not calls and 'm10_run' not in app.session_state
    app.button(key='m10_train').click().run()
    losses = list(app.session_state.m10_run['losses'])
    for section in (TRANSFORMER[0], TRANSFORMER[11], TRANSFORMER[10]):
        app.session_state.m10_section = section
        app.run()
        assert not app.exception
    assert len(calls) == 1
    assert app.slider(key='m10_steps').value == 2
    assert app.session_state.m10_run['losses'] == losses


def test_distinct_ids_labels_and_module_keys():
    app = AppTest.from_string('''
import streamlit as st
from learning.navigation import render_learning_section_navigation
for module in ('first', 'second'):
    for section, tab in render_learning_section_navigation(
        {'intro': 'Concepts', 'quiz': 'Quiz'}, module).items():
        if tab.open:
            with tab:
                st.write(module + ':' + section)
''').run()
    app.session_state.first_section = 'Quiz'
    app.run()
    assert not app.exception
    assert app.session_state.first_section == 'Quiz'
    assert app.session_state.second_section == 'Concepts'
    assert [m.value for m in app.markdown] == ['first:quiz', 'second:intro']
