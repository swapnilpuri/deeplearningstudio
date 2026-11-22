"""M11 real-app navigation, state retention, explicit training and curriculum."""
from pathlib import Path
import json
import pytest
from streamlit.testing.v1 import AppTest
from learning.seq2seq_content import SECTIONS, QUESTIONS, evaluate_answers
from learning import seq2seq_ui

APP = Path(__file__).resolve().parents[1] / 'app.py'


def start():
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    app.radio(key='workspace').set_value('Learning Studio').run()
    app.selectbox(key='learn_lesson').set_value('seq2seq').run()
    assert not app.exception
    return app


def section(app, name):
    app.session_state.m11_section = name
    app.run()
    assert not app.exception
    return app


def test_all_tabs_lazy_no_training_and_namespaced_keys(monkeypatch):
    def unexpected(*args):
        raise AssertionError('Training must require button action')
    monkeypatch.setattr(seq2seq_ui, 'train_seq2seq', unexpected)
    app = start()
    assert [tab.label for tab in app.tabs] == list(SECTIONS)
    assert not any(w.key == 'm11_section' for w in app.selectbox)
    for name in SECTIONS:
        section(app, name)
        assert not app.warning
        assert [t.label for t in app.tabs if len(t.children)] == [name]
        for kind in ('slider', 'number_input', 'text_input', 'text_area', 'button'):
            assert all(w.key.startswith('m11_') for w in app.get(kind))


def test_progress_quiz_notes_and_model_survive_sections():
    app = start()
    model = app.session_state.m11_demo_model
    section(app, 'Quiz')
    app.button(key='m11_grade').click().run()
    assert app.warning and not app.exception
    for i, (_, options, correct, _) in enumerate(QUESTIONS):
        app.radio(key=f'm11_quiz_{i}').set_value(options[correct])
    app.button(key='m11_grade').click().run()
    assert app.session_state.learning_progress_v1['quiz_scores']['seq2seq']['correct'] == 15
    section(app, 'Exercises & Progress')
    app.text_area(key='m11_notes').set_value('Keep this draft').run()
    app.button(key='m11_save_notes').click().run()
    app.button(key='m11_complete').click().run()
    section(app, 'Concepts & Math')
    assert app.session_state.m11_demo_model is model
    section(app, 'Quiz')
    assert app.radio(key='m11_quiz_0').value == QUESTIONS[0][1][QUESTIONS[0][2]]
    assert app.success
    section(app, 'Exercises & Progress')
    assert app.text_area(key='m11_notes').value == 'Keep this draft'
    assert 'seq2seq' in app.session_state.learning_progress_v1['completed']


def test_explicit_training_reset_and_saved_config(monkeypatch):
    calls = []
    train = seq2seq_ui.train_seq2seq
    def counted(config):
        calls.append(config)
        return train(config)
    monkeypatch.setattr(seq2seq_ui, 'train_seq2seq', counted)
    app = start()
    section(app, 'Seq2Seq & Training')
    app.slider(key='m11_steps').set_value(2).run()
    assert not calls
    app.button(key='m11_train').click().run()
    assert not app.exception and not app.warning
    losses = list(app.session_state.m11_run['losses'])
    for name in ('Decoder', 'Quiz', 'Cross-Attention', 'Seq2Seq & Training'):
        section(app, name)
        assert not app.warning
    assert len(calls) == 1
    assert app.slider(key='m11_steps').value == 2
    assert app.session_state.m11_run['losses'] == losses
    app.slider(key='m11_steps').set_value(3).run()
    assert app.session_state.m11_run['losses'] == losses and len(calls) == 1
    app.button(key='m11_generate').click().run()
    assert 'm11_generation' in app.session_state
    app.button(key='m11_reset').click().run()
    assert 'm11_run' not in app.session_state and len(calls) == 1


def test_invalid_builder_and_unequal_lengths():
    app = start()
    app.slider(key='m11_d').set_value(7).run()
    assert app.warning and not app.exception
    app.slider(key='m11_d').set_value(8).run()
    app.text_input(key='m11_source').set_value('3 4 5 2 0').run()
    section(app, 'Cross-Attention')
    assert not app.warning
    app.selectbox(key='m11_cross_key').set_value(4).run()
    assert not app.exception
    section(app, 'Decoder')
    for mask in ('Encoder source padding', 'Decoder causal + target padding', 'Cross source padding'):
        app.selectbox(key='m11_mask_kind').set_value(mask).run()
        assert not app.warning and not app.exception
    app.text_input(key='m11_source').set_value('0 0').run()
    assert app.warning and not app.exception


def test_curriculum_correctness_and_incomplete_feedback():
    assert len(QUESTIONS) == 15
    assert evaluate_answers([options[correct] for _, options, correct, _ in QUESTIONS]) == 15
    with pytest.raises(ValueError):
        evaluate_answers([None]*15)


def test_imported_notes_replace_editor_draft_and_preserve_other_lessons(monkeypatch):
    imported = dict(schema_version=1, completed=['seq2seq', 'transformer'],
        quiz_scores={'transformer': {'correct': 2, 'total': 3}},
        exercise_notes={'seq2seq': 'Imported M11 notes', 'transformer': 'Earlier lesson notes'})
    class TestUpload:
        def getvalue(self):
            return json.dumps(imported).encode()
    original = seq2seq_ui.st.file_uploader
    def uploader(*args, **kwargs):
        return TestUpload() if kwargs.get('key') == 'm11_import' else original(*args, **kwargs)
    monkeypatch.setattr(seq2seq_ui.st, 'file_uploader', uploader)
    app = start()
    section(app, 'Exercises & Progress')
    app.text_area(key='m11_notes').set_value('Unsaved draft').run()
    app.button(key='m11_load').click().run()
    assert not app.exception and not app.warning
    assert app.session_state.learning_progress_v1 == imported
    assert app.text_area(key='m11_notes').value == 'Imported M11 notes'
    section(app, 'Quiz')
    section(app, 'Exercises & Progress')
    assert app.text_area(key='m11_notes').value == 'Imported M11 notes'
    app.button(key='m11_save_notes').click().run()
    assert app.session_state.learning_progress_v1['exercise_notes'] == imported['exercise_notes']
