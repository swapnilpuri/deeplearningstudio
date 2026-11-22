"""Headless application workflows, separate from browser visual acceptance."""
from pathlib import Path
import json
import pytest
from streamlit.testing.v1 import AppTest
from learning.embedding_ui import SECTIONS
from learning.embedding_content import TOPICS, QUESTIONS
from learning.progress import load_progress, export_progress

APP = Path(__file__).resolve().parents[1]/'app.py'


def start(section):
    app = AppTest.from_file(str(APP),default_timeout=45).run()
    app.radio(key='workspace').set_value('Learning Studio').run()
    app.selectbox(key='learn_lesson').set_value('embedding').run()
    app.session_state['emb_section'] = section
    app.run()
    assert not app.exception and not app.warning
    return app


@pytest.mark.parametrize('section',SECTIONS)
def test_sections(section):
    start(section)


def test_concepts_and_tokenization_unknown_empty_character():
    app = start(SECTIONS[0])
    assert len(TOPICS)==22 and len(QUESTIONS)>=17
    for i,topic in enumerate(TOPICS):
        assert len(topic)==7 and all(topic)
        app.selectbox(key='emb_topic').set_value(i).run()
        assert not app.exception and not app.warning
    app.session_state['emb_section'] = 'Tokenization & vocabulary'
    app.run()
    app.text_area(key='emb_text').set_value('').run()
    app.selectbox(key='emb_tokenizer').set_value('character').run()
    app.selectbox(key='emb_vocab_source').set_value('Custom corpus').run()
    app.text_area(key='emb_corpus').set_value('').run()
    assert not app.exception and not app.warning


def test_matrix_custom_validation_and_gradient_optimizers():
    app = start('Matrix & one-hot')
    app.checkbox(key='emb_custom').check().run()
    editor = app.text_area[0]
    editor.set_value('invalid').run();assert app.warning and not app.exception
    app.checkbox(key='emb_custom').uncheck().run()
    app.text_input[0].set_value('-1').run();assert app.warning and not app.exception
    app.session_state['emb_section'] = 'Gradients & updates'
    app.run()
    app.selectbox(key='emb_grad_optimizer').set_value('Adam').run()
    assert not app.exception and not app.warning


def test_word_training_retrieval_pooling_and_invalidation():
    app = start('Word embedding training')
    app.slider(key='emb_word_epochs').set_value(1)
    app.selectbox(key='emb_word_batch').set_value(256).run()
    app.button(key='emb_word_train').click().run()
    assert 'emb_word_run' in app.session_state and not app.exception and not app.warning
    app.selectbox(key='emb_space_axes').set_value(3).run()
    assert not app.exception and not app.warning
    app.session_state['emb_section'] = 'Similarity & retrieval'
    app.run()
    app.selectbox(key='emb_sim_source').set_value('Trained skip-gram').run()
    app.selectbox(key='emb_metric').set_value('dot')
    app.checkbox(key='emb_retrieval_normalize').check().run()
    assert not app.exception and not app.warning
    app.session_state['emb_section'] = 'Sentence pooling'
    app.run()
    app.text_input(key='emb_sentence_a').set_value('')
    app.selectbox(key='emb_pooling').set_value('max').run()
    assert not app.exception and not app.warning
    app.session_state['emb_section'] = 'Word embedding training'
    app.run()
    app.number_input(key='emb_word_seed').set_value(9).run()
    assert 'emb_word_run' not in app.session_state


def test_positions_builder_validation_and_serialized_spec():
    app = start('Positions & context')
    app.selectbox(key='emb_pos_type').set_value('sinusoidal')
    app.slider(key='emb_pos_dim').set_value(1)
    app.checkbox(key='emb_reverse').check().run()
    assert not app.exception and not app.warning
    app.session_state['emb_section'] = 'Architecture builder'
    app.run()
    app.selectbox(key='emb_arch_pos').set_value('learned')
    app.slider(key='emb_arch_max').set_value(1).run()
    assert app.warning and not app.exception
    app.slider(key='emb_arch_max').set_value(12).run()
    app.selectbox(key='emb_arch_pool').set_value('max').run()
    assert not app.exception and not app.warning


def test_classifier_training_test_and_invalidation():
    app = start('Classification laboratory')
    app.slider(key='emb_class_epochs').set_value(1).run()
    app.button(key='emb_class_train').click().run()
    assert not app.exception and not app.warning and 'emb_class_run' in app.session_state
    assert 'emb_class_test' not in app.session_state
    app.button(key='emb_class_evaluate').click().run()
    assert not app.exception and not app.warning and 'emb_class_test' in app.session_state
    app.selectbox(key='emb_class_optimizer').set_value('SGD').run()
    assert 'emb_class_run' not in app.session_state and 'emb_class_test' not in app.session_state


def test_progress_quiz_notes_and_legacy_import():
    app = start('Quiz & progress')
    old = dict(schema_version=1,completed=['feed_forward','deep','cnn','rnn','lstm'],quiz_scores={'lstm':{'correct':2,'total':3}},exercise_notes={'rnn':'keep'})
    app.file_uploader(key='emb_progress_import').upload('old.json',json.dumps(old).encode()).run()
    app.button(key='emb_load_progress').click().run()
    for i,(_,options,correct,_) in enumerate(QUESTIONS):app.radio(key=f'emb_quiz_{i}').set_value(options[correct])
    app.button(key='emb_grade').click().run()
    app.text_area(key='emb_notes').set_value('Lookup is not contextualization.')
    app.button(key='emb_save_notes').click().run()
    app.button(key='emb_complete').click().run()
    progress = load_progress(export_progress(app.session_state.learning_progress_v1))
    assert set(progress['completed']) == set(old['completed'])|{'embedding'}
    assert progress['quiz_scores']['embedding']['correct'] == len(QUESTIONS)
    assert progress['quiz_scores']['lstm'] == old['quiz_scores']['lstm']
    assert progress['exercise_notes'] == {'rnn':'keep','embedding':'Lookup is not contextualization.'}
    assert not app.exception
