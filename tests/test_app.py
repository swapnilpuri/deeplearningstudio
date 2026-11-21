"""Streamlit runtime interactions; these do not assert browser appearance."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1]/'app.py'


def start(workspace='Learning Studio',cnn=True):
    app = AppTest.from_file(str(APP),default_timeout=30).run()
    app.radio(key='workspace').set_value(workspace).run()
    if workspace == 'Learning Studio' and cnn:
        app.selectbox(key='learn_lesson').set_value('cnn').run()
    assert not app.exception
    return app


@pytest.mark.parametrize('workspace',['Neural Network Studio','Dataset Management','Training Studio','Learning Studio'])
def test_all_workspaces_render(workspace):
    assert not start(workspace,cnn=False).exception


@pytest.mark.parametrize('section',['Concepts & mathematics','Convolution explorer','Pooling explorer','CNN architecture','Backpropagation','Training laboratory','Quiz & progress'])
def test_cnn_sections_render(section):
    app = start()
    app.selectbox(key='cnn_section').set_value(section).run()
    assert not app.exception


def test_one_cell_convolution_and_pooling_render():
    app = start()
    app.selectbox(key='cnn_section').set_value('Convolution explorer').run()
    app.slider(key='cnn_img_height').set_value(4)
    app.slider(key='cnn_img_width').set_value(4)
    app.slider(key='cnn_kernel').set_value(4)
    app.slider(key='cnn_kernel_width').set_value(4)
    app.slider(key='cnn_padding').set_value(0).run()
    assert not app.exception and not app.warning
    app.selectbox(key='cnn_section').set_value('Pooling explorer').run()
    app.slider(key='cnn_pool_size').set_value(4)
    app.slider(key='cnn_pool_k').set_value(4).run()
    assert not app.exception and not app.warning


def test_cnn_training_final_evaluation_and_config_reset():
    app = start()
    app.selectbox(key='cnn_section').set_value('Training laboratory').run()
    app.slider(key='cnn_epochs').set_value(1)
    app.slider(key='cnn_train_f1').set_value(2)
    app.slider(key='cnn_train_f2').set_value(2)
    app.button[0].click().run()
    assert not app.exception
    assert 'cnn_test_result' not in app.session_state
    next(b for b in app.button if b.label=='Evaluate held-out test set').click().run()
    assert not app.exception and 'cnn_test_result' in app.session_state
    app.number_input(key='cnn_training_seed').set_value(9).run()
    assert 'cnn_test_result' not in app.session_state
    assert 'cnn_experiment' not in app.session_state


def test_dense_lessons_remain_available():
    app = start(cnn=False)
    for lesson in ('feed_forward','deep_neural_network'):
        app.selectbox(key='learn_lesson').set_value(lesson).run()
        assert not app.exception
