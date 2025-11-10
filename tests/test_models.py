from astra_guardian.models import create_autoencoder, create_classifier
import numpy as np

def test_create_autoencoder():
    """
    Tests the create_autoencoder function to ensure it returns a compiled Keras
    model with the correct input and output shapes.
    """
    input_dim = 64
    autoencoder = create_autoencoder(input_dim)

    # Assert that the model is a Keras model
    assert hasattr(autoencoder, 'compile')

    # Assert that the input and output shapes are correct
    assert autoencoder.input_shape == (None, input_dim)
    assert autoencoder.output_shape == (None, input_dim)

    # Assert that the model is compiled
    assert autoencoder.optimizer is not None
    assert autoencoder.loss is not None

def test_create_classifier():
    """
    Tests the create_classifier function to ensure it returns an XGBoost
    classifier with the correct parameters.
    """
    n_classes = 10
    classifier = create_classifier(n_classes=n_classes)

    # Assert that the model is an XGBoost classifier
    assert 'XGBClassifier' in str(type(classifier))

    # Assert that the parameters are set correctly
    assert classifier.get_params()['n_estimators'] == 100
    assert classifier.get_params()['max_depth'] == 5
    assert classifier.get_params()['learning_rate'] == 0.1
    assert classifier.get_params()['objective'] == 'multi:softmax'
    assert classifier.get_params()['num_class'] == n_classes
