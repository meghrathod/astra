from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from xgboost import XGBClassifier

def create_autoencoder(input_dim, encoding_dim=32):
    """
    Creates and compiles an autoencoder model for anomaly detection.

    Args:
        input_dim (int): The number of input features.
        encoding_dim (int): The dimension of the encoded representation.

    Returns:
        Model: The compiled Keras autoencoder model.
    """
    # Define the encoder
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(128, activation='relu')(input_layer)
    encoder = Dense(64, activation='relu')(encoder)
    encoder = Dense(encoding_dim, activation='relu')(encoder)

    # Define the decoder
    decoder = Dense(64, activation='relu')(encoder)
    decoder = Dense(128, activation='relu')(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)

    # Create the autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoder)

    # Compile the model
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    return autoencoder

def create_classifier(n_classes, n_estimators=100, max_depth=5, learning_rate=0.1, use_gpu=False):
    """
    Creates an XGBoost classifier.

    Args:
        n_classes (int): The number of classes for the classifier.
        n_estimators (int): The number of boosting rounds.
        max_depth (int): The maximum depth of a tree.
        learning_rate (float): The step size shrinkage.
        use_gpu (bool): Whether to use GPU for training.

    Returns:
        XGBClassifier: The XGBoost classifier model.
    """
    # Set the device based on the use_gpu flag
    tree_method = 'gpu_hist' if use_gpu else 'hist'

    # Create the XGBoost classifier
    classifier = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective='multi:softmax',
        num_class=n_classes,
        tree_method=tree_method,
        random_state=42,
        n_jobs=-1  # Use all available CPU cores
    )

    return classifier
