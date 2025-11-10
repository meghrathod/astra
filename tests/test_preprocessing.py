import pandas as pd
import numpy as np
from astra_guardian.preprocessing import clean_and_prepare_data, encode_and_split_data

def test_clean_and_prepare_data():
    """
    Tests the clean_and_prepare_data function to ensure it correctly handles
    infinite values and drops the right columns.
    """
    # Create a sample DataFrame with infinite values and unnecessary columns
    data = {
        'Flow.ID': ['flow1'],
        'Source.IP': ['1.1.1.1'],
        'Destination.IP': ['2.2.2.2'],
        'Timestamp': ['2025-01-01 12:00:00'],
        'ProtocolName': ['HTTP'],
        'Source.Port': [80],
        'Destination.Port': [8080],
        'L7Protocol': ['HTTP'],
        'Feature1': [1.0],
        'Feature2': [np.inf]
    }
    df = pd.DataFrame(data)

    # Clean the DataFrame
    df_clean = clean_and_prepare_data(df)

    # Assert that infinite values are handled (row dropped)
    assert df_clean.shape[0] == 0

    # Test with a valid row
    data_valid = {
        'Flow.ID': ['flow1'],
        'Source.IP': ['1.1.1.1'],
        'Destination.IP': ['2.2.2.2'],
        'Timestamp': ['2025-01-01 12:00:00'],
        'ProtocolName': ['HTTP'],
        'Source.Port': [80],
        'Destination.Port': [8080],
        'L7Protocol': ['HTTP'],
        'Feature1': [1.0],
        'Feature2': [2.0]
    }
    df_valid = pd.DataFrame(data_valid)
    df_clean_valid = clean_and_prepare_data(df_valid)

    # Assert that unnecessary columns are dropped
    assert 'Flow.ID' not in df_clean_valid.columns
    assert 'Feature1' in df_clean_valid.columns

def test_encode_and_split_data():
    """
    Tests the encode_and_split_data function to ensure it correctly encodes,
    splits, and scales the data.
    """
    # Create a sample DataFrame
    data = {
        'L7Protocol': ['HTTP', 'DNS', 'HTTP', 'DNS'],
        'Feature1': [1.0, 2.0, 3.0, 4.0],
        'Feature2': [5.0, 6.0, 7.0, 8.0]
    }
    df = pd.DataFrame(data)

    # Encode, split, and scale the data
    X_train, X_test, y_train, y_test, scaler, label_encoder = encode_and_split_data(df, test_size=0.5)

    # Assert that the data is split correctly
    assert X_train.shape[0] == 2
    assert X_test.shape[0] == 2

    # Assert that the data is scaled
    assert np.allclose(X_train.mean(axis=0), 0)
    assert np.allclose(X_train.std(axis=0), 1)

    # Assert that the target is encoded
    assert 'HTTP' not in y_train
