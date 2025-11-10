import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clean_and_prepare_data(df: pd.DataFrame, target_column: str = 'L7Protocol'):
    """
    Cleans and prepares the network traffic data by handling infinite values and
    dropping identifier columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target variable column.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with any remaining NaN values
    df.dropna(inplace=True)

    # Define columns to drop (identifiers and potentially redundant features)
    cols_to_drop = [
        'Flow.ID', 'Source.IP', 'Destination.IP', 'Timestamp',
        'ProtocolName', 'Source.Port', 'Destination.Port'
    ]

    # Drop columns that exist in the DataFrame
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols_to_drop)

    return df

def encode_and_split_data(df: pd.DataFrame, target_column: str = 'L7Protocol', test_size: float = 0.2, random_state: int = 42):
    """
    Encodes the target variable, splits the data into training and testing sets,
    and scales the features.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        target_column (str): The name of the target variable column.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test, scaler, and label_encoder.
    """
    # Encode the target variable
    label_encoder = LabelEncoder()
    df[target_column] = label_encoder.fit_transform(df[target_column])

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the data, ensuring stratification to handle class imbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder
