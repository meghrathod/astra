import os
import pandas as pd
import joblib
import numpy as np
import gc
from astra_guardian.preprocessing import clean_and_prepare_data, encode_and_split_data
from astra_guardian.models import create_autoencoder, create_classifier

def train_models():
    """
    Loads the data, preprocesses it, trains the classifier and anomaly detector,
    and saves the artifacts.
    """
    # Define data types to reduce memory usage
    dtype_mapping = {
        'Flow.Duration': 'float32',
        'Total.Fwd.Packets': 'uint32',
        'Total.Backward.Packets': 'uint32',
        'Total.Length.of.Fwd.Packets': 'uint32',
        'Total.Length.of.Bwd.Packets': 'uint32',
        'Fwd.Packet.Length.Max': 'uint16',
        'Fwd.Packet.Length.Min': 'uint16',
        'Fwd.Packet.Length.Mean': 'float32',
        'Fwd.Packet.Length.Std': 'float32',
        'Bwd.Packet.Length.Max': 'uint16',
        'Bwd.Packet.Length.Min': 'uint16',
        'Bwd.Packet.Length.Mean': 'float32',
        'Bwd.Packet.Length.Std': 'float32',
        'Flow.Bytes.s': 'float32',
        'Flow.Packets.s': 'float32',
        'Flow.IAT.Mean': 'float32',
        'Flow.IAT.Std': 'float32',
        'Flow.IAT.Max': 'float32',
        'Flow.IAT.Min': 'float32',
        'Fwd.IAT.Total': 'float32',
        'Fwd.IAT.Mean': 'float32',
        'Fwd.IAT.Std': 'float32',
        'Fwd.IAT.Max': 'float32',
        'Fwd.IAT.Min': 'float32',
        'Bwd.IAT.Total': 'float32',
        'Bwd.IAT.Mean': 'float32',
        'Bwd.IAT.Std': 'float32',
        'Bwd.IAT.Max': 'float32',
        'Bwd.IAT.Min': 'float32',
        'Fwd.PSH.Flags': 'uint8',
        'Bwd.PSH.Flags': 'uint8',
        'Fwd.URG.Flags': 'uint8',
        'Bwd.URG.Flags': 'uint8',
        'Fwd.Header.Length': 'uint32',
        'Bwd.Header.Length': 'uint32',
        'Fwd.Packets.s': 'float32',
        'Bwd.Packets.s': 'float32',
        'Min.Packet.Length': 'uint16',
        'Max.Packet.Length': 'uint16',
        'Packet.Length.Mean': 'float32',
        'Packet.Length.Std': 'float32',
        'Packet.Length.Variance': 'float32',
        'FIN.Flag.Count': 'uint8',
        'SYN.Flag.Count': 'uint8',
        'RST.Flag.Count': 'uint8',
        'PSH.Flag.Count': 'uint8',
        'ACK.Flag.Count': 'uint8',
        'URG.Flag.Count': 'uint8',
        'CWE.Flag.Count': 'uint8',
        'ECE.Flag.Count': 'uint8',
        'Down.Up.Ratio': 'float32',
        'Average.Packet.Size': 'float32',
        'Avg.Fwd.Segment.Size': 'float32',
        'Avg.Bwd.Segment.Size': 'float32',
        'Fwd.Header.Length.1': 'uint32',
        'Fwd.Avg.Bytes.Bulk': 'uint8',
        'Fwd.Avg.Packets.Bulk': 'uint8',
        'Fwd.Avg.Bulk.Rate': 'uint8',
        'Bwd.Avg.Bytes.Bulk': 'uint8',
        'Bwd.Avg.Packets.Bulk': 'uint8',
        'Bwd.Avg.Bulk.Rate': 'uint8',
        'Subflow.Fwd.Packets': 'uint32',
        'Subflow.Fwd.Bytes': 'uint32',
        'Subflow.Bwd.Packets': 'uint32',
        'Subflow.Bwd.Bytes': 'uint32',
        'Init_Win_bytes_forward': 'int32',
        'Init_Win_bytes_backward': 'int32',
        'act_data_pkt_fwd': 'uint32',
        'min_seg_size_forward': 'uint8',
        'Active.Mean': 'float32',
        'Active.Std': 'float32',
        'Active.Max': 'float32',
        'Active.Min': 'float32',
        'Idle.Mean': 'float32',
        'Idle.Std': 'float32',
        'Idle.Max': 'float32',
        'Idle.Min': 'float32'
    }

    # Load the dataset
    data_path = 'data/Dataset-Unicauca-Version2-87Atts.csv'
    df = pd.read_csv(data_path, dtype=dtype_mapping)

    # Define "normal" traffic
    normal_apps = ['GOOGLE', 'YOUTUBE', 'FACEBOOK', 'MICROSOFT', 'CLOUDFLARE', 'AMAZON']
    normal_df = df[df['ProtocolName'].isin(normal_apps)].copy()

    # Clean and preprocess the normal data for the autoencoder
    normal_df_clean = clean_and_prepare_data(normal_df)
    X_train_normal, X_test_normal, _, _, scaler_normal, _ = encode_and_split_data(normal_df_clean)

    # Train the autoencoder on normal traffic
    input_dim = X_train_normal.shape[1]
    autoencoder = create_autoencoder(input_dim)
    autoencoder.fit(X_train_normal, X_train_normal, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test_normal, X_test_normal), verbose=1)

    del normal_df, normal_df_clean, X_train_normal, X_test_normal, scaler_normal
    gc.collect()

    # Clean and preprocess the full dataset for the classifier
    df_clean = clean_and_prepare_data(df)
    X_train, X_test, y_train, y_test, scaler, label_encoder = encode_and_split_data(df_clean)

    # Train the classifier
    n_classes = len(np.unique(y_train))
    classifier = create_classifier(n_classes=n_classes, use_gpu=False) # Fallback to CPU
    classifier.fit(X_train, y_train)

    # Save the artifacts
    artifacts_dir = 'artifacts'
    if not os.path.exists(artifacts_dir):
        os.makedirs(artifacts_dir)

    autoencoder.save(os.path.join(artifacts_dir, 'autoencoder.h5'))
    joblib.dump(classifier, os.path.join(artifacts_dir, 'classifier.joblib'))
    joblib.dump(scaler, os.path.join(artifacts_dir, 'scaler.joblib'))
    joblib.dump(label_encoder, os.path.join(artifacts_dir, 'label_encoder.joblib'))

    print("Models and artifacts saved successfully.")

if __name__ == '__main__':
    train_models()
