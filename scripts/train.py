import os
import sys

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import joblib
import numpy as np
import gc
import logging
import psutil
from datetime import datetime
from tensorflow.keras.callbacks import Callback
from astra_guardian.preprocessing import clean_and_prepare_data, encode_and_split_data
from astra_guardian.models import create_autoencoder, create_classifier

# Custom callback for detailed epoch logging
class EpochProgressLogger(Callback):
    """Callback to log detailed progress for each epoch."""
    def __init__(self, logger, total_epochs=50):
        super().__init__()
        self.logger = logger
        self.total_epochs = total_epochs
        self.epoch_times = []
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = datetime.now()
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / 1024**2
        self.logger.info(f"Epoch {epoch + 1}/{self.total_epochs} starting... (Memory: {mem_mb:.1f} MB)")
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = (datetime.now() - self.epoch_start).total_seconds()
        self.epoch_times.append(epoch_time)
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / 1024**2
        
        train_loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        avg_time = np.mean(self.epoch_times)
        remaining_epochs = self.total_epochs - (epoch + 1)
        eta_seconds = avg_time * remaining_epochs
        
        self.logger.info(
            f"Epoch {epoch + 1}/{self.total_epochs} completed | "
            f"Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Time: {epoch_time:.2f}s | Memory: {mem_mb:.1f} MB | "
            f"ETA: {eta_seconds/60:.1f} min"
        )

# Setup logging
def setup_logging():
    """Configure logging with both file and console handlers."""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Setup root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.info(f"System: M2 MacBook Air optimization mode")
    return logger

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

def train_models():
    """
    Loads the data, preprocesses it, trains the classifier and anomaly detector,
    and saves the artifacts.
    """
    logger = setup_logging()
    logger.info(f"Initial memory usage: {get_memory_usage():.1f} MB")
    training_start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("Starting Astra Guardian Model Training")
    logger.info("=" * 80)
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
    logger.info(f"Loading dataset from: {data_path}")
    logger.info(f"Memory before loading: {get_memory_usage():.1f} MB")
    start_time = datetime.now()
    
    # Load in chunks to reduce memory peak usage
    logger.info("Loading dataset with optimized memory settings...")
    df = pd.read_csv(data_path, dtype=dtype_mapping, low_memory=False)
    
    load_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Dataset loaded successfully in {load_time:.2f} seconds")
    logger.info(f"Dataset shape: {df.shape} (rows, columns)")
    logger.info(f"Memory after loading: {get_memory_usage():.1f} MB")
    logger.info(f"DataFrame memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Define "normal" traffic - expanded list based on EDA analysis
    # These are legitimate/trusted applications for anomaly detector training
    normal_apps = [
        'GOOGLE', 'YOUTUBE', 'FACEBOOK', 'MICROSOFT', 'CLOUDFLARE', 'AMAZON',
        'GMAIL', 'YAHOO', 'MSN', 'DROPBOX', 'OFFICE_365', 'SKYPE',
        'TWITTER', 'WINDOWS_UPDATE', 'APPLE', 'CONTENT_FLASH',
        'HTTP', 'SSL'
    ]
    logger.info(f"Filtering normal traffic from {len(normal_apps)} applications")
    logger.info(f"Normal apps: {', '.join(normal_apps)}")
    
    if 'ProtocolName' not in df.columns:
        # Fallback: use L7Protocol if ProtocolName doesn't exist
        logger.warning("ProtocolName not found, using L7Protocol for filtering")
        normal_df = df[df['L7Protocol'].isin(normal_apps)].copy()
    else:
        normal_df = df[df['ProtocolName'].isin(normal_apps)].copy()
    
    logger.info(f"Normal traffic samples: {len(normal_df):,} ({len(normal_df)/len(df)*100:.2f}% of total)")

    # Clean and preprocess the normal data for the autoencoder
    logger.info("Cleaning and preprocessing normal traffic data...")
    start_time = datetime.now()
    normal_df_clean = clean_and_prepare_data(normal_df)
    clean_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Data cleaning completed in {clean_time:.2f} seconds")
    logger.info(f"Cleaned normal data shape: {normal_df_clean.shape}")
    
    # For autoencoder, we need features only (no target encoding needed)
    # Extract features and target separately
    if 'L7Protocol' not in normal_df_clean.columns:
        logger.error("L7Protocol column not found after cleaning")
        raise ValueError("L7Protocol column not found after cleaning")
    
    X_normal = normal_df_clean.drop(columns=['L7Protocol'])
    logger.info(f"Features extracted: {X_normal.shape[1]} features")
    
    # Split normal data
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    logger.info("Splitting normal data into train/test sets (80/20)...")
    X_train_normal, X_test_normal = train_test_split(
        X_normal, test_size=0.2, random_state=42
    )
    
    # Scale the normal data
    logger.info("Scaling normal traffic features...")
    scaler_normal = StandardScaler()
    X_train_normal = scaler_normal.fit_transform(X_train_normal)
    X_test_normal = scaler_normal.transform(X_test_normal)
    logger.info("Feature scaling completed")

    # Train the autoencoder on normal traffic
    logger.info("=" * 80)
    logger.info("TRAINING AUTOENCODER (Anomaly Detector)")
    logger.info("=" * 80)
    input_dim = X_train_normal.shape[1]
    
    # M2 MacBook Air optimized settings - smaller batch size for memory efficiency
    batch_size = 128  # Reduced from 256 for M2 MacBook Air
    epochs = 50
    
    logger.info(f"Autoencoder architecture:")
    logger.info(f"  Input dimension: {input_dim}")
    logger.info(f"  Training samples: {X_train_normal.shape[0]:,}")
    logger.info(f"  Validation samples: {X_test_normal.shape[0]:,}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size} (optimized for M2 MacBook Air)")
    logger.info(f"  Memory before training: {get_memory_usage():.1f} MB")
    
    start_time = datetime.now()
    autoencoder = create_autoencoder(input_dim)
    logger.info("Autoencoder model created")
    
    # Create callback for detailed epoch logging
    epoch_logger = EpochProgressLogger(logger, total_epochs=epochs)
    
    logger.info("Starting autoencoder training...")
    logger.info("-" * 80)
    history = autoencoder.fit(
        X_train_normal, X_train_normal, 
        epochs=epochs, 
        batch_size=batch_size, 
        shuffle=True, 
        validation_data=(X_test_normal, X_test_normal), 
        verbose=0,  # Set to 0 since we're using custom callback
        callbacks=[epoch_logger]
    )
    logger.info("-" * 80)
    
    train_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Autoencoder training completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
    logger.info(f"Memory after training: {get_memory_usage():.1f} MB")
    
    # Log final training metrics
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    best_val_loss = min(history.history['val_loss'])
    best_epoch = history.history['val_loss'].index(best_val_loss) + 1
    
    logger.info("=" * 80)
    logger.info("AUTOENCODER TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Final training loss: {final_train_loss:.6f}")
    logger.info(f"Final validation loss: {final_val_loss:.6f}")
    logger.info(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
    logger.info(f"Average epoch time: {np.mean(epoch_logger.epoch_times):.2f} seconds")

    logger.info("Cleaning up normal traffic data from memory...")
    logger.info(f"Memory before cleanup: {get_memory_usage():.1f} MB")
    del normal_df, normal_df_clean, X_train_normal, X_test_normal, scaler_normal
    gc.collect()
    logger.info(f"Memory after cleanup: {get_memory_usage():.1f} MB")
    logger.info("Memory cleanup completed")

    # Clean and preprocess the full dataset for the classifier
    logger.info("=" * 80)
    logger.info("PREPARING CLASSIFIER TRAINING DATA")
    logger.info("=" * 80)
    logger.info("Cleaning and preprocessing full dataset...")
    start_time = datetime.now()
    df_clean = clean_and_prepare_data(df)
    clean_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Full dataset cleaning completed in {clean_time:.2f} seconds")
    logger.info(f"Cleaned dataset shape: {df_clean.shape}")
    
    logger.info("Encoding labels and splitting data...")
    start_time = datetime.now()
    X_train, X_test, y_train, y_test, scaler, label_encoder = encode_and_split_data(df_clean)
    encode_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Data encoding and splitting completed in {encode_time:.2f} seconds")
    
    # Clean up original dataframe now that we have processed data
    logger.info("Cleaning up original dataset from memory...")
    logger.info(f"Memory before cleanup: {get_memory_usage():.1f} MB")
    del df, df_clean
    gc.collect()
    logger.info(f"Memory after cleanup: {get_memory_usage():.1f} MB")
    
    logger.info(f"Classifier training samples: {X_train.shape[0]:,}")
    logger.info(f"Classifier test samples: {X_test.shape[0]:,}")
    logger.info(f"Number of features: {X_train.shape[1]}")
    logger.info(f"Number of classes: {len(np.unique(y_train))}")
    
    # Log class distribution
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    logger.info("Class distribution (top 10):")
    for i, (cls, count) in enumerate(zip(unique_classes[:10], class_counts[:10])):
        logger.info(f"  Class {cls}: {count:,} samples ({count/len(y_train)*100:.2f}%)")

    # Train the classifier
    logger.info("=" * 80)
    logger.info("TRAINING XGBOOST CLASSIFIER")
    logger.info("=" * 80)
    n_classes = len(np.unique(y_train))
    
    # M2 MacBook Air optimized settings - reduce memory usage
    # Use fewer estimators and adjust other parameters for memory efficiency
    n_estimators = 50  # Reduced from 100 for faster training and less memory
    max_depth = 4  # Reduced from 5 for less memory usage
    
    logger.info(f"Classifier configuration (M2 MacBook Air optimized):")
    logger.info(f"  Number of classes: {n_classes}")
    logger.info(f"  Training samples: {X_train.shape[0]:,}")
    logger.info(f"  Test samples: {X_test.shape[0]:,}")
    logger.info(f"  Features: {X_train.shape[1]}")
    logger.info(f"  N estimators: {n_estimators} (reduced for memory efficiency)")
    logger.info(f"  Max depth: {max_depth} (reduced for memory efficiency)")
    logger.info(f"  Device: CPU")
    logger.info(f"  Memory before training: {get_memory_usage():.1f} MB")
    
    start_time = datetime.now()
    # Create classifier with optimized parameters
    classifier = create_classifier(
        n_classes=n_classes, 
        n_estimators=n_estimators,
        max_depth=max_depth,
        use_gpu=False
    )
    logger.info("XGBoost classifier created")
    
    logger.info("Starting classifier training...")
    logger.info("(XGBoost will show progress automatically)")
    
    # Fit with verbose output for progress
    classifier.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=True
    )
    
    train_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Classifier training completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
    logger.info(f"Memory after training: {get_memory_usage():.1f} MB")
    
    # Evaluate classifier
    logger.info("Evaluating classifier performance...")
    eval_start = datetime.now()
    train_score = classifier.score(X_train, y_train)
    test_score = classifier.score(X_test, y_test)
    eval_time = (datetime.now() - eval_start).total_seconds()
    
    logger.info("=" * 80)
    logger.info("CLASSIFIER PERFORMANCE METRICS")
    logger.info("=" * 80)
    logger.info(f"Training accuracy: {train_score:.4f} ({train_score*100:.2f}%)")
    logger.info(f"Test accuracy: {test_score:.4f} ({test_score*100:.2f}%)")
    logger.info(f"Evaluation time: {eval_time:.2f} seconds")
    
    # Calculate overfitting metric
    overfitting = train_score - test_score
    if overfitting > 0.05:
        logger.warning(f"Potential overfitting detected: {overfitting:.4f} difference between train and test accuracy")
    else:
        logger.info(f"Model generalization: Good (overfitting difference: {overfitting:.4f})")

    # Save the artifacts
    logger.info("=" * 80)
    logger.info("SAVING MODELS AND ARTIFACTS")
    logger.info("=" * 80)
    artifacts_dir = 'artifacts'
    if not os.path.exists(artifacts_dir):
        os.makedirs(artifacts_dir)
        logger.info(f"Created artifacts directory: {artifacts_dir}")

    save_start = datetime.now()
    logger.info("Saving autoencoder model...")
    autoencoder_path = os.path.join(artifacts_dir, 'autoencoder.h5')
    autoencoder.save(autoencoder_path)
    logger.info(f"  ✓ Saved: {autoencoder_path}")
    
    logger.info("Saving classifier model...")
    classifier_path = os.path.join(artifacts_dir, 'classifier.joblib')
    joblib.dump(classifier, classifier_path)
    logger.info(f"  ✓ Saved: {classifier_path}")
    
    logger.info("Saving scalers...")
    scaler_path = os.path.join(artifacts_dir, 'scaler.joblib')
    scaler_normal_path = os.path.join(artifacts_dir, 'scaler_normal.joblib')
    joblib.dump(scaler, scaler_path)
    joblib.dump(scaler_normal, scaler_normal_path)
    logger.info(f"  ✓ Saved: {scaler_path}")
    logger.info(f"  ✓ Saved: {scaler_normal_path}")
    
    logger.info("Saving label encoder...")
    encoder_path = os.path.join(artifacts_dir, 'label_encoder.joblib')
    joblib.dump(label_encoder, encoder_path)
    logger.info(f"  ✓ Saved: {encoder_path}")
    
    save_time = (datetime.now() - save_start).total_seconds()
    logger.info(f"All artifacts saved successfully in {save_time:.2f} seconds")
    
    # Log file sizes
    logger.info("Artifact file sizes:")
    for filename in ['autoencoder.h5', 'classifier.joblib', 'scaler.joblib', 
                     'scaler_normal.joblib', 'label_encoder.joblib']:
        filepath = os.path.join(artifacts_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / 1024**2
            logger.info(f"  {filename}: {size_mb:.2f} MB")
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    total_time = (datetime.now() - training_start_time).total_seconds()
    logger.info(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info(f"All models and artifacts saved to '{artifacts_dir}/'")
    logger.info("Training pipeline completed!")

if __name__ == '__main__':
    train_models()
