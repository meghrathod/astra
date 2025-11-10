# Astra Guardian - Agent Architecture Documentation

## Overview

Astra Guardian is an AI-powered network security system that classifies encrypted network traffic using flow-based behavioral analysis. The system uses a dual-model approach to handle both known and unknown traffic patterns.

## System Architecture

### Core Components

1. **Data Preprocessing Module** (`astra_guardian/preprocessing.py`)
   - Handles data cleaning (infinite values, NaN removal)
   - Feature extraction and scaling
   - Target encoding for classification

2. **Model Definitions** (`astra_guardian/models.py`)
   - Autoencoder for anomaly detection
   - XGBoost classifier for application identification

3. **Training Pipeline** (`scripts/train.py`)
   - Orchestrates model training
   - Manages data flow between preprocessing and models
   - Saves trained artifacts

4. **Data Download** (`scripts/download_data.py`)
   - Handles Kaggle dataset download
   - Manages API credentials

## Model Architecture

### Model 1: XGBoost Classifier

**Purpose**: Multi-class classification of known applications

**Architecture**:
- Algorithm: XGBoost (Gradient Boosting)
- Objective: `multi:softmax`
- Input: 87 flow-based features
- Output: 87 application classes
- Hyperparameters:
  - `n_estimators`: 100
  - `max_depth`: 5
  - `learning_rate`: 0.1

**Training Data**:
- Full dataset with all 87 application classes
- Target variable: `L7Protocol` (encoded)
- Features: All flow-based metrics (packet counts, sizes, IAT, flags, etc.)

**Use Case**: 
- Identifies known applications (YouTube, Netflix, Zoom, etc.)
- Provides application-level visibility for QoS and parental controls

### Model 2: Autoencoder Anomaly Detector

**Purpose**: Detect anomalous/unknown traffic patterns

**Architecture**:
- Type: Deep Autoencoder (Neural Network)
- Encoder: Input → 128 → 64 → 32 (encoding dimension)
- Decoder: 32 → 64 → 128 → Input
- Activation: ReLU (hidden), Sigmoid (output)
- Loss: Mean Squared Error
- Optimizer: Adam

**Training Data**:
- Only "normal" traffic from trusted applications:
  - GOOGLE
  - YOUTUBE
  - FACEBOOK
  - MICROSOFT
  - CLOUDFLARE
  - AMAZON
- Unsupervised learning (learns normal traffic patterns)
- No labels needed

**Use Case**:
- Flags traffic that deviates from learned normal patterns
- Detects unknown applications or potential threats
- Complements the classifier by catching what it misses

## Data Flow

### Training Pipeline

```
1. Load Dataset
   └─> Dataset-Unicauca-Version2-87Atts.csv
   
2. Filter Normal Traffic (for Autoencoder)
   └─> Filter by ProtocolName: ['GOOGLE', 'YOUTUBE', ...]
   
3. Preprocess Normal Data
   └─> clean_and_prepare_data()
   └─> Extract features (drop L7Protocol)
   └─> Split train/test
   └─> StandardScaler
   
4. Train Autoencoder
   └─> Input: Normal traffic features
   └─> Output: Reconstructed features
   └─> Save: autoencoder.h5, scaler_normal.joblib
   
5. Preprocess Full Dataset (for Classifier)
   └─> clean_and_prepare_data()
   └─> encode_and_split_data()
   └─> Encode L7Protocol labels
   └─> Split train/test (stratified)
   └─> StandardScaler
   
6. Train Classifier
   └─> Input: All traffic features
   └─> Output: Application class probabilities
   └─> Save: classifier.joblib, scaler.joblib, label_encoder.joblib
```

### Inference Pipeline (Future)

```
1. Load New Flow Data
   └─> Extract flow features
   
2. Preprocess
   └─> Apply same cleaning pipeline
   └─> Scale using saved scaler
   
3. Classifier Prediction
   └─> Predict application class
   └─> Get confidence score
   
4. Anomaly Detection
   └─> Reconstruct using autoencoder
   └─> Calculate reconstruction error
   └─> Flag if error > threshold
   
5. Decision Logic
   └─> If high confidence classification → Use classifier result
   └─> If low confidence + high anomaly score → Flag as unknown/threat
```

## Feature Engineering

The system uses 87 flow-based features including:

### Packet Statistics
- Total forward/backward packets
- Packet lengths (mean, std, min, max)
- Packet counts per second

### Flow Characteristics
- Flow duration
- Bytes per second
- Inter-arrival times (IAT) - mean, std, min, max
- Forward/backward ratios

### Protocol Flags
- TCP flags (SYN, ACK, FIN, RST, PSH, URG)
- Flag counts

### Window & Segment Info
- Initial window sizes
- Average segment sizes
- Bulk transfer rates

### Timing Features
- Active/idle times
- Subflow statistics

## Key Design Decisions

1. **Two-Model Approach**: Separates known classification from anomaly detection
   - Classifier handles known patterns efficiently
   - Autoencoder catches novel patterns

2. **Normal Traffic Filtering**: Autoencoder trained only on trusted apps
   - Reduces false positives
   - Focuses on learning legitimate patterns

3. **Feature-Based Analysis**: Uses flow metadata, not packet content
   - Works with encrypted traffic
   - Privacy-preserving
   - Efficient processing

4. **Memory Optimization**: 
   - Uses appropriate data types (float32, uint8)
   - Cleans up intermediate variables
   - Processes models sequentially

## File Structure

```
astra/
├── astra_guardian/          # Core package
│   ├── __init__.py
│   ├── models.py            # Model definitions
│   └── preprocessing.py     # Data preprocessing
├── scripts/                 # Executable scripts
│   ├── download_data.py     # Dataset download
│   └── train.py             # Training pipeline
├── notebooks/               # Jupyter notebooks
│   └── exploratory_data_analysis.ipynb
├── tests/                   # Unit tests
│   ├── test_models.py
│   └── test_preprocessing.py
├── data/                    # Dataset storage
├── artifacts/               # Trained models (generated)
│   ├── autoencoder.h5
│   ├── classifier.joblib
│   ├── scaler.joblib
│   ├── scaler_normal.joblib
│   └── label_encoder.joblib
├── requirements.txt         # Dependencies
├── .cursorrules            # Cursor IDE rules
└── agents.md               # This file
```

## Usage

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset (requires Kaggle API)
python scripts/download_data.py

# Run EDA
jupyter notebook notebooks/exploratory_data_analysis.ipynb
```

### Training
```bash
# Train both models
python scripts/train.py
```

### Testing
```bash
# Run unit tests
pytest tests/
```

## Future Enhancements

1. **Inference API**: REST API for real-time traffic classification
2. **Model Evaluation**: Comprehensive metrics and visualization
3. **Hyperparameter Tuning**: Automated optimization
4. **Online Learning**: Incremental model updates
5. **Deployment**: Docker containerization
6. **Monitoring**: Model performance tracking and drift detection

## References

- Dataset: [IP Network Traffic Flows Labeled with 87 Apps](https://www.kaggle.com/datasets/jsrojas/ip-network-traffic-flows-labeled-with-87-apps)
- Flow-based Network Traffic Analysis
- Encrypted Traffic Classification
- Anomaly Detection in Network Traffic

