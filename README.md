# Astra Guardian - AI-Powered Network Security

Astra Guardian is an AI-powered network security application that uses flow-based behavioral analysis to classify encrypted network traffic. The system employs a dual-model architecture to identify known applications and detect anomalous traffic patterns.

## Overview

Traditional firewalls rely on Deep Packet Inspection (DPI) to identify applications, but modern encrypted traffic (HTTPS, streaming, gaming) makes this impossible. Astra Guardian solves this by analyzing the **behavioral "footprint"** of network flows rather than packet content.

### Key Features

- **Flow-Based Analysis**: Analyzes metadata (packet counts, sizes, timing) instead of content
- **Dual-Model Architecture**:
  - **XGBoost Classifier**: Identifies 87 known applications
  - **Autoencoder Anomaly Detector**: Flags unknown/threat traffic
- **Privacy-Preserving**: Works with encrypted traffic without decryption
- **Memory Efficient**: Optimized data types and processing

## Architecture

### Model 1: XGBoost Classifier
- **Purpose**: Multi-class classification of known applications
- **Classes**: 87 application protocols
- **Features**: 87 flow-based metrics (packet statistics, timing, flags, etc.)
- **Use Case**: Application identification for QoS and parental controls

### Model 2: Autoencoder Anomaly Detector
- **Purpose**: Detect anomalous/unknown traffic patterns
- **Architecture**: Deep autoencoder (128 → 64 → 32 → 64 → 128)
- **Training**: Only on "normal" traffic (Google, YouTube, Facebook, Microsoft, Cloudflare, Amazon)
- **Use Case**: Threat detection and unknown application identification

## Installation

### Prerequisites
- Python 3.8+
- Kaggle API credentials (for dataset download)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd astra
```

2. Create and activate virtual environment:
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure Kaggle API (optional):
   - `kagglehub` will automatically use credentials from `~/.kaggle/kaggle.json` if available
   - Or it will prompt for authentication when needed
   - To set up manually, place `kaggle.json` in `~/.kaggle/` with your credentials

5. Download the dataset:
```bash
python scripts/download_data.py
```

## Usage

**Note**: Always ensure the `.venv` virtual environment is activated before running any commands.

### Exploratory Data Analysis

```bash
jupyter notebook notebooks/exploratory_data_analysis.ipynb
```

### Training Models

Train both the classifier and anomaly detector:

**Option 1: Using the run script (Recommended)**
```bash
./run_training.sh
```

**Option 2: Direct Python execution**
```bash
# Make sure you're in the project root and .venv is activated
source .venv/bin/activate
python scripts/train.py
```

**Option 3: Install package in development mode**
```bash
source .venv/bin/activate
pip install -e .
python scripts/train.py
```

The training script will:
- Load and preprocess the dataset
- Train the autoencoder on normal traffic (with detailed epoch-by-epoch progress)
- Train the XGBoost classifier on all traffic
- Save models and artifacts to `artifacts/`
- Generate detailed logs in `logs/` directory

**M2 MacBook Air Optimizations:**
- Reduced batch sizes for memory efficiency
- Detailed epoch progress logging with memory usage
- Optimized XGBoost parameters for lower memory usage
- Automatic memory cleanup between training phases

### Running Tests

```bash
pytest tests/
```

## Project Structure

```
astra/
├── astra_guardian/          # Core package
│   ├── models.py            # Model definitions
│   └── preprocessing.py     # Data preprocessing
├── scripts/                 # Executable scripts
│   ├── download_data.py     # Dataset download
│   └── train.py             # Training pipeline
├── notebooks/               # Jupyter notebooks
│   └── exploratory_data_analysis.ipynb
├── tests/                   # Unit tests
├── data/                    # Dataset storage
├── artifacts/               # Trained models (generated)
├── requirements.txt         # Dependencies
├── .cursorrules            # Cursor IDE rules
└── agents.md               # Architecture documentation
```

## Dataset

The project uses the [IP Network Traffic Flows Labeled with 87 Apps](https://www.kaggle.com/datasets/jsrojas/ip-network-traffic-flows-labeled-with-87-apps) dataset from Kaggle.

**Features**: 87 flow-based attributes including:
- Packet statistics (counts, sizes, rates)
- Flow characteristics (duration, bytes per second)
- Inter-arrival times (IAT)
- TCP flags
- Window and segment information
- Timing features

**Target**: `L7Protocol` - Application protocol classification (87 classes)

## Model Artifacts

After training, the following artifacts are saved to `artifacts/`:
- `autoencoder.h5` - Trained autoencoder model
- `classifier.joblib` - Trained XGBoost classifier
- `scaler.joblib` - Feature scaler for classifier
- `scaler_normal.joblib` - Feature scaler for autoencoder
- `label_encoder.joblib` - Label encoder for target classes

## Development

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add docstrings to all functions
- See `.cursorrules` for detailed guidelines

### Testing
- Write tests for all preprocessing functions
- Write tests for model creation
- Run `pytest tests/` before committing

## References

- **Dataset**: [IP Network Traffic Flows Labeled with 87 Apps](https://www.kaggle.com/datasets/jsrojas/ip-network-traffic-flows-labeled-with-87-apps)
- Flow-based Network Traffic Analysis
- Encrypted Traffic Classification
- Anomaly Detection in Network Traffic

## License

[Add your license here]

## Contributors

[Add contributors here]
