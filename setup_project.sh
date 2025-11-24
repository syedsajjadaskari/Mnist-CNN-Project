#!/bin/bash

# MNIST CNN Project Setup Script
# This script creates necessary files and adds code to them

set -e  # Exit on error

echo "Setting up MNIST CNN Project..."

# Create directories if they don't exist
mkdir -p src
mkdir -p models
mkdir -p logs
mkdir -p data

# Create main.py
echo "Creating main.py..."
cat > main.py << 'EOF'
"""
MNIST CNN Project - Main Entry Point
"""
import tensorflow as tf
from src.data_loader import load_data
from src.model import create_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.visualize import visualize_results

def main():
    """Main function to run the MNIST CNN training pipeline"""
    print("Starting MNIST CNN Training Pipeline...")
    
    # Load data
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Create model
    print("Creating CNN model...")
    model = create_model()
    
    # Train model
    print("Training model...")
    history = train_model(model, x_train, y_train, x_test, y_test)
    
    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, x_test, y_test)
    
    # Visualize results
    print("Visualizing results...")
    visualize_results(history, model, x_test, y_test)
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
EOF

# Create requirements.txt
echo "Creating requirements.txt..."
cat > requirements.txt << 'EOF'
tensorflow>=2.13.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
pandas>=2.0.0
pillow>=10.0.0
EOF

# Create README.md
echo "Creating README.md..."
cat > README.md << 'EOF'
# MNIST CNN Project

A Convolutional Neural Network (CNN) implementation for MNIST digit classification using TensorFlow/Keras.

## Project Structure

```
Mnist-CNN-Project/
├── src/
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── model.py          # CNN model architecture
│   ├── train.py          # Training logic
│   ├── evaluate.py       # Model evaluation
│   ├── callbacks.py      # Training callbacks
│   └── visualize.py      # Visualization utilities
├── models/               # Saved model files
├── logs/                 # Training logs
├── data/                 # Dataset storage
├── main.py              # Main entry point
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the training pipeline:
```bash
python main.py
```

## Features

- Custom CNN architecture for MNIST classification
- Data augmentation and preprocessing
- Training with callbacks (ModelCheckpoint, EarlyStopping, etc.)
- Model evaluation with metrics
- Visualization of training history and predictions

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- See requirements.txt for full list

## License

MIT License
EOF

# Create .gitignore
echo "Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# TensorFlow/Keras
*.h5
*.hdf5
*.pb
checkpoint
*.ckpt*
saved_model/

# Logs
logs/
*.log

# Data
data/
*.csv
*.npy

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
*.ipynb
EOF

# Create config.py for project configuration
echo "Creating src/config.py..."
cat > src/config.py << 'EOF'
"""
Configuration file for MNIST CNN Project
"""

# Model Configuration
MODEL_CONFIG = {
    'input_shape': (28, 28, 1),
    'num_classes': 10,
    'conv_layers': [
        {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'},
        {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
        {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu'}
    ],
    'dense_layers': [128, 64],
    'dropout_rate': 0.5
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 128,
    'epochs': 20,
    'validation_split': 0.1,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss': 'sparse_categorical_crossentropy',
    'metrics': ['accuracy']
}

# Paths
PATHS = {
    'models': 'models/',
    'logs': 'logs/',
    'data': 'data/',
    'checkpoints': 'models/checkpoints/'
}

# Callbacks Configuration
CALLBACKS_CONFIG = {
    'early_stopping': {
        'monitor': 'val_loss',
        'patience': 5,
        'restore_best_weights': True
    },
    'model_checkpoint': {
        'filepath': 'models/best_model.h5',
        'monitor': 'val_accuracy',
        'save_best_only': True
    },
    'reduce_lr': {
        'monitor': 'val_loss',
        'factor': 0.5,
        'patience': 3,
        'min_lr': 1e-7
    }
}
EOF

# Create __init__.py files
echo "Creating __init__.py files..."
touch src/__init__.py

echo ""
echo "✅ Project setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Install dependencies: pip install -r requirements.txt"
echo "2. Run the project: python main.py"
echo ""
echo "Files created:"
echo "  - main.py"
echo "  - requirements.txt"
echo "  - README.md"
echo "  - .gitignore"
echo "  - src/config.py"
echo "  - src/__init__.py"
echo ""
