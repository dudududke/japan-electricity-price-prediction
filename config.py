"""
Configuration file for Japan Electricity Price Prediction

This file contains all the configurable parameters for the LSTM model
and training process.

Author: JIEKAI WU
Date: August 2025
"""

# Model Architecture Parameters
SEQUENCE_LENGTH = 48        # Input sequence length (number of time steps)
HIDDEN_SIZE = 64           # Number of hidden units in LSTM layers
NUM_LAYERS = 2             # Number of LSTM layers
DROPOUT = 0.2              # Dropout rate for regularization
INPUT_SIZE = 1             # Number of input features
OUTPUT_SIZE = 1            # Number of output features

# Training Parameters
BATCH_SIZE = 64            # Batch size for training
EPOCHS = 50                # Number of training epochs
LEARNING_RATE = 0.001      # Learning rate for Adam optimizer
TEST_SIZE = 0.2            # Proportion of data for testing

# Data Parameters
TARGET_COLUMN = 'Tokyo_ibprice'  # Default target column for prediction
DATA_FILE = 'Japan_ImbalancePrice.csv'  # Default data file name

# Output Parameters
MODEL_SAVE_PATH = 'lstm_electricity_model.pth'  # Path to save trained model
PLOT_SAVE_PATH = 'prediction_results.png'       # Path to save result plots
ANALYSIS_PLOT_PATH = 'data_analysis_overview.png'  # Path to save analysis plots

# Device Configuration
USE_CUDA = True            # Whether to use CUDA if available

# Visualization Parameters
FIGURE_SIZE = (16, 12)     # Default figure size for plots
DPI = 300                  # DPI for saved plots
PLOT_STYLE = 'default'     # Matplotlib style

# Regional Information
REGIONS = [
    'Hokkaido_ibprice',
    'Tohoku_ibprice', 
    'Tokyo_ibprice',
    'Chubu_ibprice',
    'Hokuriku_ibprice',
    'Kansai_ibprice',
    'Chugoku_ibprice',
    'Shikoku_ibprice',
    'Kyushu_ibprice'
]

REGION_NAMES = [
    'Hokkaido',
    'Tohoku',
    'Tokyo', 
    'Chubu',
    'Hokuriku',
    'Kansai',
    'Chugoku',
    'Shikoku',
    'Kyushu'
]
