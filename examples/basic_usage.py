"""
Basic usage example for Japan Electricity Price Prediction

This script demonstrates the basic workflow of loading data, 
training an LSTM model, and making predictions.

Usage:
    python basic_usage.py

Author: JIEKAI WU
Date: August 2025
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lstm_predictor import ElectricityPricePredictor
import pandas as pd
import config


def main():
    """Main function demonstrating basic usage"""
    print("Japan Electricity Price Prediction - Basic Usage")
    print("=" * 60)
    
    # Load data
    try:
        df = pd.read_csv(config.DATA_FILE)
        df['date_time'] = pd.to_datetime(df['date_time'])
        print(f"✓ Data loaded successfully: {df.shape}")
    except FileNotFoundError:
        print(f"✗ Data file '{config.DATA_FILE}' not found.")
        print("Please ensure the data file is in the project root directory.")
        return
    
    # Initialize predictor with default settings
    predictor = ElectricityPricePredictor(
        sequence_length=config.SEQUENCE_LENGTH,
        test_size=config.TEST_SIZE,
        batch_size=config.BATCH_SIZE
    )
    print("✓ Predictor initialized")
    
    # Prepare data for Tokyo region
    target_region = config.TARGET_COLUMN
    data = df[['date_time', target_region]].copy()
    train_loader, test_loader = predictor.prepare_data(data, target_region)
    print(f"✓ Data prepared for {target_region}")
    
    # Build model
    predictor.build_model(
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        learning_rate=config.LEARNING_RATE
    )
    print("✓ Model built")
    
    # Train model
    print("\nTraining model...")
    train_losses, test_losses = predictor.train_model(
        train_loader, test_loader, epochs=config.EPOCHS
    )
    print("✓ Model training completed")
    
    # Make predictions
    predictions, actuals = predictor.make_predictions(test_loader)
    print("✓ Predictions generated")
    
    # Evaluate model
    metrics = predictor.evaluate_model(predictions, actuals)
    print("\n" + "=" * 40)
    print("MODEL EVALUATION RESULTS")
    print("=" * 40)
    for metric, value in metrics.items():
        if metric == 'MAPE':
            print(f"{metric:>8}: {value:8.2f}%")
        else:
            print(f"{metric:>8}: {value:8.4f}")
    
    # Plot results
    print("\nGenerating visualizations...")
    predictor.plot_results(
        predictions, actuals, train_losses, test_losses,
        save_path=config.PLOT_SAVE_PATH
    )
    print("✓ Visualizations saved")
    
    # Save model
    predictor.save_model(config.MODEL_SAVE_PATH)
    print("✓ Model saved")
    
    print("\n🎉 Basic usage example completed successfully!")
    print(f"📊 Check '{config.PLOT_SAVE_PATH}' for visualizations")
    print(f"💾 Model saved as '{config.MODEL_SAVE_PATH}'")


if __name__ == "__main__":
    main()
