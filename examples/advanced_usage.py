"""
Advanced usage example for Japan Electricity Price Prediction

This script demonstrates advanced features including:
- Multi-region comparison
- Hyperparameter experimentation
- Custom model architectures
- Advanced visualization

Usage:
    python advanced_usage.py

Author: JIEKAI WU
Date: August 2025
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lstm_predictor import ElectricityPricePredictor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config
from utils import print_metrics


def compare_regions(df, regions_to_compare=['Tokyo_ibprice', 'Hokkaido_ibprice', 'Kansai_ibprice']):
    """
    Compare LSTM performance across different regions
    
    Args:
        df (pd.DataFrame): Input data
        regions_to_compare (list): List of regions to compare
        
    Returns:
        dict: Results for each region
    """
    results = {}
    
    print("Comparing LSTM performance across regions...")
    print("=" * 60)
    
    for region in regions_to_compare:
        print(f"\nTraining model for {region}...")
        
        # Initialize predictor
        predictor = ElectricityPricePredictor(
            sequence_length=config.SEQUENCE_LENGTH,
            test_size=config.TEST_SIZE,
            batch_size=config.BATCH_SIZE
        )
        
        # Prepare data
        data = df[['date_time', region]].copy()
        train_loader, test_loader = predictor.prepare_data(data, region)
        
        # Build and train model
        predictor.build_model(
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            learning_rate=config.LEARNING_RATE
        )
        
        train_losses, test_losses = predictor.train_model(
            train_loader, test_loader, epochs=config.EPOCHS
        )
        
        # Make predictions and evaluate
        predictions, actuals = predictor.make_predictions(test_loader)
        metrics = predictor.evaluate_model(predictions, actuals)
        
        results[region] = {
            'metrics': metrics,
            'predictions': predictions,
            'actuals': actuals,
            'train_losses': train_losses,
            'test_losses': test_losses
        }
        
        print(f"✓ {region} completed - RMSE: {metrics['RMSE']:.4f}")
    
    return results


def hyperparameter_tuning(df, target_region='Tokyo_ibprice'):
    """
    Perform hyperparameter tuning
    
    Args:
        df (pd.DataFrame): Input data
        target_region (str): Target region for tuning
        
    Returns:
        dict: Results of hyperparameter experiments
    """
    print(f"\nHyperparameter tuning for {target_region}...")
    print("=" * 60)
    
    # Define hyperparameter combinations to test
    param_combinations = [
        {'hidden_size': 32, 'num_layers': 1, 'learning_rate': 0.001},
        {'hidden_size': 64, 'num_layers': 2, 'learning_rate': 0.001},
        {'hidden_size': 128, 'num_layers': 2, 'learning_rate': 0.0005},
        {'hidden_size': 64, 'num_layers': 3, 'learning_rate': 0.001},
    ]
    
    results = {}
    data = df[['date_time', target_region]].copy()
    
    for i, params in enumerate(param_combinations):
        print(f"\nExperiment {i+1}: {params}")
        
        predictor = ElectricityPricePredictor(
            sequence_length=config.SEQUENCE_LENGTH,
            test_size=config.TEST_SIZE,
            batch_size=config.BATCH_SIZE
        )
        
        train_loader, test_loader = predictor.prepare_data(data, target_region)
        
        predictor.build_model(
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=config.DROPOUT,
            learning_rate=params['learning_rate']
        )
        
        train_losses, test_losses = predictor.train_model(
            train_loader, test_loader, epochs=30  # Reduced epochs for tuning
        )
        
        predictions, actuals = predictor.make_predictions(test_loader)
        metrics = predictor.evaluate_model(predictions, actuals)
        
        results[f"experiment_{i+1}"] = {
            'params': params,
            'metrics': metrics,
            'final_train_loss': train_losses[-1],
            'final_test_loss': test_losses[-1]
        }
        
        print(f"  RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}")
    
    return results


def create_advanced_visualizations(comparison_results):
    """
    Create advanced comparison visualizations
    
    Args:
        comparison_results (dict): Results from region comparison
    """
    print("\nCreating advanced visualizations...")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multi-Region LSTM Performance Comparison', fontsize=16)
    
    regions = list(comparison_results.keys())
    colors = ['blue', 'red', 'green']
    
    # 1. RMSE Comparison
    rmse_values = [comparison_results[region]['metrics']['RMSE'] for region in regions]
    axes[0, 0].bar(range(len(regions)), rmse_values, color=colors)
    axes[0, 0].set_title('RMSE Comparison Across Regions')
    axes[0, 0].set_xlabel('Region')
    axes[0, 0].set_ylabel('RMSE (JPY/kWh)')
    axes[0, 0].set_xticks(range(len(regions)))
    axes[0, 0].set_xticklabels([r.replace('_ibprice', '') for r in regions], rotation=45)
    
    # 2. Training Loss Curves
    for i, region in enumerate(regions):
        train_losses = comparison_results[region]['train_losses']
        test_losses = comparison_results[region]['test_losses']
        epochs = range(1, len(train_losses) + 1)
        
        axes[0, 1].plot(epochs, train_losses, '--', color=colors[i], alpha=0.7, 
                       label=f'{region.replace("_ibprice", "")} Train')
        axes[0, 1].plot(epochs, test_losses, '-', color=colors[i], 
                       label=f'{region.replace("_ibprice", "")} Test')
    
    axes[0, 1].set_title('Training/Validation Loss Curves')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. Prediction Accuracy Scatter
    for i, region in enumerate(regions):
        predictions = comparison_results[region]['predictions']
        actuals = comparison_results[region]['actuals']
        
        # Sample points for readability
        sample_size = min(500, len(predictions))
        idx = np.random.choice(len(predictions), sample_size, replace=False)
        
        axes[1, 0].scatter(actuals[idx], predictions[idx], alpha=0.6, 
                          color=colors[i], label=region.replace('_ibprice', ''))
    
    # Perfect prediction line
    all_actuals = np.concatenate([comparison_results[region]['actuals'] for region in regions])
    min_val, max_val = all_actuals.min(), all_actuals.max()
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
    axes[1, 0].set_title('Prediction vs Actual (Sampled)')
    axes[1, 0].set_xlabel('Actual Price (JPY/kWh)')
    axes[1, 0].set_ylabel('Predicted Price (JPY/kWh)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. Error Distribution
    for i, region in enumerate(regions):
        predictions = comparison_results[region]['predictions']
        actuals = comparison_results[region]['actuals']
        errors = actuals - predictions
        
        axes[1, 1].hist(errors, bins=30, alpha=0.7, color=colors[i], 
                       label=region.replace('_ibprice', ''))
    
    axes[1, 1].set_title('Error Distribution Comparison')
    axes[1, 1].set_xlabel('Prediction Error (JPY/kWh)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].axvline(x=0, color='black', linestyle='--', alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('advanced_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Advanced visualizations saved as 'advanced_comparison_results.png'")


def main():
    """Main function for advanced usage demonstration"""
    print("Japan Electricity Price Prediction - Advanced Usage")
    print("=" * 70)
    
    # Load data
    try:
        df = pd.read_csv(config.DATA_FILE)
        df['date_time'] = pd.to_datetime(df['date_time'])
        print(f"✓ Data loaded successfully: {df.shape}")
    except FileNotFoundError:
        print(f"✗ Data file '{config.DATA_FILE}' not found.")
        return
    
    # 1. Multi-region comparison
    comparison_results = compare_regions(df)
    
    # Print comparison summary
    print("\n" + "=" * 60)
    print("MULTI-REGION COMPARISON SUMMARY")
    print("=" * 60)
    for region, results in comparison_results.items():
        region_name = region.replace('_ibprice', '')
        metrics = results['metrics']
        print(f"\n{region_name}:")
        print(f"  RMSE: {metrics['RMSE']:8.4f}")
        print(f"  MAE:  {metrics['MAE']:8.4f}")
        print(f"  MAPE: {metrics['MAPE']:8.2f}%")
    
    # 2. Hyperparameter tuning
    tuning_results = hyperparameter_tuning(df)
    
    # Print tuning summary
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("=" * 60)
    best_experiment = min(tuning_results.items(), key=lambda x: x[1]['metrics']['RMSE'])
    
    for exp_name, results in tuning_results.items():
        params = results['params']
        metrics = results['metrics']
        print(f"\n{exp_name}:")
        print(f"  Params: {params}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAE:  {metrics['MAE']:.4f}")
    
    print(f"\n🏆 Best configuration: {best_experiment[0]}")
    print(f"   Parameters: {best_experiment[1]['params']}")
    print(f"   RMSE: {best_experiment[1]['metrics']['RMSE']:.4f}")
    
    # 3. Advanced visualizations
    create_advanced_visualizations(comparison_results)
    
    print("\n🎉 Advanced usage example completed successfully!")
    print("📊 Check 'advanced_comparison_results.png' for detailed comparisons")


if __name__ == "__main__":
    main()
