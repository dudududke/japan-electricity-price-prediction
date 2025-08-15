"""
Utility functions for Japan Electricity Price Prediction

This module contains helper functions used throughout the project
for data processing, visualization, and model utilities.

Author: JIEKAI WU
Date: August 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict, Any
import config


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error safely (avoiding division by zero)
    
    Args:
        actual (np.ndarray): Actual values
        predicted (np.ndarray): Predicted values
        
    Returns:
        float: MAPE value as percentage
    """
    mask = actual != 0
    if np.sum(mask) == 0:
        return np.inf
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def create_sequences(data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction
    
    Args:
        data (np.ndarray): Time series data
        sequence_length (int): Length of input sequences
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Input sequences and target values
    """
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def train_test_split_timeseries(X: np.ndarray, y: np.ndarray, 
                               test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split time series data into training and testing sets
    
    Args:
        X (np.ndarray): Input sequences
        y (np.ndarray): Target values
        test_size (float): Proportion of data for testing
        
    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test


def normalize_data(data: np.ndarray, scaler: MinMaxScaler = None) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Normalize data using MinMaxScaler
    
    Args:
        data (np.ndarray): Data to normalize
        scaler (MinMaxScaler, optional): Pre-fitted scaler
        
    Returns:
        Tuple[np.ndarray, MinMaxScaler]: Normalized data and fitted scaler
    """
    if scaler is None:
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data.reshape(-1, 1))
    else:
        normalized_data = scaler.transform(data.reshape(-1, 1))
    
    return normalized_data.flatten(), scaler


def denormalize_data(data: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """
    Denormalize data using fitted scaler
    
    Args:
        data (np.ndarray): Normalized data
        scaler (MinMaxScaler): Fitted scaler
        
    Returns:
        np.ndarray: Denormalized data
    """
    return scaler.inverse_transform(data.reshape(-1, 1)).flatten()


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Calculate various evaluation metrics
    
    Args:
        actual (np.ndarray): Actual values
        predicted (np.ndarray): Predicted values
        
    Returns:
        Dict[str, float]: Dictionary containing various metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = calculate_mape(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }


def print_metrics(metrics: Dict[str, float]) -> None:
    """
    Print metrics in a formatted way
    
    Args:
        metrics (Dict[str, float]): Dictionary of metrics
    """
    print("=" * 40)
    print("MODEL EVALUATION METRICS")
    print("=" * 40)
    for metric, value in metrics.items():
        if metric == 'MAPE':
            print(f"{metric:10}: {value:.2f}%")
        else:
            print(f"{metric:10}: {value:.4f}")
    print("=" * 40)


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility
    
    Args:
        seed (int): Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Get the appropriate device (CUDA or CPU)
    
    Returns:
        torch.device: Available device
    """
    if config.USE_CUDA and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def save_results(predictions: np.ndarray, actuals: np.ndarray, 
                metrics: Dict[str, float], filepath: str) -> None:
    """
    Save prediction results to CSV file
    
    Args:
        predictions (np.ndarray): Predicted values
        actuals (np.ndarray): Actual values
        metrics (Dict[str, float]): Evaluation metrics
        filepath (str): Path to save results
    """
    results_df = pd.DataFrame({
        'Actual': actuals,
        'Predicted': predictions,
        'Error': actuals - predictions,
        'Absolute_Error': np.abs(actuals - predictions),
        'Percentage_Error': np.abs((actuals - predictions) / actuals) * 100
    })
    
    # Add metrics as metadata in the first few rows
    metrics_df = pd.DataFrame([metrics])
    
    with open(filepath, 'w') as f:
        f.write("# Model Evaluation Metrics\n")
        metrics_df.to_csv(f, index=False)
        f.write("\n# Prediction Results\n")
        results_df.to_csv(f, index=False)
    
    print(f"Results saved to {filepath}")


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess the electricity price data
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded and preprocessed data
    """
    try:
        df = pd.read_csv(filepath)
        df['date_time'] = pd.to_datetime(df['date_time'])
        print(f"Data loaded successfully: {df.shape}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file '{filepath}' not found.")
    except Exception as e:
        raise Exception(f"Error loading data: {e}")


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate the loaded data
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    # Check required columns
    required_columns = ['date_time'] + config.REGIONS
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print("Data contains missing values")
        return False
    
    # Check data types
    if not pd.api.types.is_datetime64_any_dtype(df['date_time']):
        print("date_time column is not datetime type")
        return False
    
    return True


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional time-based features
    
    Args:
        df (pd.DataFrame): Input dataframe with date_time column
        
    Returns:
        pd.DataFrame: DataFrame with additional time features
    """
    df = df.copy()
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    df['day_of_year'] = df['date_time'].dt.dayofyear
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    return df
