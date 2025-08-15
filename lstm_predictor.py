"""
Japan Electricity Imbalance Price Prediction using LSTM

This module implements an LSTM-based time series prediction model
for Japanese electricity imbalance prices.

Author: JIEKAI WU
Date: August 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesDataset(Dataset):
    """Custom PyTorch Dataset for time series data"""
    
    def __init__(self, X, y):
        """
        Initialize the dataset
        
        Args:
            X (numpy.ndarray): Input sequences
            y (numpy.ndarray): Target values
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """LSTM Neural Network Model for Time Series Prediction"""
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, 
                 output_size=1, dropout=0.2):
        """
        Initialize LSTM model
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of hidden units in LSTM layers
            num_layers (int): Number of LSTM layers
            output_size (int): Number of output features
            dropout (float): Dropout rate for regularization
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            torch.Tensor: Output predictions
        """
        batch_size = x.size(0)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x.unsqueeze(-1), (h0, c0))
        
        # Take the last time step output
        output = self.dropout(lstm_out[:, -1, :])
        output = self.fc(output)
        
        return output


class ElectricityPricePredictor:
    """Main class for electricity price prediction using LSTM"""
    
    def __init__(self, sequence_length=48, test_size=0.2, batch_size=64):
        """
        Initialize the predictor
        
        Args:
            sequence_length (int): Length of input sequences (number of time steps)
            test_size (float): Proportion of data to use for testing
            batch_size (int): Batch size for training
        """
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = MinMaxScaler()
        self.model = None
        
    def prepare_data(self, data, target_column):
        """
        Prepare time series data for training
        
        Args:
            data (pd.DataFrame): Input dataframe with datetime and price columns
            target_column (str): Name of the target column to predict
            
        Returns:
            tuple: Training and testing data loaders, and fitted scaler
        """
        # Extract price data
        prices = data[target_column].values.reshape(-1, 1)
        
        # Normalize data
        prices_scaled = self.scaler.fit_transform(prices)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(prices_scaled)):
            X.append(prices_scaled[i-self.sequence_length:i, 0])
            y.append(prices_scaled[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Train/test split
        split_idx = int(len(X) * (1 - self.test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create datasets and data loaders
        train_dataset = TimeSeriesDataset(X_train, y_train)
        test_dataset = TimeSeriesDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def build_model(self, input_size=1, hidden_size=64, num_layers=2, 
                    output_size=1, dropout=0.2, learning_rate=0.001):
        """
        Build and initialize the LSTM model
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of hidden units
            num_layers (int): Number of LSTM layers
            output_size (int): Number of output features
            dropout (float): Dropout rate
            learning_rate (float): Learning rate for optimizer
        """
        self.model = LSTMModel(input_size, hidden_size, num_layers, 
                              output_size, dropout)
        self.model = self.model.to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        print(f"Model built with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Using device: {self.device}")
    
    def train_model(self, train_loader, test_loader, epochs=50):
        """
        Train the LSTM model
        
        Args:
            train_loader: Training data loader
            test_loader: Testing data loader
            epochs (int): Number of training epochs
            
        Returns:
            tuple: Training and validation loss histories
        """
        train_losses = []
        test_losses = []
        
        print("Starting training...")
        print("-" * 50)
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            test_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs.squeeze(), batch_y)
                    test_loss += loss.item()
            
            # Record losses
            avg_train_loss = train_loss / len(train_loader)
            avg_test_loss = test_loss / len(test_loader)
            
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}] - '
                      f'Train Loss: {avg_train_loss:.6f}, '
                      f'Test Loss: {avg_test_loss:.6f}')
        
        print("Training completed!")
        return train_losses, test_losses
    
    def make_predictions(self, test_loader):
        """
        Make predictions using the trained model
        
        Args:
            test_loader: Test data loader
            
        Returns:
            tuple: Predictions and actual values (denormalized)
        """
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                
                predictions.extend(outputs.squeeze().cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())
        
        # Denormalize predictions
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals = self.scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
        
        return predictions, actuals
    
    def evaluate_model(self, predictions, actuals):
        """
        Evaluate model performance
        
        Args:
            predictions (np.array): Predicted values
            actuals (np.array): Actual values
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        
        # Calculate MAPE safely (avoid division by zero)
        mask = actuals != 0
        mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
        
        return metrics
    
    def plot_results(self, predictions, actuals, train_losses, test_losses, save_path=None):
        """
        Plot training results and predictions
        
        Args:
            predictions (np.array): Predicted values
            actuals (np.array): Actual values
            train_losses (list): Training loss history
            test_losses (list): Validation loss history
            save_path (str): Path to save the plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('LSTM Model Prediction Analysis', fontsize=16)
        
        # 1. Training loss curve
        axes[0, 0].plot(train_losses, label='Training Loss', color='blue')
        axes[0, 0].plot(test_losses, label='Validation Loss', color='red')
        axes[0, 0].set_title('Training/Validation Loss Curve')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. Prediction vs actual (first 1000 points)
        n_points = min(1000, len(predictions))
        axes[0, 1].plot(actuals[:n_points], label='Actual', alpha=0.7)
        axes[0, 1].plot(predictions[:n_points], label='Predicted', alpha=0.7)
        axes[0, 1].set_title(f'Predicted vs Actual (First {n_points} points)')
        axes[0, 1].set_xlabel('Time Points')
        axes[0, 1].set_ylabel('Price (JPY/kWh)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Scatter plot
        axes[1, 0].scatter(actuals, predictions, alpha=0.5)
        axes[1, 0].plot([actuals.min(), actuals.max()], 
                       [actuals.min(), actuals.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual Values')
        axes[1, 0].set_ylabel('Predicted Values')
        axes[1, 0].set_title('Predicted vs Actual Scatter Plot')
        axes[1, 0].grid(True)
        
        # 4. Error distribution
        errors = actuals - predictions
        axes[1, 1].hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].set_title('Prediction Error Distribution')
        axes[1, 1].set_xlabel('Error (JPY/kWh)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the trained model
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'sequence_length': self.sequence_length
            }, filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save. Please train the model first.")
    
    def load_model(self, filepath, model_params):
        """
        Load a pre-trained model
        
        Args:
            filepath (str): Path to the saved model
            model_params (dict): Model architecture parameters
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.build_model(**model_params)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.sequence_length = checkpoint['sequence_length']
        
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    print("Japan Electricity Price Prediction with LSTM")
    print("=" * 50)
    
    # Load data
    try:
        df = pd.read_csv('Japan_ImbalancePrice.csv')
        df['date_time'] = pd.to_datetime(df['date_time'])
        print(f"Data loaded successfully: {df.shape}")
    except FileNotFoundError:
        print("Data file 'Japan_ImbalancePrice.csv' not found.")
        print("Please ensure the data file is in the same directory.")
        exit(1)
    
    # Initialize predictor
    predictor = ElectricityPricePredictor(sequence_length=48, test_size=0.2, batch_size=64)
    
    # Prepare data
    target_column = 'Tokyo_ibprice'
    data = df[['date_time', target_column]].copy()
    train_loader, test_loader = predictor.prepare_data(data, target_column)
    
    # Build and train model
    predictor.build_model(hidden_size=64, num_layers=2, dropout=0.2, learning_rate=0.001)
    train_losses, test_losses = predictor.train_model(train_loader, test_loader, epochs=50)
    
    # Make predictions and evaluate
    predictions, actuals = predictor.make_predictions(test_loader)
    metrics = predictor.evaluate_model(predictions, actuals)
    
    # Print results
    print("\n" + "=" * 30)
    print("MODEL EVALUATION RESULTS")
    print("=" * 30)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot results
    predictor.plot_results(predictions, actuals, train_losses, test_losses, 
                          save_path='prediction_results.png')
    
    # Save model
    predictor.save_model('lstm_electricity_model.pth')
    
    print("\nAnalysis completed successfully!")
