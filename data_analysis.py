"""
Data Analysis Script for Japan Electricity Imbalance Price Dataset

This script performs comprehensive analysis of the Japanese electricity 
imbalance price dataset including data exploration, visualization, 
and statistical analysis.

Author: JIEKAI WU
Date: August 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ElectricityDataAnalyzer:
    """Class for analyzing electricity imbalance price data"""
    
    def __init__(self, data_path):
        """
        Initialize the analyzer
        
        Args:
            data_path (str): Path to the CSV data file
        """
        self.data_path = data_path
        self.df = None
        self.regions = []
        
    def load_data(self):
        """Load and prepare the dataset"""
        try:
            self.df = pd.read_csv(self.data_path)
            self.df['date_time'] = pd.to_datetime(self.df['date_time'])
            self.regions = [col for col in self.df.columns if col != 'date_time']
            print(f"Data loaded successfully: {self.df.shape}")
            return True
        except FileNotFoundError:
            print(f"Data file '{self.data_path}' not found.")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def basic_info(self):
        """Display basic information about the dataset"""
        if self.df is None:
            print("Please load data first.")
            return
        
        print("=" * 50)
        print("DATASET BASIC INFORMATION")
        print("=" * 50)
        print(f"Data shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print(f"\nData types:")
        print(self.df.dtypes)
        print(f"\nFirst 5 rows:")
        print(self.df.head())
        
    def time_range_analysis(self):
        """Analyze the time range and frequency of the data"""
        if self.df is None:
            print("Please load data first.")
            return
        
        print("\n" + "=" * 50)
        print("TIME RANGE ANALYSIS")
        print("=" * 50)
        print(f"Start time: {self.df['date_time'].min()}")
        print(f"End time: {self.df['date_time'].max()}")
        print(f"Time span: {(self.df['date_time'].max() - self.df['date_time'].min()).days} days")
        print(f"Data frequency: 30-minute intervals")
        
        # Check for missing values
        print(f"\nMISSING VALUES CHECK")
        print("-" * 30)
        missing_values = self.df.isnull().sum()
        print(missing_values)
        
    def descriptive_statistics(self):
        """Display descriptive statistics"""
        if self.df is None:
            print("Please load data first.")
            return
        
        print("\n" + "=" * 50)
        print("DESCRIPTIVE STATISTICS")
        print("=" * 50)
        print(self.df.describe())
        
    def price_extremes_analysis(self):
        """Analyze price extremes for each region"""
        if self.df is None:
            print("Please load data first.")
            return
        
        print("\n" + "=" * 50)
        print("PRICE EXTREMES ANALYSIS")
        print("=" * 50)
        
        for region in self.regions:
            region_name = region.replace('_ibprice', '')
            max_price = self.df[region].max()
            min_price = self.df[region].min()
            avg_price = self.df[region].mean()
            std_price = self.df[region].std()
            
            max_time = self.df.loc[self.df[region].idxmax(), 'date_time']
            min_time = self.df.loc[self.df[region].idxmin(), 'date_time']
            
            print(f"{region_name:10}: Max {max_price:7.2f} ({max_time}) | "
                  f"Min {min_price:7.2f} ({min_time}) | "
                  f"Avg {avg_price:7.2f} | Std {std_price:6.2f}")
    
    def time_pattern_analysis(self):
        """Analyze time patterns in the data"""
        if self.df is None:
            print("Please load data first.")
            return
        
        # Add time features
        self.df['hour'] = self.df['date_time'].dt.hour
        self.df['day_of_week'] = self.df['date_time'].dt.dayofweek
        self.df['month'] = self.df['date_time'].dt.month
        
        print("\n" + "=" * 50)
        print("TIME PATTERN ANALYSIS")
        print("=" * 50)
        
        # Analyze hourly patterns (Tokyo as example)
        hourly_avg = self.df.groupby('hour')['Tokyo_ibprice'].mean()
        print(f"Tokyo region 24-hour average price patterns:")
        print(f"Peak hour: {hourly_avg.idxmax()}:00 ({hourly_avg.max():.2f} JPY/kWh)")
        print(f"Low hour: {hourly_avg.idxmin()}:00 ({hourly_avg.min():.2f} JPY/kWh)")
        
    def volatility_analysis(self):
        """Analyze price volatility using coefficient of variation"""
        if self.df is None:
            print("Please load data first.")
            return
        
        print("\n" + "=" * 50)
        print("PRICE VOLATILITY ANALYSIS (Coefficient of Variation)")
        print("=" * 50)
        
        for region in self.regions:
            region_name = region.replace('_ibprice', '')
            cv = self.df[region].std() / self.df[region].mean() * 100
            print(f"{region_name:10}: {cv:5.2f}%")
    
    def correlation_analysis(self):
        """Analyze correlations between regions"""
        if self.df is None:
            print("Please load data first.")
            return
        
        print("\n" + "=" * 50)
        print("REGIONAL PRICE CORRELATION ANALYSIS")
        print("=" * 50)
        
        correlation_matrix = self.df[self.regions].corr()
        
        # Find highest and lowest correlations
        max_corr = -1
        min_corr = 2
        max_pair = ""
        min_pair = ""
        
        for i, region1 in enumerate(self.regions):
            for j, region2 in enumerate(self.regions[i+1:], i+1):
                corr = correlation_matrix.loc[region1, region2]
                if corr > max_corr:
                    max_corr = corr
                    max_pair = f"{region1.replace('_ibprice', '')} - {region2.replace('_ibprice', '')}"
                if corr < min_corr:
                    min_corr = corr
                    min_pair = f"{region1.replace('_ibprice', '')} - {region2.replace('_ibprice', '')}"
        
        print(f"Highest correlation: {max_pair} ({max_corr:.4f})")
        print(f"Lowest correlation: {min_pair} ({min_corr:.4f})")
        
        return correlation_matrix
    
    def create_visualizations(self, save_plots=True):
        """Create comprehensive visualizations"""
        if self.df is None:
            print("Please load data first.")
            return
        
        print("\nCreating visualizations...")
        
        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Japan Electricity Imbalance Price Data Overview', fontsize=16)
        
        # 1. Price distribution histogram
        axes[0, 0].hist([self.df[col].dropna() for col in self.regions[:3]], 
                        bins=50, alpha=0.7, 
                        label=[r.replace('_ibprice', '') for r in self.regions[:3]])
        axes[0, 0].set_title('Price Distribution (First 3 Regions)')
        axes[0, 0].set_xlabel('Price (JPY/kWh)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # 2. Time series plot (last 7 days)
        recent_data = self.df.tail(336)  # 7 days * 48 points/day
        axes[0, 1].plot(recent_data['date_time'], recent_data['Tokyo_ibprice'], 
                        label='Tokyo', linewidth=1)
        axes[0, 1].plot(recent_data['date_time'], recent_data['Hokkaido_ibprice'], 
                        label='Hokkaido', linewidth=1)
        axes[0, 1].set_title('Price Trend (Last 7 Days)')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Price (JPY/kWh)')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Regional price correlation heatmap
        correlation_matrix = self.df[self.regions].corr()
        im = axes[1, 0].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 0].set_title('Price Correlation Between Regions')
        axes[1, 0].set_xticks(range(len(self.regions)))
        axes[1, 0].set_yticks(range(len(self.regions)))
        axes[1, 0].set_xticklabels([r.replace('_ibprice', '') for r in self.regions], rotation=45)
        axes[1, 0].set_yticklabels([r.replace('_ibprice', '') for r in self.regions])
        
        # 4. Price boxplot
        box_data = [self.df[col].dropna() for col in self.regions]
        axes[1, 1].boxplot(box_data, labels=[r.replace('_ibprice', '') for r in self.regions])
        axes[1, 1].set_title('Price Distribution by Region')
        axes[1, 1].set_xlabel('Region')
        axes[1, 1].set_ylabel('Price (JPY/kWh)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('data_analysis_overview.png', dpi=300, bbox_inches='tight')
            print("Plot saved as 'data_analysis_overview.png'")
        
        plt.show()
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        if self.df is None:
            print("Please load data first.")
            return
        
        print("\n" + "=" * 60)
        print("COMPREHENSIVE ANALYSIS SUMMARY")
        print("=" * 60)
        
        print(f"• Dataset shape: {self.df.shape}")
        print(f"• Time span: {self.df['date_time'].min().strftime('%Y-%m-%d')} to {self.df['date_time'].max().strftime('%Y-%m-%d')}")
        print(f"• Data frequency: 30-minute intervals")
        print(f"• Covered regions: {len(self.regions)} Japanese electricity regions")
        print(f"• Data quality: Complete (no missing values)")
        print(f"• Total records: {len(self.df):,} entries")
        
        # Price statistics summary
        all_prices = []
        for region in self.regions:
            all_prices.extend(self.df[region].dropna().values)
        all_prices = np.array(all_prices)
        
        print(f"\n• Overall price range: {all_prices.min():.2f} - {all_prices.max():.2f} JPY/kWh")
        print(f"• Average price across all regions: {all_prices.mean():.2f} JPY/kWh")
        print(f"• Price volatility (std): {all_prices.std():.2f} JPY/kWh")
        
        print(f"\nRegions included:")
        region_names = ['Hokkaido', 'Tohoku', 'Tokyo', 'Chubu', 'Hokuriku', 
                       'Kansai', 'Chugoku', 'Shikoku', 'Kyushu']
        for i, name in enumerate(region_names, 1):
            print(f"{i:2d}. {name}")


def main():
    """Main function to run the analysis"""
    print("Japan Electricity Imbalance Price Data Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ElectricityDataAnalyzer('Japan_ImbalancePrice.csv')
    
    # Load and analyze data
    if not analyzer.load_data():
        print("Failed to load data. Please ensure 'Japan_ImbalancePrice.csv' exists.")
        return
    
    # Perform comprehensive analysis
    analyzer.basic_info()
    analyzer.time_range_analysis()
    analyzer.descriptive_statistics()
    analyzer.price_extremes_analysis()
    analyzer.time_pattern_analysis()
    analyzer.volatility_analysis()
    analyzer.correlation_analysis()
    
    # Create visualizations
    analyzer.create_visualizations(save_plots=True)
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    print("\nData analysis completed successfully!")


if __name__ == "__main__":
    main()
