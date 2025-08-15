# Data Information

## Dataset Included

✅ **Good News**: The dataset `Japan_ImbalancePrice.csv` is already included in this repository for your convenience!

## Dataset Information

This project uses the Japan Electricity Imbalance Prices dataset from Kaggle.

### Data Source
- **Dataset**: Japan Imbalance Prices
- **Author**: @mitsuyasuhoshino
- **Platform**: Kaggle
- **URL**: https://www.kaggle.com/datasets/mitsuyasuhoshino/japan-imbalance-prices

### File Requirements

The project expects a CSV file named `Japan_ImbalancePrice.csv` in the project root directory with the following structure:

```csv
date_time,Hokkaido_ibprice,Tohoku_ibprice,Tokyo_ibprice,Chubu_ibprice,Hokuriku_ibprice,Kansai_ibprice,Chugoku_ibprice,Shikoku_ibprice,Kyushu_ibprice
2022/03/01 00:00:00,16.44,16.44,16.44,17.62,17.62,17.62,17.62,17.62,12.04
2022/03/01 00:30:00,12.04,12.04,12.04,12.04,12.04,12.04,12.04,12.04,12.04
...
```

### How to Get the Data

1. **Option 1: Download from Kaggle**
   ```bash
   # Install kaggle CLI
   pip install kaggle
   
   # Setup your Kaggle API credentials
   # Place kaggle.json in ~/.kaggle/
   
   # Download the dataset
   kaggle datasets download -d mitsuyasuhoshino/japan-imbalance-prices
   
   # Extract and rename the file
   unzip japan-imbalance-prices.zip
   mv [original_filename].csv Japan_ImbalancePrice.csv
   ```

2. **Option 2: Manual Download**
   - Visit: https://www.kaggle.com/datasets/mitsuyasuhoshino/japan-imbalance-prices
   - Click "Download" button
   - Extract the CSV file
   - Rename it to `Japan_ImbalancePrice.csv`
   - Place it in the project root directory

### Data Validation

Once you have the data file, you can validate it by running:

```python
import pandas as pd
from utils import validate_data, load_data

# Load and validate data
try:
    df = load_data('Japan_ImbalancePrice.csv')
    if validate_data(df):
        print("✓ Data validation successful!")
        print(f"Dataset shape: {df.shape}")
    else:
        print("✗ Data validation failed!")
except Exception as e:
    print(f"Error loading data: {e}")
```

### Expected Data Properties

- **Time Range**: March 2022 - August 2025
- **Frequency**: 30-minute intervals
- **Regions**: 9 Japanese electricity regions
- **Total Records**: ~60,546 entries
- **Data Quality**: Complete dataset with no missing values

### Troubleshooting

If you encounter issues with the data:

1. **File not found**: Ensure `Japan_ImbalancePrice.csv` is in the project root
2. **Column names mismatch**: Check that column names match the expected format
3. **Date format issues**: The `date_time` column should be parseable by pandas
4. **Missing data**: The dataset should be complete with no NaN values

For any data-related issues, please refer to the original Kaggle dataset page or contact the dataset author.
