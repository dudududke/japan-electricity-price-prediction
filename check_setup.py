#!/usr/bin/env python3
"""
Quick setup verification script for Japan Electricity Price Prediction

This script checks if your environment is properly set up to run the project.

Usage:
    python check_setup.py

Author: JIEKAI WU
Date: August 2025
"""

import sys
import os
import importlib
from pathlib import Path


def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro} (Good)")
        return True
    else:
        print(f"   ✗ Python {version.major}.{version.minor}.{version.micro} (Need Python 3.7+)")
        return False


def check_dependencies():
    """Check required dependencies"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'torch',
        'pandas', 
        'numpy',
        'matplotlib',
        'seaborn',
        'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"   ✓ {package}")
        except ImportError:
            print(f"   ✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n   Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True


def check_project_structure():
    """Check project file structure"""
    print("\n📁 Checking project structure...")
    
    required_files = [
        'lstm_predictor.py',
        'data_analysis.py',
        'utils.py',
        'config.py',
        'requirements.txt',
        'README.md'
    ]
    
    required_dirs = [
        'examples'
    ]
    
    all_good = True
    
    # Check files
    for file in required_files:
        if Path(file).exists():
            print(f"   ✓ {file}")
        else:
            print(f"   ✗ {file} (missing)")
            all_good = False
    
    # Check directories
    for directory in required_dirs:
        if Path(directory).is_dir():
            print(f"   ✓ {directory}/")
        else:
            print(f"   ✗ {directory}/ (missing)")
            all_good = False
    
    return all_good


def check_data_file():
    """Check if data file exists"""
    print("\n📊 Checking data file...")
    
    data_file = 'Japan_ImbalancePrice.csv'
    
    if Path(data_file).exists():
        print(f"   ✓ {data_file}")
        
        # Quick data validation
        try:
            import pandas as pd
            df = pd.read_csv(data_file, nrows=5)  # Read first 5 rows only
            expected_columns = ['date_time'] + [f'{region}_ibprice' for region in 
                              ['Hokkaido', 'Tohoku', 'Tokyo', 'Chubu', 
                               'Hokuriku', 'Kansai', 'Chugoku', 'Shikoku', 'Kyushu']]
            
            if all(col in df.columns for col in expected_columns):
                print(f"   ✓ Data structure looks correct")
                return True
            else:
                print(f"   ⚠ Data structure may be incorrect")
                print(f"     Expected columns: {expected_columns}")
                print(f"     Found columns: {list(df.columns)}")
                return False
                
        except Exception as e:
            print(f"   ⚠ Could not validate data file: {e}")
            return False
    else:
        print(f"   ✗ {data_file} (missing)")
        print(f"   📝 See DATA_SETUP.md for instructions on how to get the data")
        return False


def main():
    """Main function"""
    print("🔧 Japan Electricity Price Prediction - Setup Check")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Data File", check_data_file)
    ]
    
    results = []
    for check_name, check_func in checks:
        result = check_func()
        results.append((check_name, result))
    
    print("\n" + "=" * 60)
    print("📋 SETUP CHECK SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for check_name, result in results:
        status = "PASS" if result else "FAIL"
        icon = "✓" if result else "✗"
        print(f"{icon} {check_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All checks passed! You're ready to run the project.")
        print("\nNext steps:")
        print("1. python data_analysis.py        # Explore the data")
        print("2. python lstm_predictor.py       # Train the model")
        print("3. python examples/basic_usage.py # Run basic example")
    else:
        print("⚠️  Some checks failed. Please address the issues above.")
        print("\nFor help:")
        print("- Check README.md for setup instructions")
        print("- Check DATA_SETUP.md for data acquisition")
        print("- Install missing dependencies with: pip install -r requirements.txt")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
