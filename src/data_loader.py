# Part 1: Data Loading and Basic Exploration

import pandas as pd
import numpy as np
import os

def load_data(csv_path='data/metadata.csv'):
    """
    Load metadata.csv directly into DataFrame
    
    Args:
        csv_path (str): Path to the CSV file
    
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    print("=" * 50)
    print("PART 1: DATA LOADING AND BASIC EXPLORATION")
    print("=" * 50)
    
    try:
        # Check if file exists
        if not os.path.exists(csv_path):
            print(f"❌ File not found: {csv_path}")
            print(f"💡 Please ensure metadata.csv is in the data/ directory")
            return None
        
        # Load the CSV file
        print(f"📂 Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"✓ Successfully loaded metadata.csv")
        print(f"✓ Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Check file size
        file_size = os.path.getsize(csv_path) / (1024 * 1024)  # MB
        print(f"✓ File size: {file_size:.2f} MB")
        
        return df
                
    except pd.errors.EmptyDataError:
        print(f"❌ The CSV file is empty or corrupted")
        return None
    except pd.errors.ParserError as e:
        print(f"❌ Error parsing CSV file: {e}")
        return None
    except MemoryError:
        print(f"❌ Not enough memory to load the file")
        print(f"💡 Try loading a smaller subset or increase available RAM")
        return None
    except Exception as e:
        print(f"❌ Unexpected error loading data: {e}")
        return None

def basic_exploration(df):
    """
    Perform basic data exploration and return summary information
    
    Args:
        df (pandas.DataFrame): Dataset to explore
    
    Returns:
        dict: Summary information about the dataset
    """
    print("\n" + "-" * 40)
    print("BASIC DATA EXPLORATION")
    print("-" * 40)
    
    # Basic information
    print(f"📊 Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"📅 Columns: {list(df.columns[:5])}..." if len(df.columns) > 5 else f"📅 Columns: {list(df.columns)}")
    
    # Check data types
    print(f"\n🔢 Data types:")
    for dtype in df.dtypes.value_counts().items():
        print(f"   {dtype[0]}: {dtype[1]} columns")
    
    # Missing values analysis
    print(f"\n❓ Missing values:")
    missing_data = df.isnull().sum()
    total_missing = missing_data.sum()
    
    if total_missing > 0:
        missing_percent = (missing_data / len(df)) * 100
        missing_summary = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percent': missing_percent
        }).sort_values('Missing_Percent', ascending=False)
        
        # Show top 10 columns with missing values
        top_missing = missing_summary[missing_summary['Missing_Count'] > 0].head(10)
        for col, row in top_missing.iterrows():
            print(f"   {col}: {row['Missing_Count']:,} ({row['Missing_Percent']:.1f}%)")
    else:
        print("   ✓ No missing values found!")
    
    # Basic statistics for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        print(f"\n📈 Numerical columns: {len(numerical_cols)}")
        for col in numerical_cols[:3]:  # Show stats for first 3 numerical columns
            print(f"   {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")
    
    # Sample of the data
    print(f"\n📋 First 3 rows:")
    print(df.head(3).to_string())
    
    # Return summary information
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'missing_data': missing_data.to_dict(),
        'data_types': df.dtypes.to_dict(),
        'numerical_columns': list(numerical_cols)
    }
    
    return summary

def check_data_quality(df):
    """
    Perform data quality checks
    
    Args:
        df (pandas.DataFrame): Dataset to check
    
    Returns:
        dict: Data quality report
    """
    print(f"\n🔍 DATA QUALITY CHECKS:")
    print("-" * 30)
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"   Duplicate rows: {duplicates:,}")
    
    # Check for completely empty rows
    empty_rows = df.isnull().all(axis=1).sum()
    print(f"   Completely empty rows: {empty_rows:,}")
    
    # Check important columns
    important_columns = ['title', 'abstract', 'journal', 'publish_time', 'authors']
    existing_important = [col for col in important_columns if col in df.columns]
    print(f"   Important columns present: {len(existing_important)}/{len(important_columns)}")
    
    quality_report = {
        'duplicates': duplicates,
        'empty_rows': empty_rows,
        'important_columns_present': existing_important,
        'total_records': len(df)
    }
    
    return quality_report

def main():
    """Main function to run data loading and exploration"""
    # Load data
    df = load_data()
    
    if df is not None:
        # Basic exploration
        summary = basic_exploration(df)
        
        # Quality checks
        quality = check_data_quality(df)
        
        print(f"\n✅ Data loading completed successfully!")
        print(f"✅ Ready for cleaning and analysis")
        
        return df, summary, quality
    else:
        print(f"\n❌ Data loading failed!")
        return None, None, None

if __name__ == "__main__":
    df, summary, quality = main()