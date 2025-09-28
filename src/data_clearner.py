# Part 2: Data Cleaning and Preparation

import pandas as pd
import numpy as np
from datetime import datetime
import os

def clean_missing_data(df):
    """
    Handle missing data in the dataset
    
    Args:
        df (pandas.DataFrame): Raw dataset
    
    Returns:
        pandas.DataFrame: Dataset with missing data handled
    """
    print("\n" + "=" * 50)
    print("PART 2: DATA CLEANING AND PREPARATION")
    print("=" * 50)
    
    print("🧹 CLEANING MISSING DATA:")
    print("-" * 30)
    
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    # Remove rows without titles (essential for analysis)
    df_clean = df_clean.dropna(subset=['title'])
    removed_no_title = initial_rows - len(df_clean)
    print(f"   ❌ Removed {removed_no_title:,} rows without titles")
    
    # Handle missing abstracts
    df_clean['abstract'] = df_clean['abstract'].fillna('')
    abstract_filled = df['abstract'].isnull().sum()
    print(f"   🔧 Filled {abstract_filled:,} missing abstracts with empty string")
    
    # Handle missing journal names
    if 'journal' in df_clean.columns:
        df_clean['journal'] = df_clean['journal'].fillna('Unknown Journal')
        journal_filled = df['journal'].isnull().sum() if 'journal' in df.columns else 0
        print(f"   🔧 Filled {journal_filled:,} missing journal names")
    
    # Handle missing authors
    if 'authors' in df_clean.columns:
        df_clean['authors'] = df_clean['authors'].fillna('Unknown Authors')
        authors_filled = df['authors'].isnull().sum() if 'authors' in df.columns else 0
        print(f"   🔧 Filled {authors_filled:,} missing author names")
    
    print(f"   ✅ Dataset size after cleaning: {len(df_clean):,} rows")
    
    return df_clean

def process_dates(df_clean):
    """
    Process and clean date columns
    
    Args:
        df_clean (pandas.DataFrame): Dataset with basic cleaning done
    
    Returns:
        pandas.DataFrame: Dataset with processed dates
    """
    print(f"\n📅 PROCESSING DATES:")
    print("-" * 30)
    
    if 'publish_time' in df_clean.columns:
        # Convert to datetime
        df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
        
        # Extract publication year
        df_clean['publication_year'] = df_clean['publish_time'].dt.year
        
        # Filter realistic years (2019 onwards for COVID-19 research)
        current_year = datetime.now().year
        initial_count = len(df_clean)
        
        df_clean = df_clean[
            (df_clean['publication_year'] >= 2019) & 
            (df_clean['publication_year'] <= current_year)
        ]
        
        filtered_count = initial_count - len(df_clean)
        print(f"   📅 Converted publish_time to datetime")
        print(f"   📅 Extracted publication_year column")
        print(f"   📅 Filtered to years 2019-{current_year}")
        print(f"   ❌ Removed {filtered_count:,} rows with invalid years")
        
        # Show year distribution
        if len(df_clean) > 0:
            year_counts = df_clean['publication_year'].value_counts().sort_index()
            print(f"   📊 Year distribution:")
            for year, count in year_counts.head(10).items():
                print(f"      {year}: {count:,} papers")
    else:
        print(f"   ⚠️  No 'publish_time' column found")
    
    return df_clean

def create_derived_columns(df_clean):
    """
    Create new useful columns for analysis
    
    Args:
        df_clean (pandas.DataFrame): Cleaned dataset
    
    Returns:
        pandas.DataFrame: Dataset with new derived columns
    """
    print(f"\n🔧 CREATING DERIVED COLUMNS:")
    print("-" * 30)
    
    # Abstract word count
    if 'abstract' in df_clean.columns:
        df_clean['abstract_word_count'] = df_clean['abstract'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) and str(x).strip() != '' else 0
        )
        avg_abstract_words = df_clean['abstract_word_count'].mean()
        print(f"   ✅ Created 'abstract_word_count' (avg: {avg_abstract_words:.1f} words)")
    
    # Title word count
    if 'title' in df_clean.columns:
        df_clean['title_word_count'] = df_clean['title'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        avg_title_words = df_clean['title_word_count'].mean()
        print(f"   ✅ Created 'title_word_count' (avg: {avg_title_words:.1f} words)")
    
    # Has abstract flag
    if 'abstract' in df_clean.columns:
        df_clean['has_abstract'] = df_clean['abstract'].apply(
            lambda x: len(str(x).strip()) > 0 if pd.notna(x) else False
        )
        papers_with_abstract = df_clean['has_abstract'].sum()
        abstract_percentage = (papers_with_abstract / len(df_clean)) * 100
        print(f"   ✅ Created 'has_abstract' ({papers_with_abstract:,} papers, {abstract_percentage:.1f}%)")
    
    # Clean title for analysis
    if 'title' in df_clean.columns:
        df_clean['title_clean'] = df_clean['title'].str.lower().str.strip()
        print(f"   ✅ Created 'title_clean' for text analysis")
    
    return df_clean

def remove_duplicates_and_outliers(df_clean):
    """
    Remove duplicates and handle outliers
    
    Args:
        df_clean (pandas.DataFrame): Dataset with derived columns
    
    Returns:
        pandas.DataFrame: Final cleaned dataset
    """
    print(f"\n🔍 REMOVING DUPLICATES AND OUTLIERS:")
    print("-" * 30)
    
    initial_count = len(df_clean)
    
    # Remove duplicate titles
    df_final = df_clean.drop_duplicates(subset=['title'], keep='first')
    duplicate_count = initial_count - len(df_final)
    print(f"   ❌ Removed {duplicate_count:,} duplicate titles")
    
    # Handle outliers in word counts (optional)
    if 'abstract_word_count' in df_final.columns:
        # Remove abstracts that are too long (likely errors)
        max_reasonable_words = 1000
        long_abstracts = (df_final['abstract_word_count'] > max_reasonable_words).sum()
        if long_abstracts > 0:
            df_final = df_final[df_final['abstract_word_count'] <= max_reasonable_words]
            print(f"   ❌ Removed {long_abstracts:,} papers with unreasonably long abstracts")
    
    # Final data quality check
    print(f"   ✅ Final dataset size: {len(df_final):,} rows")
    
    return df_final

def save_cleaned_data(df_final, output_path='data/metadata_clean.csv'):
    """
    Save the cleaned dataset
    
    Args:
        df_final (pandas.DataFrame): Final cleaned dataset
        output_path (str): Path to save the cleaned data
    """
    print(f"\n💾 SAVING CLEANED DATA:")
    print("-" * 30)
    
    try:
        df_final.to_csv(output_path, index=False)
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"   ✅ Saved cleaned data to: {output_path}")
        print(f"   ✅ File size: {file_size:.2f} MB")
        print(f"   ✅ Ready for analysis!")
    except Exception as e:
        print(f"   ❌ Error saving file: {e}")

def generate_cleaning_report(df_original, df_final):
    """
    Generate a summary report of the cleaning process
    
    Args:
        df_original (pandas.DataFrame): Original dataset
        df_final (pandas.DataFrame): Final cleaned dataset
    
    Returns:
        dict: Cleaning report summary
    """
    print(f"\n📋 CLEANING SUMMARY REPORT:")
    print("-" * 30)
    
    original_rows = len(df_original)
    final_rows = len(df_final)
    rows_removed = original_rows - final_rows
    retention_rate = (final_rows / original_rows) * 100
    
    print(f"   📊 Original rows: {original_rows:,}")
    print(f"   📊 Final rows: {final_rows:,}")
    print(f"   📊 Rows removed: {rows_removed:,}")
    print(f"   📊 Data retention: {retention_rate:.1f}%")
    
    # Column comparison
    original_cols = set(df_original.columns)
    final_cols = set(df_final.columns)
    added_cols = final_cols - original_cols
    
    if added_cols:
        print(f"   ✅ New columns added: {list(added_cols)}")
    
    report = {
        'original_rows': original_rows,
        'final_rows': final_rows,
        'rows_removed': rows_removed,
        'retention_rate': retention_rate,
        'columns_added': list(added_cols),
        'cleaning_successful': True
    }
    
    return report

def main(df_original):
    """
    Main function to run complete data cleaning pipeline
    
    Args:
        df_original (pandas.DataFrame): Original dataset from data_loader
    
    Returns:
        pandas.DataFrame: Cleaned dataset ready for analysis
    """
    if df_original is None:
        print("❌ No data provided for cleaning!")
        return None
    
    # Step 1: Handle missing data
    df_clean = clean_missing_data(df_original)
    
    # Step 2: Process dates
    df_clean = process_dates(df_clean)
    
    # Step 3: Create derived columns
    df_clean = create_derived_columns(df_clean)
    
    # Step 4: Remove duplicates and outliers
    df_final = remove_duplicates_and_outliers(df_clean)
    
    # Step 5: Save cleaned data
    save_cleaned_data(df_final)
    
    # Step 6: Generate report
    report = generate_cleaning_report(df_original, df_final)
    
    print(f"\n✅ DATA CLEANING COMPLETED!")
    print(f"✅ Dataset ready for analysis: {len(df_final):,} rows")
    
    return df_final

if __name__ == "__main__":
    # For testing independently
    from data_loader import load_data
    
    # Load data
    df = load_data()
    
    if df is not None:
        # Clean data
        cleaned_df = main(df)
    else:
        print("❌ Could not load data for cleaning!")