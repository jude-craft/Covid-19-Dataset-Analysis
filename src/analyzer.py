# src/analyzer.py
# Part 3: Data Analysis Functions

import pandas as pd
import numpy as np
from collections import Counter
import re

def analyze_publications_by_year(df):
    """
    Analyze publication trends over time
    
    Args:
        df (pandas.DataFrame): Cleaned dataset
    
    Returns:
        pandas.Series: Publications count by year
    """
    print("\n" + "=" * 50)
    print("PART 3: DATA ANALYSIS")
    print("=" * 50)
    
    print("📈 PUBLICATIONS BY YEAR ANALYSIS:")
    print("-" * 40)
    
    if 'publication_year' not in df.columns:
        print("   ⚠️  No publication_year column found!")
        return None
    
    # Count publications by year
    year_counts = df['publication_year'].value_counts().sort_index()
    
    print(f"   📊 Publication trends:")
    for year, count in year_counts.items():
        print(f"      {int(year)}: {count:,} papers")
    
    # Calculate year-over-year growth
    print(f"\n   📈 Year-over-year growth:")
    for i in range(1, len(year_counts)):
        prev_year = year_counts.iloc[i-1]
        curr_year = year_counts.iloc[i]
        growth = ((curr_year - prev_year) / prev_year * 100) if prev_year > 0 else 0
        year_name = int(year_counts.index[i])
        print(f"      {year_name}: {growth:+.1f}% ({curr_year - prev_year:+,} papers)")
    
    # Peak year
    peak_year = year_counts.idxmax()
    peak_count = year_counts.max()
    print(f"\n   🔥 Peak year: {int(peak_year)} with {peak_count:,} papers")
    
    return year_counts

def analyze_top_journals(df, top_n=15):
    """
    Analyze top journals publishing COVID-19 research
    
    Args:
        df (pandas.DataFrame): Cleaned dataset
        top_n (int): Number of top journals to analyze
    
    Returns:
        pandas.Series: Top journals by publication count
    """
    print(f"\n📰 TOP {top_n} JOURNALS ANALYSIS:")
    print("-" * 40)
    
    if 'journal' not in df.columns:
        print("   ⚠️  No journal column found!")
        return None
    
    # Count papers by journal
    journal_counts = df['journal'].value_counts().head(top_n)
    
    print(f"   📊 Top publishing journals:")
    for i, (journal, count) in enumerate(journal_counts.items(), 1):
        percentage = (count / len(df)) * 100
        journal_short = journal[:50] + "..." if len(journal) > 50 else journal
        print(f"      {i:2d}. {journal_short}: {count:,} papers ({percentage:.1f}%)")
    
    # Journal diversity metrics
    total_journals = df['journal'].nunique()
    top_5_papers = journal_counts.head(5).sum()
    top_5_percentage = (top_5_papers / len(df)) * 100
    
    print(f"\n   📊 Journal diversity:")
    print(f"      Total unique journals: {total_journals:,}")
    print(f"      Top 5 journals publish: {top_5_percentage:.1f}% of all papers")
    
    return journal_counts

def analyze_title_words(df, top_n=20):
    """
    Analyze most frequent words in paper titles
    
    Args:
        df (pandas.DataFrame): Cleaned dataset
        top_n (int): Number of top words to analyze
    
    Returns:
        list: List of (word, count) tuples
    """
    print(f"\n🔤 TOP {top_n} WORDS IN TITLES ANALYSIS:")
    print("-" * 40)
    
    if 'title' not in df.columns:
        print("   ⚠️  No title column found!")
        return None
    
    # Combine all titles
    all_titles = ' '.join(df['title'].fillna('').astype(str))
    
    # Extract words (3+ characters, only letters)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_titles.lower())
    
    # Remove common stop words
    stop_words = {
        'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 'they', 'been',
        'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might',
        'can', 'did', 'does', 'was', 'were', 'but', 'not', 'you', 'all', 'any',
        'her', 'his', 'our', 'out', 'day', 'get', 'use', 'new', 'now', 'old',
        'see', 'him', 'two', 'way', 'who', 'its', 'said', 'each', 'she', 'which',
        'their', 'time', 'than', 'only', 'come', 'over', 'also', 'back', 'after',
        'first', 'well', 'year', 'work', 'such', 'make', 'even', 'most', 'take',
        'good', 'high', 'small', 'large', 'right', 'early', 'important', 'different'
    }
    
    # Filter words
    filtered_words = [word for word in words if word not in stop_words]
    
    # Count frequencies
    word_counts = Counter(filtered_words)
    top_words = word_counts.most_common(top_n)
    
    print(f"   📊 Most frequent words in titles:")
    for i, (word, count) in enumerate(top_words, 1):
        percentage = (count / len(df)) * 100
        print(f"      {i:2d}. {word}: {count:,} times ({percentage:.1f}% of papers)")
    
    # COVID-related terms analysis
    covid_terms = ['covid', 'coronavirus', 'sars', 'pandemic', 'vaccine', 'virus', 'infection']
    covid_word_counts = {}
    
    for term in covid_terms:
        count = sum(1 for word in filtered_words if term in word.lower())
        if count > 0:
            covid_word_counts[term] = count
    
    if covid_word_counts:
        print(f"\n   🦠 COVID-related terms:")
        for term, count in sorted(covid_word_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(df)) * 100
            print(f"      {term}: {count:,} times ({percentage:.1f}% of papers)")
    
    return top_words

def analyze_abstract_patterns(df):
    """
    Analyze patterns in paper abstracts
    
    Args:
        df (pandas.DataFrame): Cleaned dataset
    
    Returns:
        dict: Abstract analysis results
    """
    print(f"\n📝 ABSTRACT ANALYSIS:")
    print("-" * 40)
    
    if 'abstract' not in df.columns:
        print("   ⚠️  No abstract column found!")
        return None
    
    # Basic abstract statistics
    abstracts_available = df['has_abstract'].sum() if 'has_abstract' in df.columns else df['abstract'].notna().sum()
    abstract_coverage = (abstracts_available / len(df)) * 100
    
    print(f"   📊 Abstract statistics:")
    print(f"      Papers with abstracts: {abstracts_available:,} ({abstract_coverage:.1f}%)")
    
    if 'abstract_word_count' in df.columns:
        avg_words = df['abstract_word_count'].mean()
        median_words = df['abstract_word_count'].median()
        max_words = df['abstract_word_count'].max()
        
        print(f"      Average abstract length: {avg_words:.1f} words")
        print(f"      Median abstract length: {median_words:.1f} words")
        print(f"      Longest abstract: {max_words:,} words")
        
        # Abstract length categories
        short_abstracts = (df['abstract_word_count'] < 100).sum()
        medium_abstracts = ((df['abstract_word_count'] >= 100) & (df['abstract_word_count'] < 300)).sum()
        long_abstracts = (df['abstract_word_count'] >= 300).sum()
        
        print(f"\n   📊 Abstract length distribution:")
        print(f"      Short (<100 words): {short_abstracts:,} ({short_abstracts/len(df)*100:.1f}%)")
        print(f"      Medium (100-300 words): {medium_abstracts:,} ({medium_abstracts/len(df)*100:.1f}%)")
        print(f"      Long (>300 words): {long_abstracts:,} ({long_abstracts/len(df)*100:.1f}%)")
    
    return {
        'abstracts_available': abstracts_available,
        'abstract_coverage': abstract_coverage,
        'avg_words': df['abstract_word_count'].mean() if 'abstract_word_count' in df.columns else None
    }

def analyze_research_evolution(df):
    """
    Analyze how research topics evolved over time
    
    Args:
        df (pandas.DataFrame): Cleaned dataset
    
    Returns:
        dict: Research evolution analysis
    """
    print(f"\n🔬 RESEARCH EVOLUTION ANALYSIS:")
    print("-" * 40)
    
    if 'publication_year' not in df.columns or 'title' not in df.columns:
        print("   ⚠️  Missing required columns for evolution analysis!")
        return None
    
    # Analyze key terms by year
    key_terms = ['vaccine', 'treatment', 'diagnosis', 'prevention', 'mutation', 'variant', 'lockdown']
    evolution_data = {}
    
    for year in sorted(df['publication_year'].unique()):
        year_data = df[df['publication_year'] == year]
        year_titles = ' '.join(year_data['title'].fillna('').astype(str).str.lower())
        
        year_term_counts = {}
        for term in key_terms:
            count = len(re.findall(r'\b' + term + r'\b', year_titles))
            year_term_counts[term] = count
        
        evolution_data[int(year)] = year_term_counts
    
    print(f"   📊 Research focus evolution:")
    for term in key_terms:
        print(f"      {term.capitalize()}:")
        for year in sorted(evolution_data.keys()):
            count = evolution_data[year][term]
            year_total = len(df[df['publication_year'] == year])
            percentage = (count / year_total * 100) if year_total > 0 else 0
            if count > 0:
                print(f"        {year}: {count} papers ({percentage:.1f}%)")
    
    return evolution_data

def analyze_data_completeness(df):
    """
    Analyze completeness of different data fields
    
    Args:
        df (pandas.DataFrame): Cleaned dataset
    
    Returns:
        dict: Data completeness analysis
    """
    print(f"\n📊 DATA COMPLETENESS ANALYSIS:")
    print("-" * 40)
    
    important_fields = ['title', 'abstract', 'journal', 'authors', 'publish_time', 'doi']
    completeness_report = {}
    
    print(f"   📈 Field completeness rates:")
    for field in important_fields:
        if field in df.columns:
            non_null_count = df[field].notna().sum()
            completeness_rate = (non_null_count / len(df)) * 100
            completeness_report[field] = completeness_rate
            
            status = "🟢" if completeness_rate >= 90 else "🟡" if completeness_rate >= 70 else "🔴"
            print(f"      {status} {field}: {completeness_rate:.1f}% ({non_null_count:,}/{len(df):,})")
        else:
            print(f"      ❌ {field}: Column not found")
            completeness_report[field] = 0
    
    # Overall data quality score
    avg_completeness = np.mean(list(completeness_report.values()))
    quality_grade = "A" if avg_completeness >= 90 else "B" if avg_completeness >= 80 else "C" if avg_completeness >= 70 else "D"
    
    print(f"\n   🎯 Overall data quality: {avg_completeness:.1f}% (Grade: {quality_grade})")
    
    return completeness_report

def generate_analysis_summary(df, year_counts, journal_counts, top_words):
    """
    Generate a comprehensive analysis summary
    
    Args:
        df (pandas.DataFrame): Cleaned dataset
        year_counts (pandas.Series): Publications by year
        journal_counts (pandas.Series): Publications by journal
        top_words (list): Most frequent words
    
    Returns:
        dict: Complete analysis summary
    """
    print(f"\n📋 ANALYSIS SUMMARY:")
    print("-" * 40)
    
    # Dataset overview
    print(f"   📊 Dataset Overview:")
    print(f"      Total papers analyzed: {len(df):,}")
    print(f"      Time period: {int(df['publication_year'].min())}-{int(df['publication_year'].max())}")
    print(f"      Unique journals: {df['journal'].nunique():,}")
    
    # Key findings
    if year_counts is not None:
        peak_year = int(year_counts.idxmax())
        peak_count = year_counts.max()
        print(f"      Peak publication year: {peak_year} ({peak_count:,} papers)")
    
    if journal_counts is not None:
        top_journal = journal_counts.index[0]
        top_journal_count = journal_counts.iloc[0]
        print(f"      Top journal: {top_journal[:40]}... ({top_journal_count:,} papers)")
    
    if top_words:
        most_common_word = top_words[0][0]
        word_frequency = top_words[0][1]
        print(f"      Most frequent title word: '{most_common_word}' ({word_frequency:,} times)")
    
    # Research insights
    print(f"\n   🔍 Key Research Insights:")
    covid_papers = df[df['title'].str.contains('covid|coronavirus', case=False, na=False)]
    covid_percentage = (len(covid_papers) / len(df)) * 100
    print(f"      Papers explicitly about COVID: {len(covid_papers):,} ({covid_percentage:.1f}%)")
    
    if 'has_abstract' in df.columns:
        with_abstract = df['has_abstract'].sum()
        abstract_rate = (with_abstract / len(df)) * 100
        print(f"      Papers with abstracts: {with_abstract:,} ({abstract_rate:.1f}%)")
    
    summary = {
        'total_papers': len(df),
        'time_period': f"{int(df['publication_year'].min())}-{int(df['publication_year'].max())}",
        'unique_journals': df['journal'].nunique(),
        'peak_year': int(year_counts.idxmax()) if year_counts is not None else None,
        'top_journal': journal_counts.index[0] if journal_counts is not None else None,
        'covid_papers_count': len(covid_papers),
        'covid_percentage': covid_percentage
    }
    
    return summary

def main(df_clean):
    """
    Main function to run complete data analysis
    
    Args:
        df_clean (pandas.DataFrame): Cleaned dataset
    
    Returns:
        dict: All analysis results
    """
    if df_clean is None or len(df_clean) == 0:
        print("❌ No cleaned data provided for analysis!")
        return None
    
    print(f"🚀 Starting analysis of {len(df_clean):,} papers...")
    
    # Run all analyses
    year_counts = analyze_publications_by_year(df_clean)
    journal_counts = analyze_top_journals(df_clean)
    top_words = analyze_title_words(df_clean)
    abstract_analysis = analyze_abstract_patterns(df_clean)
    evolution_analysis = analyze_research_evolution(df_clean)
    completeness_analysis = analyze_data_completeness(df_clean)
    
    # Generate summary
    summary = generate_analysis_summary(df_clean, year_counts, journal_counts, top_words)
    
    print(f"\n✅ DATA ANALYSIS COMPLETED!")
    print(f"✅ Ready for visualization!")
    
    # Compile all results
    analysis_results = {
        'summary': summary,
        'year_counts': year_counts,
        'journal_counts': journal_counts,
        'top_words': top_words,
        'abstract_analysis': abstract_analysis,
        'evolution_analysis': evolution_analysis,
        'completeness_analysis': completeness_analysis,
        'dataset_info': {
            'shape': df_clean.shape,
            'columns': list(df_clean.columns)
        }
    }
    
    return analysis_results

if __name__ == "__main__":
    # For testing independently
    try:
        df = pd.read_csv('data/metadata_clean.csv')
        results = main(df)
    except FileNotFoundError:
        print("❌ Cleaned data file not found!")
        print("   Run data_loader.py and data_cleaner.py first")
    except Exception as e:
        print(f"❌ Error in analysis: {e}")