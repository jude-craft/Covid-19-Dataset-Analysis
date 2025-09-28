# Part 3: Data Visualization Functions

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import os

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

def create_publications_timeline(year_counts, save_path='outputs/figures/publications_by_year.png'):
    """
    Create publication timeline visualization
    
    Args:
        year_counts (pandas.Series): Publications count by year
        save_path (str): Path to save the figure
    """
    print(f"\n📊 CREATING VISUALIZATIONS:")
    print("-" * 40)
    print(f"   📈 Creating publication timeline...")
    
    if year_counts is None or len(year_counts) == 0:
        print("   ⚠️  No year data available for visualization!")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Create bar chart
    bars = plt.bar(year_counts.index, year_counts.values, color='skyblue', alpha=0.8, edgecolor='navy')
    
    # Customize the plot
    plt.title('COVID-19 Research Publications by Year', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Publication Year', fontsize=12)
    plt.ylabel('Number of Publications', fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(year_counts.values) * 0.01,
                f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    # Format y-axis to show values in thousands
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K' if x >= 1000 else f'{int(x)}'))
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Saved timeline to: {save_path}")

def create_top_journals_chart(journal_counts, top_n=15, save_path='outputs/figures/top_journals.png'):
    """
    Create top journals horizontal bar chart
    
    Args:
        journal_counts (pandas.Series): Journal publication counts
        top_n (int): Number of top journals to show
        save_path (str): Path to save the figure
    """
    print(f"   📰 Creating top journals chart...")
    
    if journal_counts is None or len(journal_counts) == 0:
        print("   ⚠️  No journal data available for visualization!")
        return
    
    # Take top N journals
    top_journals = journal_counts.head(top_n)
    
    plt.figure(figsize=(14, 8))
    
    # Create horizontal bar chart
    bars = plt.barh(range(len(top_journals)), top_journals.values, color='lightcoral', alpha=0.8)
    
    # Customize the plot
    plt.title(f'Top {top_n} Journals Publishing COVID-19 Research', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of Publications', fontsize=12)
    plt.ylabel('Journal', fontsize=12)
    
    # Set y-axis labels (truncate long journal names)
    journal_labels = [journal[:60] + '...' if len(journal) > 60 else journal 
                     for journal in top_journals.index]
    plt.yticks(range(len(top_journals)), journal_labels)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + max(top_journals.values) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{int(width):,}', ha='left', va='center', fontweight='bold')
    
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Saved journals chart to: {save_path}")

def create_word_frequency_chart(top_words, top_n=20, save_path='outputs/figures/word_frequency.png'):
    """
    Create word frequency horizontal bar chart
    
    Args:
        top_words (list): List of (word, count) tuples
        top_n (int): Number of top words to show
        save_path (str): Path to save the figure
    """
    print(f"   🔤 Creating word frequency chart...")
    
    if not top_words or len(top_words) == 0:
        print("   ⚠️  No word data available for visualization!")
        return
    
    # Extract words and counts
    words = [item[0] for item in top_words[:top_n]]
    counts = [item[1] for item in top_words[:top_n]]
    
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar chart
    bars = plt.barh(range(len(words)), counts, color='lightgreen', alpha=0.8)
    
    # Customize the plot
    plt.title(f'Top {top_n} Most Frequent Words in Paper Titles', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Words', fontsize=12)
    
    # Set y-axis labels
    plt.yticks(range(len(words)), words)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + max(counts) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{int(width):,}', ha='left', va='center', fontweight='bold')
    
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Saved word frequency chart to: {save_path}")

def create_wordcloud(df, save_path='outputs/figures/wordcloud.png'):
    """
    Create word cloud from paper titles
    
    Args:
        df (pandas.DataFrame): Dataset with titles
        save_path (str): Path to save the figure
    """
    print(f"   ☁️  Creating word cloud...")
    
    if 'title' not in df.columns:
        print("   ⚠️  No title column found for word cloud!")
        return
    
    try:
        # Combine all titles
        text = ' '.join(df['title'].fillna('').astype(str))
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            random_state=42
        ).generate(text)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Paper Titles', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Save the figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Saved word cloud to: {save_path}")
        
    except Exception as e:
        print(f"   ❌ Error creating word cloud: {e}")
        print("   💡 Try installing wordcloud: pip install wordcloud")

def create_data_completeness_heatmap(df, save_path='outputs/figures/data_completeness.png'):
    """
    Create heatmap showing data completeness
    
    Args:
        df (pandas.DataFrame): Dataset to analyze
        save_path (str): Path to save the figure
    """
    print(f"   🔥 Creating data completeness heatmap...")
    
    # Select important columns for analysis
    important_cols = ['title', 'abstract', 'journal', 'authors', 'publish_time', 'doi']
    available_cols = [col for col in important_cols if col in df.columns]
    
    if len(available_cols) == 0:
        print("   ⚠️  No important columns found for completeness analysis!")
        return
    
    # Calculate completeness rates
    completeness_data = []
    years = sorted(df['publication_year'].unique()) if 'publication_year' in df.columns else [2020]
    
    for year in years:
        if 'publication_year' in df.columns:
            year_data = df[df['publication_year'] == year]
        else:
            year_data = df
            
        year_completeness = []
        for col in available_cols:
            completeness_rate = (year_data[col].notna().sum() / len(year_data)) * 100
            year_completeness.append(completeness_rate)
        
        completeness_data.append(year_completeness)
    
    # Create heatmap
    plt.figure(figsize=(10, 6))
    
    completeness_df = pd.DataFrame(completeness_data, 
                                  index=[str(int(year)) for year in years], 
                                  columns=available_cols)
    
    sns.heatmap(completeness_df, annot=True, fmt='.1f', cmap='RdYlGn', 
                vmin=0, vmax=100, cbar_kws={'label': 'Completeness %'})
    
    plt.title('Data Completeness by Year and Field', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Data Fields', fontsize=12)
    plt.ylabel('Publication Year', fontsize=12)
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Saved completeness heatmap to: {save_path}")

def create_publication_distribution_pie(df, save_path='outputs/figures/publication_distribution.png'):
    """
    Create pie chart showing distribution by source or journal type
    
    Args:
        df (pandas.DataFrame): Dataset to analyze
        save_path (str): Path to save the figure
    """
    print(f"   🥧 Creating publication distribution chart...")
    
    # Try to find source column or use journal
    source_col = None
    for col in ['source_x', 'source', 'journal']:
        if col in df.columns:
            source_col = col
            break
    
    if source_col is None:
        print("   ⚠️  No source/journal column found for distribution!")
        return
    
    # Get top sources/journals and group others
    source_counts = df[source_col].value_counts().head(8)
    others_count = df[source_col].value_counts().iloc[8:].sum() if len(df[source_col].value_counts()) > 8 else 0
    
    if others_count > 0:
        source_counts = pd.concat([source_counts, pd.Series({'Others': others_count})])
    
    plt.figure(figsize=(10, 8))
    
    # Create pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(source_counts)))
    wedges, texts, autotexts = plt.pie(source_counts.values, 
                                      labels=[label[:20] + '...' if len(label) > 20 else label 
                                             for label in source_counts.index],
                                      autopct='%1.1f%%', 
                                      startangle=90,
                                      colors=colors)
    
    # Beautify the chart
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.title(f'Distribution of Papers by {source_col.replace("_", " ").title()}', 
              fontsize=16, fontweight='bold', pad=20)
    plt.axis('equal')
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Saved distribution chart to: {save_path}")

def create_abstract_length_distribution(df, save_path='outputs/figures/abstract_length_distribution.png'):
    """
    Create histogram of abstract length distribution
    
    Args:
        df (pandas.DataFrame): Dataset with abstract_word_count
        save_path (str): Path to save the figure
    """
    print(f"   📊 Creating abstract length distribution...")
    
    if 'abstract_word_count' not in df.columns:
        print("   ⚠️  No abstract_word_count column found!")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Filter out zero-length abstracts for better visualization
    abstract_lengths = df[df['abstract_word_count'] > 0]['abstract_word_count']
    
    # Create histogram
    plt.hist(abstract_lengths, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    plt.axvline(abstract_lengths.mean(), color='red', linestyle='--', 
                label=f'Mean: {abstract_lengths.mean():.1f} words')
    plt.axvline(abstract_lengths.median(), color='orange', linestyle='--', 
                label=f'Median: {abstract_lengths.median():.1f} words')
    
    plt.title('Distribution of Abstract Lengths', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Words', fontsize=12)
    plt.ylabel('Number of Papers', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Saved abstract length distribution to: {save_path}")

def create_summary_dashboard(analysis_results, df):
    """
    Create a summary dashboard with multiple subplots
    
    Args:
        analysis_results (dict): Results from analyzer.py
        df (pandas.DataFrame): Clean dataset
    """
    print(f"   📊 Creating summary dashboard...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Publications by year (top-left)
    if analysis_results['year_counts'] is not None:
        year_counts = analysis_results['year_counts']
        ax1.bar(year_counts.index, year_counts.values, color='skyblue', alpha=0.8)
        ax1.set_title('Publications by Year', fontweight='bold')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Number of Papers')
        ax1.grid(axis='y', alpha=0.3)
    
    # 2. Top 10 journals (top-right)
    if analysis_results['journal_counts'] is not None:
        top_journals = analysis_results['journal_counts'].head(10)
        ax2.barh(range(len(top_journals)), top_journals.values, color='lightcoral')
        ax2.set_yticks(range(len(top_journals)))
        ax2.set_yticklabels([j[:25] + '...' if len(j) > 25 else j for j in top_journals.index])
        ax2.set_title('Top 10 Journals', fontweight='bold')
        ax2.set_xlabel('Number of Papers')
    
    # 3. Top 15 words (bottom-left)
    if analysis_results['top_words']:
        top_15_words = analysis_results['top_words'][:15]
        words = [item[0] for item in top_15_words]
        counts = [item[1] for item in top_15_words]
        ax3.barh(range(len(words)), counts, color='lightgreen')
        ax3.set_yticks(range(len(words)))
        ax3.set_yticklabels(words)
        ax3.set_title('Top 15 Title Words', fontweight='bold')
        ax3.set_xlabel('Frequency')
    
    # 4. Abstract length distribution (bottom-right)
    if 'abstract_word_count' in df.columns:
        abstract_lengths = df[df['abstract_word_count'] > 0]['abstract_word_count']
        ax4.hist(abstract_lengths, bins=30, color='gold', alpha=0.7)
        ax4.axvline(abstract_lengths.mean(), color='red', linestyle='--', label='Mean')
        ax4.set_title('Abstract Length Distribution', fontweight='bold')
        ax4.set_xlabel('Word Count')
        ax4.set_ylabel('Number of Papers')
        ax4.legend()
    
    plt.suptitle('COVID-19 Research Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save dashboard
    save_path = 'outputs/figures/analysis_dashboard.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Saved dashboard to: {save_path}")

def main(analysis_results, df_clean):
    """
    Main function to create all visualizations
    
    Args:
        analysis_results (dict): Results from analyzer.py
        df_clean (pandas.DataFrame): Cleaned dataset
    """
    if analysis_results is None:
        print("❌ No analysis results provided for visualization!")
        return
    
    print(f"🎨 Creating visualizations...")
    
    # Create individual visualizations
    create_publications_timeline(analysis_results['year_counts'])
    create_top_journals_chart(analysis_results['journal_counts'])
    create_word_frequency_chart(analysis_results['top_words'])
    create_wordcloud(df_clean)
    create_data_completeness_heatmap(df_clean)
    create_publication_distribution_pie(df_clean)
    create_abstract_length_distribution(df_clean)
    
    # Create summary dashboard
    create_summary_dashboard(analysis_results, df_clean)
    
    print(f"\n✅ ALL VISUALIZATIONS COMPLETED!")
    print(f"✅ Check the 'outputs/figures/' directory for saved plots")
    
    return True

if __name__ == "__main__":
    # For testing independently
    try:
        import sys
        sys.path.append('src')
        from analyzer import main as analyze_data
        
        df = pd.read_csv('data/metadata_clean.csv')
        results = analyze_data(df)
        
        if results:
            main(results, df)
        else:
            print("❌ No analysis results to visualize!")
            
    except FileNotFoundError:
        print("❌ Cleaned data file not found!")
        print("   Run data_loader.py and data_cleaner.py first")
    except Exception as e:
        print(f"❌ Error in visualization: {e}")