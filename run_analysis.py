import os
import sys
from datetime import datetime

# Add src directory to Python path
sys.path.append('src')

def run_complete_pipeline():
    """
    Run the complete analysis pipeline in the correct order
    """
    print("🚀" + "=" * 60)
    print("🚀 COVID-19 RESEARCH ANALYSIS PIPELINE")
    print("🚀" + "=" * 60)
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Data Loading
        print("\n" + "🔄 STEP 1: DATA LOADING")
        print("-" * 40)
        from data_loader import main as load_data
        
        df_raw, summary, quality = load_data()
        
        if df_raw is None:
            print("❌ Data loading failed! Please check your metadata.csv file.")
            print("💡 Make sure metadata.csv is in the data/ directory")
            return False
        
        print(f"✅ Successfully loaded {len(df_raw):,} records")
        
        # Step 2: Data Cleaning
        print("\n" + "🔄 STEP 2: DATA CLEANING")
        print("-" * 40)
        from data_cleaner import main as clean_data
        
        df_clean = clean_data(df_raw)
        
        if df_clean is None or len(df_clean) == 0:
            print("❌ Data cleaning failed!")
            return False
        
        print(f"✅ Successfully cleaned data: {len(df_clean):,} records")
        
        # Step 3: Data Analysis
        print("\n" + "🔄 STEP 3: DATA ANALYSIS")
        print("-" * 40)
        from analyzer import main as analyze_data
        
        analysis_results = analyze_data(df_clean)
        
        if analysis_results is None:
            print("❌ Data analysis failed!")
            return False
        
        print(f"✅ Successfully completed analysis")
        
        # Step 4: Visualization
        print("\n" + "🔄 STEP 4: CREATING VISUALIZATIONS")
        print("-" * 40)
        from visualizer import main as create_visualizations
        
        viz_success = create_visualizations(analysis_results, df_clean)
        
        if not viz_success:
            print("⚠️  Visualization step had issues, but continuing...")
        else:
            print(f"✅ Successfully created all visualizations")
        
        # Step 5: Generate Final Report
        print("\n" + "🔄 STEP 5: GENERATING FINAL REPORT")
        print("-" * 40)
        
        generate_final_report(summary, analysis_results, df_raw, df_clean)
        
        # Success summary
        print("\n" + "🎉" + "=" * 60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("🎉" + "=" * 60)
        
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"   📄 Original papers: {len(df_raw):,}")
        print(f"   🧹 Cleaned papers: {len(df_clean):,}")
        print(f"   📈 Data retention: {(len(df_clean)/len(df_raw)*100):.1f}%")
        
        if analysis_results and analysis_results['year_counts'] is not None:
            peak_year = int(analysis_results['year_counts'].idxmax())
            peak_count = analysis_results['year_counts'].max()
            print(f"   🔥 Peak year: {peak_year} ({peak_count:,} papers)")
        
        if analysis_results and analysis_results['journal_counts'] is not None:
            top_journal = analysis_results['journal_counts'].index[0]
            print(f"   📰 Top journal: {top_journal[:50]}...")
        
        print(f"\n📁 OUTPUT FILES CREATED:")
        print(f"   📄 data/metadata_clean.csv")
        print(f"   📊 outputs/figures/ (all visualization files)")
        print(f"   📋 outputs/analysis_report.txt")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Review generated visualizations in outputs/figures/")
        print(f"   2. Read the analysis report: outputs/analysis_report.txt")
        print(f"   3. Run Streamlit dashboard: streamlit run streamlit_app.py")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all required packages are installed:")
        print("   pip install pandas matplotlib seaborn wordcloud plotly streamlit")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_final_report(summary, analysis_results, df_raw, df_clean):
    """
    Generate a comprehensive final report
    """
    report_path = "outputs/analysis_report.txt"
    
    # Ensure output directory exists
    os.makedirs("outputs", exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("COVID-19 RESEARCH PAPERS ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"This report presents an analysis of {len(df_clean):,} COVID-19 research papers\n")
        f.write(f"from the CORD-19 dataset. The analysis covers publication trends, journal\n")
        f.write(f"distribution, and keyword frequency from research conducted between 2019-2024.\n\n")
        
        # Dataset Overview
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write(f"Original records: {len(df_raw):,}\n")
        f.write(f"Records after cleaning: {len(df_clean):,}\n")
        f.write(f"Data retention rate: {(len(df_clean)/len(df_raw)*100):.1f}%\n")
        f.write(f"Columns in final dataset: {len(df_clean.columns)}\n")
        f.write(f"Time period covered: {df_clean['publication_year'].min():.0f}-{df_clean['publication_year'].max():.0f}\n\n")
        
        # Key Findings
        if analysis_results:
            f.write("KEY FINDINGS\n")
            f.write("-" * 40 + "\n")
            
            # Publication trends
            if analysis_results['year_counts'] is not None:
                year_counts = analysis_results['year_counts']
                peak_year = int(year_counts.idxmax())
                peak_count = year_counts.max()
                f.write(f"1. PUBLICATION TRENDS:\n")
                f.write(f"   - Peak publication year: {peak_year} ({peak_count:,} papers)\n")
                f.write(f"   - Year-by-year breakdown:\n")
                for year, count in year_counts.items():
                    f.write(f"     {int(year)}: {count:,} papers\n")
                f.write("\n")
            
            # Journal analysis
            if analysis_results['journal_counts'] is not None:
                journal_counts = analysis_results['journal_counts']
                f.write(f"2. JOURNAL ANALYSIS:\n")
                f.write(f"   - Unique journals: {df_clean['journal'].nunique():,}\n")
                f.write(f"   - Top 10 publishing journals:\n")
                for i, (journal, count) in enumerate(journal_counts.head(10).items(), 1):
                    percentage = (count / len(df_clean)) * 100
                    f.write(f"     {i:2d}. {journal}: {count:,} papers ({percentage:.1f}%)\n")
                f.write("\n")
            
            # Word analysis
            if analysis_results['top_words']:
                f.write(f"3. KEYWORD ANALYSIS:\n")
                f.write(f"   - Most frequent words in titles:\n")
                for i, (word, count) in enumerate(analysis_results['top_words'][:15], 1):
                    percentage = (count / len(df_clean)) * 100
                    f.write(f"     {i:2d}. {word}: {count:,} times ({percentage:.1f}% of papers)\n")
                f.write("\n")
            
            # Data quality
            if analysis_results['completeness_analysis']:
                f.write(f"4. DATA QUALITY ASSESSMENT:\n")
                f.write(f"   - Field completeness rates:\n")
                for field, completeness in analysis_results['completeness_analysis'].items():
                    f.write(f"     {field}: {completeness:.1f}%\n")
                f.write("\n")
        
        # Research Insights
        f.write("RESEARCH INSIGHTS\n")
        f.write("-" * 40 + "\n")
        
        # COVID-specific papers
        covid_papers = df_clean[df_clean['title'].str.contains('covid|coronavirus', case=False, na=False)]
        covid_percentage = (len(covid_papers) / len(df_clean)) * 100
        f.write(f"- Papers explicitly mentioning COVID/Coronavirus: {len(covid_papers):,} ({covid_percentage:.1f}%)\n")
        
        # Abstract availability
        if 'has_abstract' in df_clean.columns:
            with_abstract = df_clean['has_abstract'].sum()
            abstract_percentage = (with_abstract / len(df_clean)) * 100
            f.write(f"- Papers with abstracts: {with_abstract:,} ({abstract_percentage:.1f}%)\n")
        
        # Average lengths
        if 'abstract_word_count' in df_clean.columns:
            avg_abstract = df_clean['abstract_word_count'].mean()
            f.write(f"- Average abstract length: {avg_abstract:.1f} words\n")
        
        if 'title_word_count' in df_clean.columns:
            avg_title = df_clean['title_word_count'].mean()
            f.write(f"- Average title length: {avg_title:.1f} words\n")
        
        f.write("\n")
        
        # Methodology
        f.write("METHODOLOGY\n")
        f.write("-" * 40 + "\n")
        f.write("1. Data Loading: Extracted metadata.csv from CORD-19 dataset\n")
        f.write("2. Data Cleaning: Removed duplicates, handled missing values, filtered invalid dates\n")
        f.write("3. Analysis: Computed publication trends, journal rankings, word frequencies\n")
        f.write("4. Visualization: Created charts for trends, distributions, and word clouds\n")
        f.write("5. Dashboard: Built interactive Streamlit application for exploration\n\n")
        
        # Files Generated
        f.write("FILES GENERATED\n")
        f.write("-" * 40 + "\n")
        f.write("- data/metadata_clean.csv: Cleaned dataset\n")
        f.write("- outputs/figures/publications_by_year.png: Publication timeline\n")
        f.write("- outputs/figures/top_journals.png: Top journals chart\n")
        f.write("- outputs/figures/word_frequency.png: Word frequency analysis\n")
        f.write("- outputs/figures/wordcloud.png: Title word cloud\n")
        f.write("- outputs/figures/analysis_dashboard.png: Summary dashboard\n")
        f.write("- streamlit_app.py: Interactive web application\n")
        f.write("- outputs/analysis_report.txt: This report\n\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS FOR FURTHER ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write("1. Conduct sentiment analysis on abstracts to gauge research outlook\n")
        f.write("2. Analyze collaboration patterns between authors and institutions\n")
        f.write("3. Perform topic modeling to identify research themes\n")
        f.write("4. Study citation patterns and impact factors\n")
        f.write("5. Analyze geographic distribution of research\n")
        f.write("6. Compare pre-pandemic vs pandemic research trends\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"✅ Final report saved to: {report_path}")

def check_requirements():
    """
    Check if all required packages are installed
    """
    print("🔍 Checking requirements...")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'wordcloud', 'plotly', 'streamlit'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print(f"💡 Install with: pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed!")
    return True

def check_data_file():
    """
    Check if data file exists
    """
    print("🔍 Checking data file...")
    
    data_paths = [
        'data/metadata.csv',
        'metadata.csv',
        'data/archive.csv'
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            file_size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"✅ Found data file: {path} ({file_size:.2f} MB)")
            return path
    
    print("❌ Data file not found!")
    print("💡 Please ensure your metadata.csv is in the 'data/' directory")
    print("   Expected location: data/metadata.csv")
    return None

def setup_directories():
    """
    Create necessary directories
    """
    print("🔍 Setting up directories...")
    
    directories = [
        'data',
        'outputs',
        'outputs/figures',
        'outputs/reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ {directory}/")
    
    print("✅ Directory structure ready!")

def main():
    """
    Main function to run everything
    """
    print("🚀 COVID-19 Research Analysis Pipeline")
    print("🚀 Starting pre-flight checks...\n")
    
    # Pre-flight checks
    if not check_requirements():
        print("\n❌ Requirements check failed!")
        print("Please install missing packages and try again.")
        return
    
    data_file = check_data_file()
    if not data_file:
        print("\n❌ Data file check failed!")
        print("Please place your archive.zip in the data/ directory.")
        return
    
    setup_directories()
    
    # Ask user if they want to proceed
    print(f"\n🚀 Ready to start analysis!")
    print(f"📁 Data file: {data_file}")
    print(f"⏱️  Estimated time: 5-10 minutes")
    
    response = input("\n❓ Proceed with analysis? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        success = run_complete_pipeline()
        
        if success:
            print(f"\n🎉 Analysis completed successfully!")
            print(f"\n🚀 Next steps:")
            print(f"   1. Review the report: outputs/analysis_report.txt")
            print(f"   2. Check visualizations: outputs/figures/")
            print(f"   3. Launch dashboard: streamlit run streamlit_app.py")
            
            # Ask if user wants to launch Streamlit
            launch = input(f"\n❓ Launch Streamlit dashboard now? (y/N): ").strip().lower()
            if launch in ['y', 'yes']:
                print(f"\n🚀 Launching Streamlit dashboard...")
                os.system("streamlit run streamlit_app.py")
        else:
            print(f"\n❌ Analysis pipeline failed!")
            print(f"Please check the error messages above and try again.")
    else:
        print(f"\n⏹️  Analysis cancelled by user.")

if __name__ == "__main__":
    main()