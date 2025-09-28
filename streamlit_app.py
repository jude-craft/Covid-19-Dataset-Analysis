# Part 4: Interactive Streamlit Dashboard

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import numpy as np
from collections import Counter
import re
import os

# Page configuration
st.set_page_config(
    page_title="🦠 CORD-19 Data Explorer",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
.stAlert {
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the cleaned dataset with caching for better performance"""
    try:
        # Try to load cleaned data first
        if os.path.exists('data/metadata_clean.csv'):
            df = pd.read_csv('data/metadata_clean.csv')
            return df, None
        else:
            error_msg = "❌ Cleaned data file not found! Please run the analysis pipeline first."
            return None, error_msg
    except Exception as e:
        error_msg = f"❌ Error loading data: {str(e)}"
        return None, error_msg

def create_year_analysis(df_filtered):
    """Create year-based analysis visualizations"""
    if 'publication_year' not in df_filtered.columns:
        st.warning("⚠️ Publication year data not available")
        return
    
    # Publications by year
    year_counts = df_filtered['publication_year'].value_counts().sort_index()
    
    # Create interactive plot with Plotly
    fig = px.bar(
        x=year_counts.index,
        y=year_counts.values,
        labels={'x': 'Publication Year', 'y': 'Number of Publications'},
        title="📈 COVID-19 Research Publications Over Time",
        color=year_counts.values,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        title_font_size=18,
        title_x=0.5,
        showlegend=False,
        height=500
    )
    
    # Add hover information
    fig.update_traces(
        hovertemplate='<b>Year:</b> %{x}<br><b>Publications:</b> %{y:,}<extra></extra>'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Year-over-year growth analysis
    if len(year_counts) > 1:
        st.subheader("📊 Year-over-Year Growth")
        growth_data = []
        for i in range(1, len(year_counts)):
            prev_count = year_counts.iloc[i-1]
            curr_count = year_counts.iloc[i]
            growth_rate = ((curr_count - prev_count) / prev_count * 100) if prev_count > 0 else 0
            growth_data.append({
                'Year': int(year_counts.index[i]),
                'Publications': curr_count,
                'Growth Rate (%)': growth_rate,
                'Change': curr_count - prev_count
            })
        
        growth_df = pd.DataFrame(growth_data)
        st.dataframe(growth_df, use_container_width=True)

def create_journal_analysis(df_filtered):
    """Create journal analysis visualizations"""
    if 'journal' not in df_filtered.columns:
        st.warning("⚠️ Journal data not available")
        return
    
    # Top journals
    top_n = st.slider("Number of top journals to show", 5, 25, 15)
    journal_counts = df_filtered['journal'].value_counts().head(top_n)
    
    # Create horizontal bar chart
    fig = px.bar(
        x=journal_counts.values,
        y=journal_counts.index,
        orientation='h',
        labels={'x': 'Number of Publications', 'y': 'Journal'},
        title=f"📰 Top {top_n} Journals Publishing COVID-19 Research",
        color=journal_counts.values,
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        title_font_size=18,
        title_x=0.5,
        height=max(400, top_n * 25),
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    
    fig.update_traces(
        hovertemplate='<b>Journal:</b> %{y}<br><b>Publications:</b> %{x:,}<extra></extra>'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Journal statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Unique Journals", df_filtered['journal'].nunique())
    with col2:
        top_journal_name = journal_counts.index[0][:30] + "..." if len(journal_counts.index[0]) > 30 else journal_counts.index[0]
        st.metric("Top Journal", top_journal_name)
    with col3:
        top_5_percentage = (journal_counts.head(5).sum() / len(df_filtered)) * 100
        st.metric("Top 5 Journals Share", f"{top_5_percentage:.1f}%")

def create_word_analysis(df_filtered):
    """Create word frequency analysis"""
    if 'title' not in df_filtered.columns:
        st.warning("⚠️ Title data not available for word analysis")
        return
    
    # Word frequency analysis
    st.subheader("🔤 Most Frequent Words in Titles")
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        top_n_words = st.slider("Number of top words to show", 10, 50, 20)
    with col2:
        min_word_length = st.slider("Minimum word length", 3, 8, 4)
    
    # Process text
    all_titles = ' '.join(df_filtered['title'].fillna('').astype(str))
    words = re.findall(rf'\b[a-zA-Z]{{{min_word_length},}}\b', all_titles.lower())
    
    # Remove stop words
    stop_words = {
        'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 'they', 'been',
        'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might',
        'can', 'did', 'does', 'was', 'were', 'but', 'not', 'you', 'all', 'any',
        'her', 'his', 'our', 'out', 'day', 'get', 'use', 'new', 'now', 'old',
        'see', 'him', 'two', 'way', 'who', 'its', 'said', 'each', 'she', 'which',
        'their', 'time', 'than', 'only', 'come', 'over', 'also', 'back', 'after',
        'first', 'well', 'year', 'work', 'such', 'make', 'even', 'most', 'take'
    }
    
    filtered_words = [word for word in words if word not in stop_words]
    word_counts = Counter(filtered_words)
    top_words_data = word_counts.most_common(top_n_words)
    
    # Create visualization
    if top_words_data:
        words_list = [item[0] for item in top_words_data]
        counts_list = [item[1] for item in top_words_data]
        
        fig = px.bar(
            x=counts_list,
            y=words_list,
            orientation='h',
            labels={'x': 'Frequency', 'y': 'Words'},
            title=f"Top {top_n_words} Most Frequent Words (min length: {min_word_length})",
            color=counts_list,
            color_continuous_scale='Greens'
        )
        
        fig.update_layout(
            title_font_size=16,
            title_x=0.5,
            height=max(400, top_n_words * 20),
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # COVID-related terms
        covid_terms = ['covid', 'coronavirus', 'sars', 'pandemic', 'vaccine', 'virus', 'infection']
        covid_counts = {}
        for term in covid_terms:
            count = sum(1 for word in filtered_words if term in word.lower())
            if count > 0:
                covid_counts[term] = count
        
        if covid_counts:
            st.subheader("🦠 COVID-Related Terms Frequency")
            covid_df = pd.DataFrame(list(covid_counts.items()), columns=['Term', 'Frequency'])
            covid_df = covid_df.sort_values('Frequency', ascending=False)
            st.bar_chart(covid_df.set_index('Term'))

def create_wordcloud_visualization(df_filtered):
    """Create word cloud visualization"""
    if 'title' not in df_filtered.columns:
        return
    
    try:
        # Combine all titles
        text = ' '.join(df_filtered['title'].fillna('').astype(str))
        
        if len(text.strip()) > 0:
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
            
            # Display using matplotlib
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Word Cloud of Paper Titles', fontsize=16, fontweight='bold')
            
            st.pyplot(fig)
        else:
            st.warning("⚠️ No text data available for word cloud")
            
    except Exception as e:
        st.error(f"❌ Error creating word cloud: {str(e)}")
        st.info("💡 Word cloud requires the 'wordcloud' package. Install with: pip install wordcloud")

def create_data_overview(df_filtered):
    """Create data overview and statistics"""
    st.subheader("📊 Dataset Overview")
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📄 Total Papers",
            value=f"{len(df_filtered):,}",
            delta=None
        )
    
    with col2:
        if 'journal' in df_filtered.columns:
            unique_journals = df_filtered['journal'].nunique()
            st.metric(
                label="📰 Unique Journals",
                value=f"{unique_journals:,}",
                delta=None
            )
    
    with col3:
        if 'publication_year' in df_filtered.columns:
            year_range = f"{df_filtered['publication_year'].min():.0f}-{df_filtered['publication_year'].max():.0f}"
            st.metric(
                label="📅 Year Range",
                value=year_range,
                delta=None
            )
    
    with col4:
        if 'has_abstract' in df_filtered.columns:
            with_abstract = df_filtered['has_abstract'].sum()
            st.metric(
                label="📝 Papers with Abstract",
                value=f"{with_abstract:,}",
                delta=f"{(with_abstract/len(df_filtered)*100):.1f}%"
            )
    
    # Additional statistics
    if 'abstract_word_count' in df_filtered.columns:
        st.subheader("📈 Abstract Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_words = df_filtered['abstract_word_count'].mean()
            st.metric("Average Abstract Length", f"{avg_words:.1f} words")
        
        with col2:
            median_words = df_filtered['abstract_word_count'].median()
            st.metric("Median Abstract Length", f"{median_words:.1f} words")
        
        with col3:
            max_words = df_filtered['abstract_word_count'].max()
            st.metric("Longest Abstract", f"{max_words:,} words")
        
        # Abstract length distribution
        st.subheader("📊 Abstract Length Distribution")
        
        # Filter out zero-length abstracts
        abstract_lengths = df_filtered[df_filtered['abstract_word_count'] > 0]['abstract_word_count']
        
        if len(abstract_lengths) > 0:
            fig = px.histogram(
                abstract_lengths,
                nbins=50,
                title="Distribution of Abstract Lengths",
                labels={'value': 'Number of Words', 'count': 'Number of Papers'}
            )
            
            # Add mean and median lines
            fig.add_vline(
                x=abstract_lengths.mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {abstract_lengths.mean():.1f}"
            )
            
            fig.add_vline(
                x=abstract_lengths.median(),
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Median: {abstract_lengths.median():.1f}"
            )
            
            fig.update_layout(title_x=0.5, height=400)
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🦠 CORD-19 Data Explorer</h1>', unsafe_allow_html=True)
    st.markdown("### Interactive exploration of COVID-19 research papers")
    
    # Load data
    with st.spinner("Loading data..."):
        df, error = load_data()
    
    if error:
        st.error(error)
        st.info("Please run the complete analysis pipeline first:")
        st.code("""
        # Run these files in order:
        python src/data_loader.py
        python src/data_cleaner.py
        python src/analyzer.py
        python src/visualizer.py
        """)
        return
    
    # Sidebar controls
    st.sidebar.header("🎛️ Filter Options")
    st.sidebar.markdown("---")
    
    # Year filter
    if 'publication_year' in df.columns:
        min_year = int(df['publication_year'].min())
        max_year = int(df['publication_year'].max())
        
        year_range = st.sidebar.slider(
            "📅 Select Year Range",
            min_year, max_year, (min_year, max_year),
            help="Filter papers by publication year"
        )
        
        # Apply year filter
        df_filtered = df[
            (df['publication_year'] >= year_range[0]) & 
            (df['publication_year'] <= year_range[1])
        ]
    else:
        df_filtered = df.copy()
        year_range = None
    
    # Journal filter
    if 'journal' in df.columns:
        st.sidebar.markdown("### 📰 Journal Filter")
        top_journals = df['journal'].value_counts().head(20).index.tolist()
        
        selected_journals = st.sidebar.multiselect(
            "Select Journals (leave empty for all)",
            options=top_journals,
            default=[],
            help="Filter by specific journals"
        )
        
        if selected_journals:
            df_filtered = df_filtered[df_filtered['journal'].isin(selected_journals)]
    
    # Search filter
    st.sidebar.markdown("### 🔍 Keyword Search")
    search_term = st.sidebar.text_input(
        "Search in titles",
        placeholder="e.g., vaccine, treatment, diagnosis",
        help="Search for papers containing specific keywords in titles"
    )
    
    if search_term:
        df_filtered = df_filtered[
            df_filtered['title'].str.contains(search_term, case=False, na=False)
        ]
    
    # Show filtered results info
    if len(df_filtered) != len(df):
        st.info(f"📊 Showing {len(df_filtered):,} papers out of {len(df):,} total papers")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "📈 Time Analysis", 
        "📰 Journals", 
        "🔤 Word Analysis", 
        "💾 Data Export"
    ])
    
    with tab1:
        create_data_overview(df_filtered)
    
    with tab2:
        st.header("📈 Publication Timeline Analysis")
        create_year_analysis(df_filtered)
    
    with tab3:
        st.header("📰 Journal Analysis")
        create_journal_analysis(df_filtered)
    
    with tab4:
        st.header("🔤 Text and Word Analysis")
        create_word_analysis(df_filtered)
        
        st.markdown("---")
        st.subheader("☁️ Word Cloud")
        create_wordcloud_visualization(df_filtered)
    
    with tab5:
        st.header("💾 Data Export and Sample")
        
        # Show sample data
        st.subheader("📋 Sample Data")
        
        # Column selection
        if not df_filtered.empty:
            available_columns = df_filtered.columns.tolist()
            default_columns = ['title', 'journal', 'publication_year', 'authors'][:4]
            default_columns = [col for col in default_columns if col in available_columns]
            
            display_columns = st.multiselect(
                "Select columns to display:",
                options=available_columns,
                default=default_columns if default_columns else available_columns[:4]
            )
            
            if display_columns:
                sample_size = st.slider("Number of rows to display", 10, 100, 50)
                st.dataframe(df_filtered[display_columns].head(sample_size), use_container_width=True)
                
                # Download button
                if st.button("📥 Prepare Download"):
                    csv_data = df_filtered.to_csv(index=False)
                    
                    filename = f"covid_papers_filtered"
                    if year_range:
                        filename += f"_{year_range[0]}-{year_range[1]}"
                    if selected_journals:
                        filename += f"_{len(selected_journals)}_journals"
                    if search_term:
                        filename += f"_{search_term.replace(' ', '_')}"
                    filename += ".csv"
                    
                    st.download_button(
                        label="📥 Download Filtered Data as CSV",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv",
                        help="Download the currently filtered dataset"
                    )
        else:
            st.warning("⚠️ No data available with current filters")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>📊 COVID-19 Research Analysis Dashboard | Built with Streamlit & Python</p>
        <p>Data source: CORD-19 Dataset | Last updated: 2024</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()