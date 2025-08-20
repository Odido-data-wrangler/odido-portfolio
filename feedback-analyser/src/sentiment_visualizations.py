import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Mapping between Dutch and English sentiment labels for consistency
SENTIMENT_MAPPING = {
    # Dutch to English
    'positief': 'positive',
    'neutraal': 'neutral',
    'negatief': 'negative',
    'gemengd': 'mixed',
    # English to Dutch
    'positive': 'positief',
    'neutral': 'neutraal',
    'negative': 'negatief',
    'mixed': 'gemengd'
}

# Dutch UI text for visualizations
DUTCH_LABELS = {
    'dashboard_title': 'LLM vs. BERT Sentimentanalyse Vergelijking',
    'sentiment_distribution': 'LLM vs. BERT Sentimentverdeling',
    'sentiment_agreement': 'Sentimentovereenstemming per Onderwerp',
    'score_correlation': 'Sentimentscore Correlatie',
    'confidence_comparison': 'Vergelijking van Betrouwbaarheidsscores',
    'sentiment': 'Sentiment',
    'count': 'Aantal',
    'agreement': 'Overeenkomst (%)',
    'topic': 'Onderwerp',
    'llm_score': 'LLM Score',
    'bert_score': 'BERT Score',
    'text': 'Tekst',
    'perfect_correlation': 'Perfecte correlatie',
    'correlation': 'Correlatie',
    'model_sentiment': 'Model & Sentiment',
    'confidence': 'Betrouwbaarheidsscore',
    'confusion_title': 'LLM vs. BERT Sentiment Confusion Matrix',
    'bert_prediction': 'BERT Sentiment Voorspelling',
    'llm_assignment': 'LLM Sentiment Toekenning',
    'percentage': 'Percentage (%)',
    'topic_sentiment_title': 'LLM vs. BERT Sentimentverdeling per Onderwerp'
}

def normalize_sentiment_labels(df):
    """
    Ensure consistent sentiment labels by mapping Dutch/English.
    If a label isn't in our mapping, leave it as is.
    
    Args:
        df: DataFrame with sentiment columns
        
    Returns:
        DataFrame with normalized sentiment labels
    """
    df_copy = df.copy()
    
    # Normalize LLM sentiment if present
    if 'llm_sentiment' in df_copy.columns:
        df_copy['llm_sentiment'] = df_copy['llm_sentiment'].map(
            lambda x: SENTIMENT_MAPPING.get(x, x)
        )
    
    # Normalize BERT sentiment if present
    if 'bert_sentiment' in df_copy.columns:
        df_copy['bert_sentiment'] = df_copy['bert_sentiment'].map(
            lambda x: SENTIMENT_MAPPING.get(x, x)
        )
    
    return df_copy

def create_sentiment_comparison_dashboard(df, output_file="sentiment_comparison.html", use_dutch=True):
    """
    Create a dashboard comparing LLM and BERT sentiment analysis.
    
    Args:
        df: DataFrame with both LLM and BERT sentiment
        output_file: Path to save the output HTML file
        use_dutch: Whether to use Dutch labels in the visualizations
    """
    # Normalize sentiment labels to ensure consistency
    df = normalize_sentiment_labels(df)
    
    # Select language labels
    if use_dutch:
        labels = DUTCH_LABELS
    else:
        labels = {
            'dashboard_title': 'LLM vs. BERT Sentiment Analysis Comparison',
            'sentiment_distribution': 'LLM vs. BERT Sentiment Distribution',
            'sentiment_agreement': 'Sentiment Agreement by Topic',
            'score_correlation': 'Sentiment Score Correlation',
            'confidence_comparison': 'Sentiment Confidence Comparison',
            'sentiment': 'Sentiment',
            'count': 'Count',
            'agreement': 'Agreement (%)',
            'topic': 'Topic',
            'llm_score': 'LLM Score',
            'bert_score': 'BERT Score',
            'text': 'Text',
            'perfect_correlation': 'Perfect correlation',
            'correlation': 'Correlation',
            'model_sentiment': 'Model & Sentiment',
            'confidence': 'Absolute Confidence Score',
            'confusion_title': 'LLM vs. BERT Sentiment Confusion Matrix',
            'bert_prediction': 'BERT Sentiment Prediction',
            'llm_assignment': 'LLM Sentiment Assignment',
            'percentage': 'Percentage (%)',
            'topic_sentiment_title': 'LLM vs. BERT Sentiment Distribution by Topic'
        }
    
    # Create subplots with 2 rows and 2 columns
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            labels['sentiment_distribution'], 
            labels['sentiment_agreement'],
            labels['score_correlation'], 
            labels['confidence_comparison']
        ],
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "box"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 1. Compare sentiment distributions (LLM vs BERT)
    llm_counts = df['llm_sentiment'].value_counts().sort_index()
    bert_counts = df['bert_sentiment'].value_counts().sort_index()
    
    # Ensure all categories are represented
    all_sentiments = sorted(set(list(llm_counts.index) + list(bert_counts.index)))
    llm_counts = llm_counts.reindex(all_sentiments, fill_value=0)
    bert_counts = bert_counts.reindex(all_sentiments, fill_value=0)
    
    fig.add_trace(
        go.Bar(
            x=all_sentiments,
            y=llm_counts.values,
            name="LLM Sentiment",
            marker_color='#636EFA'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=all_sentiments,
            y=bert_counts.values,
            name="BERT Sentiment",
            marker_color='#EF553B'
        ),
        row=1, col=1
    )
    
    # 2. Agreement by topic
    # Calculate agreement percentage by topic
    topic_agreement = df.groupby('topic_name').apply(
        lambda x: (x['llm_sentiment'] == x['bert_sentiment']).mean() * 100
    ).reset_index()
    topic_agreement.columns = ['topic_name', 'agreement_percent']
    
    # Sort by agreement
    topic_agreement = topic_agreement.sort_values('agreement_percent', ascending=True)
    
    # Keep only top 10 topics for readability
    if len(topic_agreement) > 10:
        topic_agreement = topic_agreement.tail(10)
    
    fig.add_trace(
        go.Bar(
            x=topic_agreement['agreement_percent'],
            y=topic_agreement['topic_name'],
            orientation='h',
            marker_color='#00CC96'
        ),
        row=1, col=2
    )
    
    # 3. Sentiment score correlation
    fig.add_trace(
        go.Scatter(
            x=df['llm_sentiment_score'],
            y=df['bert_sentiment_score'],
            mode='markers',
            opacity=0.7,
            marker=dict(
                size=8,
                color=df['topic_id'],
                colorscale='Viridis',
                showscale=False
            ),
            text=df['combined_text'].str[:50] + '...',
            hovertemplate=f'<b>{labels["llm_score"]}:</b> %{{x:.2f}}<br><b>{labels["bert_score"]}:</b> %{{y:.2f}}<br><b>{labels["text"]}:</b> %{{text}}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add correlation line
    corr = np.corrcoef(df['llm_sentiment_score'], df['bert_sentiment_score'])[0, 1]
    
    # Add diagonal reference line
    x_range = [df['llm_sentiment_score'].min(), df['llm_sentiment_score'].max()]
    y_range = [df['bert_sentiment_score'].min(), df['bert_sentiment_score'].max()]
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_range,
            mode='lines',
            line=dict(color='rgba(0,0,0,0.3)', dash='dash'),
            name=labels['perfect_correlation']
        ),
        row=2, col=1
    )
    
    # Add annotation with correlation coefficient
    fig.add_annotation(
        x=x_range[0] + (x_range[1] - x_range[0]) * 0.1,
        y=y_range[1] - (y_range[1] - y_range[0]) * 0.1,
        text=f"{labels['correlation']}: {corr:.2f}",
        showarrow=False,
        font=dict(size=14),
        row=2, col=1
    )
    
    # 4. Confidence comparison (boxplot)
    # Create a long-format DataFrame for boxplot
    scores_data = pd.DataFrame({
        'Source': ['LLM'] * len(df) + ['BERT'] * len(df),
        'Sentiment': list(df['llm_sentiment']) + list(df['bert_sentiment']),
        'Score': list(np.abs(df['llm_sentiment_score'])) + list(np.abs(df['bert_sentiment_score'])),
    })
    
    # Create box plot
    for sentiment in sorted(scores_data['Sentiment'].unique()):
        # Filter data for this sentiment
        sentiment_data = scores_data[scores_data['Sentiment'] == sentiment]
        
        # LLM scores
        llm_data = sentiment_data[sentiment_data['Source'] == 'LLM']
        bert_data = sentiment_data[sentiment_data['Source'] == 'BERT']
        
        # Add box plots side by side
        fig.add_trace(
            go.Box(
                y=llm_data['Score'],
                name=f"LLM - {sentiment}",
                boxmean=True,
                marker_color='#636EFA'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Box(
                y=bert_data['Score'],
                name=f"BERT - {sentiment}",
                boxmean=True,
                marker_color='#EF553B'
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title_text=labels['dashboard_title'],
        height=900,
        width=1200,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes
    fig.update_xaxes(title_text=labels['sentiment'], row=1, col=1)
    fig.update_yaxes(title_text=labels['count'], row=1, col=1)
    
    fig.update_xaxes(title_text=labels['agreement'], row=1, col=2)
    fig.update_yaxes(title_text=labels['topic'], row=1, col=2)
    
    fig.update_xaxes(title_text=labels['llm_score'], row=2, col=1)
    fig.update_yaxes(title_text=labels['bert_score'], row=2, col=1)
    
    fig.update_xaxes(title_text=labels['model_sentiment'], row=2, col=2)
    fig.update_yaxes(title_text=labels['confidence'], row=2, col=2)
    
    # Save figure
    fig.write_html(output_file)
    print(f"Sentiment comparison dashboard saved to {output_file}")
    
    return fig

def create_confusion_matrix(df, output_file="sentiment_confusion_matrix.png", use_dutch=True):
    """
    Create a confusion matrix comparing LLM and BERT sentiment.
    
    Args:
        df: DataFrame with both LLM and BERT sentiment
        output_file: Path to save the output image
        use_dutch: Whether to use Dutch labels in the visualizations
    """
    # Normalize sentiment labels to ensure consistency
    df = normalize_sentiment_labels(df)
    
    # Select language labels
    if use_dutch:
        labels = DUTCH_LABELS
    else:
        labels = {
            'confusion_title': 'LLM vs. BERT Sentiment Confusion Matrix',
            'bert_prediction': 'BERT Sentiment Prediction',
            'llm_assignment': 'LLM Sentiment Assignment',
            'percentage': 'Percentage (%)'
        }
    
    # Create cross-tabulation
    confusion = pd.crosstab(
        df['llm_sentiment'], 
        df['bert_sentiment'],
        normalize='index'
    ) * 100
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion, 
        annot=True, 
        fmt='.1f', 
        cmap='Blues',
        cbar_kws={'label': labels['percentage']}
    )
    plt.xlabel(labels['bert_prediction'])
    plt.ylabel(labels['llm_assignment'])
    plt.title(labels['confusion_title'])
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Confusion matrix saved to {output_file}")

def create_topic_sentiment_comparison(df, output_file="topic_sentiment_comparison.html", use_dutch=True):
    """
    Create a comparison of topic sentiment between LLM and BERT.
    
    Args:
        df: DataFrame with topic and sentiment data
        output_file: Path to save the output HTML file
        use_dutch: Whether to use Dutch labels in the visualizations
    """
    # Normalize sentiment labels to ensure consistency
    df = normalize_sentiment_labels(df)
    
    # Select language labels
    if use_dutch:
        labels = DUTCH_LABELS
    else:
        labels = {
            'topic_sentiment_title': 'LLM vs. BERT Sentiment Distribution by Topic',
        }
    
    # Get top topics by count
    top_topics = df['topic_name'].value_counts().head(8).index.tolist()
    
    # Filter data for top topics
    filtered_df = df[df['topic_name'].isin(top_topics)].copy()
    
    # Create a DataFrame with sentiment percentages for each topic
    llm_sentiment = filtered_df.groupby(['topic_name', 'llm_sentiment']).size().unstack(fill_value=0)
    llm_sentiment_pct = llm_sentiment.div(llm_sentiment.sum(axis=1), axis=0) * 100
    
    bert_sentiment = filtered_df.groupby(['topic_name', 'bert_sentiment']).size().unstack(fill_value=0)
    bert_sentiment_pct = bert_sentiment.div(bert_sentiment.sum(axis=1), axis=0) * 100
    
    # Reshape for plotting
    llm_data = []
    for topic, row in llm_sentiment_pct.iterrows():
        for sentiment, value in row.items():
            llm_data.append({
                'Topic': topic,
                'Sentiment': sentiment,
                'Percentage': value,
                'Source': 'LLM'
            })
    
    bert_data = []
    for topic, row in bert_sentiment_pct.iterrows():
        for sentiment, value in row.items():
            bert_data.append({
                'Topic': topic,
                'Sentiment': sentiment,
                'Percentage': value,
                'Source': 'BERT'
            })
    
    # Combine data
    plot_data = pd.DataFrame(llm_data + bert_data)
    
    # Define sentiment colors (works for both English and Dutch)
    color_map = {
        'positive': '#2ecc71', 'positief': '#2ecc71',
        'neutral': '#95a5a6', 'neutraal': '#95a5a6',
        'negative': '#e74c3c', 'negatief': '#e74c3c',
        'mixed': '#f39c12', 'gemengd': '#f39c12'
    }
    
    # Create grouped bar chart
    fig = px.bar(
        plot_data,
        x='Topic',
        y='Percentage',
        color='Sentiment',
        barmode='group',
        facet_row='Source',
        color_discrete_map=color_map,
        title=labels['topic_sentiment_title'],
        height=800
    )
    
    # Update layout
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Save figure
    fig.write_html(output_file)
    print(f"Topic sentiment comparison saved to {output_file}")
    
    return fig

def create_all_sentiment_comparisons(df, output_dir="output/visualizations", use_dutch=True):
    """
    Create all sentiment comparison visualizations.
    
    Args:
        df: DataFrame with both LLM and BERT sentiment
        output_dir: Directory to save visualizations
        use_dutch: Whether to use Dutch labels in the visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison dashboard
    create_sentiment_comparison_dashboard(
        df, 
        output_file=os.path.join(output_dir, "sentiment_comparison_dashboard.html"),
        use_dutch=use_dutch
    )
    
    # Create confusion matrix
    create_confusion_matrix(
        df,
        output_file=os.path.join(output_dir, "sentiment_confusion_matrix.png"),
        use_dutch=use_dutch
    )
    
    # Create topic sentiment comparison
    create_topic_sentiment_comparison(
        df,
        output_file=os.path.join(output_dir, "topic_sentiment_comparison.html"),
        use_dutch=use_dutch
    )

# Example usage
if __name__ == "__main__":
    # Load analysis results
    results_df = pd.read_csv("output/analysis_results.csv")
    
    # Create comparison visualizations
    create_all_sentiment_comparisons(results_df, use_dutch=True) 