"""
Visualization module for customer feedback analysis.
This module imports and re-exports visualization functions from sentiment_visualizations.py
and adds any additional visualization functions needed.
"""

# Import visualization functions from sentiment_visualizations
from sentiment_visualizations import (
    create_sentiment_comparison_dashboard,
    normalize_sentiment_labels
)

# Define aliases and additional functions needed by create_visualizations method
def create_dashboard(results_df, output_file="dashboard.html"):
    """Create a dashboard of visualization results."""
    return create_sentiment_comparison_dashboard(results_df, output_file=output_file)

def create_topic_distribution_chart(df, output_file):
    """Create a chart showing the distribution of topics."""
    import plotly.express as px
    
    # Count topics
    topic_counts = df['topic_name'].value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']
    
    # Create chart
    fig = px.bar(topic_counts, x='Count', y='Topic', orientation='h',
                 title='Topic Distribution', 
                 labels={'Count': 'Number of Comments', 'Topic': 'Topic'})
    
    # Save to file
    fig.write_html(output_file)
    return fig

def create_sentiment_pie_chart(df, sentiment_column='bert_sentiment', output_file=None):
    """Create a pie chart showing sentiment distribution."""
    import plotly.express as px
    
    # Count sentiments
    sentiment_counts = df[sentiment_column].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    # Create chart
    fig = px.pie(sentiment_counts, values='Count', names='Sentiment',
                title=f'Sentiment Distribution ({sentiment_column})')
    
    # Save to file
    if output_file:
        fig.write_html(output_file)
    return fig

def create_topic_sentiment_heatmap(df, sentiment_column='bert_sentiment', output_file=None):
    """Create a heatmap showing sentiment distribution by topic."""
    import plotly.express as px
    import pandas as pd
    
    # Create cross-tabulation
    topic_sentiment = pd.crosstab(df['topic_name'], df[sentiment_column], normalize='index') * 100
    
    # Reshape for plotting
    plot_df = topic_sentiment.reset_index().melt(id_vars='topic_name', 
                                                var_name='Sentiment', 
                                                value_name='Percentage')
    
    # Create heatmap
    fig = px.density_heatmap(plot_df, x='Sentiment', y='topic_name', z='Percentage',
                           title='Sentiment Distribution by Topic (%)',
                           labels={'topic_name': 'Topic', 'Percentage': 'Percentage (%)'})
    
    # Save to file
    if output_file:
        fig.write_html(output_file)
    return fig

def create_theme_network_graph(df, min_co_occurrence=3, output_file=None):
    """Create a network graph of co-occurring topics/themes."""
    import plotly.graph_objects as go
    import pandas as pd
    import networkx as nx
    
    # Create a graph
    G = nx.Graph()
    
    # Add topics as nodes
    topics = df['topic_name'].unique()
    for topic in topics:
        G.add_node(topic)
    
    # Find co-occurrences (simplified approach)
    for topic1 in topics:
        for topic2 in topics:
            if topic1 != topic2:
                # Count reviews that mention both topics
                topic1_indices = set(df[df['topic_name'] == topic1].index)
                topic2_indices = set(df[df['topic_name'] == topic2].index)
                co_occurrences = len(topic1_indices.intersection(topic2_indices))
                
                if co_occurrences >= min_co_occurrence:
                    G.add_edge(topic1, topic2, weight=co_occurrences)
    
    # Convert to plotly
    pos = nx.spring_layout(G)
    
    # Create edges
    edge_trace = go.Scatter(
        x=[], y=[], line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
    
    # Create nodes
    node_trace = go.Scatter(
        x=[], y=[], text=[], mode='markers+text', textposition="top center",
        hoverinfo='text', marker=dict(color=[], size=[], line=None))
    
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['text'] += (node,)
        node_trace['marker']['color'] += ('rgb(255, 0, 0)',)
        
        # Size based on degree
        size = 10 + 5 * G.degree(node)
        node_trace['marker']['size'] += (size,)
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='Topic Co-occurrence Network',
                    showlegend=False,
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    
    # Save to file
    if output_file:
        fig.write_html(output_file)
    return fig

def create_bertopic_visualizations(topic_model, output_dir):
    """Create standard BERTopic visualizations."""
    import os
    
    # Create topic hierarchical clustering - skip this as it requires original docs
    '''
    try:
        hierarchical_topics = topic_model.hierarchical_topics()
        fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
        fig.write_html(os.path.join(output_dir, "topic_hierarchy.html"))
    except Exception as e:
        print(f"Error creating hierarchical visualization: {str(e)}")
    '''
    
    # Create topic similarity map
    try:
        fig = topic_model.visualize_topics()
        fig.write_html(os.path.join(output_dir, "topic_similarity.html"))
    except Exception as e:
        print(f"Error creating topic similarity visualization: {str(e)}")
    
    # Create barcharts for each topic
    try:
        # Use custom_labels=True to show descriptive topic names
        fig = topic_model.visualize_barchart(top_n_topics=10, custom_labels=True, height=200)
        fig.write_html(os.path.join(output_dir, "topic_barchart.html"))
    except Exception as e:
        print(f"Error creating barchart visualization: {str(e)}")
    
    # Create topic word scores
    try:
        for topic_id in sorted(set(topic_model.topics_)):
            if topic_id != -1:  # Skip outlier topic
                # Use custom_labels=True for term rank visualizations as well
                fig = topic_model.visualize_term_rank(topics=[topic_id], custom_labels=True)
                fig.write_html(os.path.join(output_dir, f"topic_{topic_id}_term_rank.html"))
    except Exception as e:
        print(f"Error creating term rank visualizations: {str(e)}")
    
    return True 