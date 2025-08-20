import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(BASE_DIR, "streamlit_temp")):
    os.makedirs(os.path.join(BASE_DIR, "streamlit_temp"))

# Set page configuration
st.set_page_config(
    page_title="Dutch Feedback Analyzer - Categories",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("Dutch Feedback Analyzer - Category Explorer")
st.markdown("""
This application helps analyze Dutch customer feedback data by exploring predefined business categories.
""")

# Define structured topic schema for categorization
STRUCTURED_TOPIC_SCHEMA = {
    "Relevantie van Email": [
        "Ongewenste reclame", "Spam berichten", "Overbodige communicatie",
        "Opt-out verzoeken", "Reden van contactopname", "Gebrek aan interesse"
    ],
    "Inhoud van Email": [
        "Onjuiste gegevens", "Foute persoonlijke informatie", "Prijsinformatie ontbreekt",
        "Onduidelijke voorwaarden", "Verbruiksinformatie", "Misleidende communicatie"
    ],
    "Prijs / Korting": [
        "Prijsniveau", "Speciale aanbiedingen", "Klantloyaliteit & Kortingen",
        "Kortingspercentage", "Concurrerende aanbieders", "Prijsvergelijking"
    ],
    "Merk": [
        "Merkperceptie", "Naamsbekendheid", "Merkverandering",
        "Brand identity", "Odido merkbeleving", "Algemene reputatie",
        "Ben merkbeleving", "Simpel merkbeleving", "Vergelijking merken"
    ],
    "Klantenservice": [
        "Bereikbaarheid", "Telefonisch contact", "In-store ervaring",
        "Medewerker kwaliteit", "Wachttijden", "Klachtafhandeling"
    ],
    "Kwaliteit van Product": [
        "Netwerkkwaliteit", "Verbindingsstabiliteit", "Service-uitval",
        "Netwerkbereik", "Internetprestaties", "Technische problemen"
    ]
}

# Keywords for mapping to categories
SCHEMA_CATEGORY_KEYWORDS = {
    "Relevantie van Email": ["reclame", "spam", "overbodig", "ik wil dit niet", "waarom mail je", "stoppen met mail"],
    "Inhoud van Email": ["gegevens kloppen niet", "naam onjuist", "prijs niet zichtbaar", "wat kost", "onduidelijk"],
    "Prijs / Korting": ["te duur", "aanbod", "geen korting voor bestaande klanten", "korting te laag", "goedkoper"],
    "Merk": ["slechte naam", "odido", "brand", "merk", "imago", "reputatie", "identiteit", "naam"],
    "Klantenservice": ["service", "contact", "telefonisch contact", "winkel medewerker", "personeel", "wachttijd"],
    "Kwaliteit van Product": ["netwerk", "verbinding", "uitvallen", "valt uit", "bereik", "geen verbinding", "dekking"]
}

# Function to download a dataframe as CSV
def download_link(df, filename, link_text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# Explore Categories
st.header("Explore Categories")

# Select directory containing results
output_dirs = [d for d in os.listdir() if os.path.isdir(d) and (
    "output" in d or 
    "result" in d or 
    "feedback" in d
)]

if not output_dirs:
    st.warning("No output directories found. Please run an analysis first.")
else:
    selected_dir = st.selectbox(
        "Select results directory to explore",
        options=output_dirs
    )
    
    # Check for results file
    results_path = None
    
    # Find the appropriate results file
    result_files = [
        os.path.join(selected_dir, "analysis_results.csv"),
        os.path.join(selected_dir, "final_analysis_results.csv"),
        os.path.join(selected_dir, "topic_modeling_results.csv"),
        os.path.join(selected_dir, "feedback_with_bert_sentiment.csv")
    ]
    
    for file_path in result_files:
        if os.path.exists(file_path):
            results_path = file_path
            break
    
    if not results_path:
        st.error("No results file found in the selected directory.")
    else:
        # Load results
        df = pd.read_csv(results_path)
        
        # Show data info
        st.subheader("Dataset Information")
        st.write(f"Total records: {len(df)}")
        
        # Show available columns
        st.write("Available columns:", ", ".join(df.columns))
        
        # Determine text column
        text_columns = [col for col in df.columns if any(term in col.lower() for term in ['text', 'comment', 'feedback'])]
        text_column = text_columns[0] if text_columns else None
        
        if not text_column:
            st.error("No text column found in the results.")
        else:
            # Let the user select a main category
            main_categories = list(STRUCTURED_TOPIC_SCHEMA.keys())
            selected_category = st.selectbox(
                "Select a main category",
                options=main_categories
            )
            
            if selected_category:
                # Display subcategories
                subcategories = STRUCTURED_TOPIC_SCHEMA[selected_category]
                
                # Display as a table with selection
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.subheader("Subcategories")
                    selected_subcategory = st.radio(
                        "Select a subcategory",
                        options=subcategories
                    )
                
                with col2:
                    st.subheader(f"{selected_category}: {selected_subcategory}")
                    
                    # Find feedback items that match this category and subcategory
                    # We'll use the keywords for this category
                    
                    # First, get the relevant keywords for this category
                    category_keywords = SCHEMA_CATEGORY_KEYWORDS.get(selected_category, [])
                    
                    # Find items containing keywords related to this category and subcategory
                    pattern = '|'.join(category_keywords)
                    if pattern:
                        filtered_df = df[df[text_column].str.contains(pattern, case=False, na=False)]
                        
                        st.write(f"Found {len(filtered_df)} feedback items related to '{selected_category}'")
                        
                        # Display the top 5 items
                        if len(filtered_df) > 0:
                            for idx, row in filtered_df.head(5).iterrows():
                                st.markdown(f"**Feedback {idx}**")
                                st.markdown(f"> {row[text_column]}")
                                st.markdown("---")
                            
                            # Add download option
                            csv = filtered_df.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="category_{selected_category}_{selected_subcategory}.csv">Download all matched items as CSV</a>'
                            st.markdown(href, unsafe_allow_html=True)
                        else:
                            st.info("No feedback items found for this category.")
                            
                    # Show sentiment distribution for this category if available
                    sentiment_col = next((col for col in df.columns if 'sentiment' in col.lower()), None)
                    if sentiment_col and len(filtered_df) > 0:
                        st.subheader("Sentiment Distribution")
                        
                        # Create a bar chart for sentiment distribution
                        sentiment_counts = filtered_df[sentiment_col].value_counts().reset_index()
                        sentiment_counts.columns = ["Sentiment", "Count"]
                        
                        # Map sentiment values to labels
                        sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
                        sentiment_counts["SentimentLabel"] = sentiment_counts["Sentiment"].map(sentiment_labels)
                        
                        # Create the chart
                        fig = px.bar(
                            sentiment_counts,
                            x="SentimentLabel",
                            y="Count",
                            title=f"Sentiment Distribution for '{selected_category}'",
                            color="SentimentLabel",
                            color_discrete_map={"Negative": "red", "Neutral": "gray", "Positive": "green"}
                        )
                        st.plotly_chart(fig, use_container_width=True)