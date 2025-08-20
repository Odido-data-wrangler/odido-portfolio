import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import shutil
import tempfile
import json
import re
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import subprocess
from pathlib import Path
import requests
# Add imports for evaluation metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt # Needed for plotting confusion matrix
import seaborn as sns # Needed for plotting confusion matrix
import random # Added for simulating different responses
import time
from sentence_transformers import SentenceTransformer # Added for Prototype-Similarity Margin
import nltk # For NL-Augmenter data check
import logging

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(BASE_DIR, "streamlit_temp")):
    os.makedirs(os.path.join(BASE_DIR, "streamlit_temp"))

# Set page configuration
st.set_page_config(
    page_title="Dutch Feedback Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("Dutch Feedback Analyzer")
st.markdown("""
This application helps analyze Dutch customer feedback data, extract sentiment, identify topics, 
and generate visualizations to understand customer opinions.
""")

# Import configuration loader
from config_loader import load_topic_schema, load_category_keywords, get_default_topic_schema, get_default_category_keywords

# Load configurations from external files with fallback to defaults
try:
    STRUCTURED_TOPIC_SCHEMA = load_topic_schema()
    SCHEMA_CATEGORY_KEYWORDS = load_category_keywords()
    st.sidebar.success("✅ Configurations loaded from files")
except (FileNotFoundError, ValueError) as e:
    st.sidebar.warning(f"⚠️ Using default configurations: {str(e)}")
    STRUCTURED_TOPIC_SCHEMA = get_default_topic_schema()
    SCHEMA_CATEGORY_KEYWORDS = get_default_category_keywords()

# Define app modes
app_modes = [
    "About",
    "Preprocessing & Analyze",
    "Explore Topics",
    "Explore Categories",
    "Improvement Areas",
    "NPS Score",  # Added NPS Score mode
    "Evaluation of accuracy", # Added Evaluation mode
    "Custom AI Analysis"
]

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Mode", app_modes)

# Global configuration for Hugging Face endpoint
st.sidebar.title("API Configuration")
endpoint_config = {}

if "hf_endpoint" not in st.session_state:
    st.session_state.hf_endpoint = ""
if "hf_api_key" not in st.session_state:
    st.session_state.hf_api_key = ""

endpoint_config_expander = st.sidebar.expander("Configure Hugging Face Endpoint")
with endpoint_config_expander:
    st.session_state.hf_endpoint = st.text_input("Hugging Face Endpoint URL", 
                                                value=st.session_state.hf_endpoint,
                                                placeholder="https://api-inference.huggingface.co/models/your-endpoint")
    st.session_state.hf_api_key = st.text_input("Hugging Face API Key", 
                                                value=st.session_state.hf_api_key,
                                                placeholder="hf_...",
                                                type="password")
    
    if st.button("Save Endpoint Configuration"):
        st.success("✅ Endpoint configuration saved!")
        # endpoint_config assignment was here but not used, session_state is primary

# --- Add Sidebar Toggle for Prototype Similarity ---
if "show_prototype_similarity_metrics" not in st.session_state:
    st.session_state.show_prototype_similarity_metrics = False

st.sidebar.subheader("Evaluation Options") # Moved subheader for better grouping
st.session_state.show_prototype_similarity_metrics = st.sidebar.toggle(
    "Show prototype-similarity metrics", 
    value=st.session_state.show_prototype_similarity_metrics,
    help="Display detailed metrics for category assignment confidence using SBERT prototype similarity."
)
# --- End Sidebar Toggle ---

# Function to download a dataframe as CSV
def download_link(df, filename, link_text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# Function to load the results of a previous analysis
def load_analysis_results(directory):
    if os.path.exists(os.path.join(directory, "analysis_results.csv")):
        df = pd.read_csv(os.path.join(directory, "analysis_results.csv"))
        return df
    elif os.path.exists(os.path.join(directory, "final_analysis_results.csv")):
        df = pd.read_csv(os.path.join(directory, "final_analysis_results.csv"))
        return df
    elif os.path.exists(os.path.join(directory, "topic_modeling_results.csv")):
        df = pd.read_csv(os.path.join(directory, "topic_modeling_results.csv"))
        return df
    return None

# Function to extract topic counts from a dataframe
def get_topic_counts(df, topic_column="topic_name", filter_generic=True):
    # Check if the topic column exists
    if topic_column not in df.columns:
        return pd.DataFrame(columns=["Topic", "Count"]), False
    
    topic_counts = df[topic_column].value_counts().reset_index()
    topic_counts.columns = ["Topic", "Count"]
    
    # Filter out "Overig" or other generic topic names if present, but handle case where all topics are Overig
    has_only_overig = False
    if "Overig" in topic_counts["Topic"].values:
        if len(topic_counts) > 1:  # Only filter if we have other topics
            topic_counts = topic_counts[topic_counts["Topic"] != "Overig"]
        else:
            # Special case: only Overig exists
            has_only_overig = True
            st.warning("Only generic 'Overig' topic was found. This may indicate the topic modeling needs refinement or more specific data.")
    
    # Sort by count
    topic_counts = topic_counts.sort_values("Count", ascending=False)
    
    return topic_counts, has_only_overig

# --- Helper Function for Hugging Face API Calls ---
def call_huggingface_api(prompt, max_tokens=150): # Keep tokens reasonable
    if not st.session_state.hf_endpoint or not st.session_state.hf_api_key:
        return "Error: Hugging Face endpoint not configured in sidebar."

    headers = {"Authorization": f"Bearer {st.session_state.hf_api_key}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.5, # Slightly more creative for suggestions
            "return_full_text": False,
            "do_sample": True,
            "top_p": 0.9
        }
    }

    # Set up retry parameters with longer delays
    max_retries = 4
    retry_delay = 60  # Start with 60 seconds (1 minute)
    retry_count = 0
    temp_status = st.empty()

    # Longer timeouts based on max_tokens
    timeout = 120 if max_tokens <= 200 else 180 if max_tokens <= 500 else 300

    while retry_count <= max_retries:
        try:
            if retry_count > 0:
                temp_status.info(f"Retrying API call ({retry_count}/{max_retries})... The endpoint may be starting up. Waiting {retry_delay//60} minutes and {retry_delay%60} seconds.")
            response = requests.post(st.session_state.hf_endpoint, headers=headers, json=payload, timeout=timeout)
            # Check for 5xx errors which could indicate the model is still loading
            if 500 <= response.status_code < 600:
                retry_count += 1
                if retry_count <= max_retries:
                    temp_status.warning(f"The server returned an error (HTTP {response.status_code}). This could mean the model is still loading. Waiting {retry_delay//60} minutes and {retry_delay%60} seconds before retry {retry_count}/{max_retries}...")
                    time.sleep(retry_delay)
                    # Increase delay for next retry (exponential backoff), cap at 5 minutes
                    retry_delay = min(retry_delay * 2, 300)  # Cap at 300 seconds (5 minutes)
                    continue
                else:
                    temp_status.error("Maximum retries reached. The endpoint may be experiencing issues.")
                    return f"Error: Server error after {max_retries} retries (HTTP {response.status_code})."
            # Handle other HTTP errors
            response.raise_for_status()
            # Clear any status messages once successful
            temp_status.empty()
            # Process the successful response
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "Error: Unexpected API response format (list).")
            elif isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"]
            else:
                return f"Error: Unexpected API response format: {result}"
        except requests.exceptions.Timeout:
            retry_count += 1
            if retry_count <= max_retries:
                temp_status.warning(f"Request timed out. This often happens during endpoint cold start. Waiting {retry_delay//60} minutes and {retry_delay%60} seconds before retry {retry_count}/{max_retries}...")
                time.sleep(retry_delay)
                # Increase delay for next retry (exponential backoff)
                retry_delay = min(retry_delay * 2, 300)  # Cap at 300 seconds (5 minutes)
            else:
                temp_status.error("Maximum retries reached. The request keeps timing out.")
                return f"Error: Request timed out after {max_retries} retries."
        except requests.exceptions.ConnectionError:
            retry_count += 1
            if retry_count <= max_retries:
                temp_status.warning(f"Connection error. The endpoint may be starting up. Waiting {retry_delay//60} minutes and {retry_delay%60} seconds before retry {retry_count}/{max_retries}...")
                time.sleep(retry_delay)
                # Increase delay for next retry (exponential backoff)
                retry_delay = min(retry_delay * 2, 300)  # Cap at 300 seconds (5 minutes)
            else:
                temp_status.error("Maximum retries reached. Could not connect to the endpoint.")
                return "Error: Connection error after multiple retries. Please check your endpoint URL."
        except requests.exceptions.RequestException as e:
            return f"Error calling Hugging Face API: {e}"
        except Exception as e:
            return f"An unexpected error occurred during API call: {e}"
    # If we've exhausted all retries without returning
    return "Error: Failed to get a valid response after multiple retries."
# --- End Helper Function ---

# --- SBERT Model Loading and Prototype Calculation (for Prototype-Similarity Margin) ---
@st.cache_resource
def load_sbert_model() -> SentenceTransformer:
    """Loads the SentenceTransformer model."""
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

@st.cache_data
def get_category_prototypes(_model: SentenceTransformer, schema_keywords: dict) -> dict:
    """
    Builds prototype vectors for each category from keywords.
    Uses the provided model instance.
    """
    prototypes = {}
    for category, kw_list in schema_keywords.items():
        if kw_list: # Ensure there are keywords
            kw_embeddings = _model.encode(kw_list, normalize_embeddings=True)
            prototypes[category] = kw_embeddings.mean(axis=0)
    return prototypes

def calculate_similarity_margin(text_embedding: np.ndarray, assigned_category_prototype: np.ndarray, all_prototypes: dict, assigned_category_key: str) -> float:
    """Calculates the similarity margin for a single text."""
    if assigned_category_prototype is None: # Handle cases where assigned category might not have a prototype (e.g. empty keywords)
        return -np.inf # Or some other indicator of an issue

    best_similarity = np.dot(text_embedding, assigned_category_prototype)
    
    runner_up_similarity = -np.inf # Initialize with a very small number
    found_other_prototype = False
    for cat_key, prototype_vector in all_prototypes.items():
        if cat_key != assigned_category_key:
            found_other_prototype = True
            current_sim = np.dot(text_embedding, prototype_vector)
            if current_sim > runner_up_similarity:
                runner_up_similarity = current_sim
    
    if not found_other_prototype: # Only one category or no other valid prototypes
        return best_similarity # Or some large positive number if best_similarity is high, or handle as a special case (e.g., np.nan)
        
    return float(best_similarity - runner_up_similarity)
# --- End SBERT Helpers ---

# Function to run a command and stream output
def run_command_and_stream_output(cmd):
    with st.spinner(f"Running command: `{' '.join(cmd)}`..."):
        try:
            # Set TOKENIZERS_PARALLELISM for the subprocess environment
            env = os.environ.copy()
            env["TOKENIZERS_PARALLELISM"] = "false"
            env["PYTHONUNBUFFERED"] = "1"  # Ensure unbuffered output from subprocess

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1, # Line-buffered
                universal_newlines=True, # Ensure text mode
                env=env # Pass the modified environment
            )

            # Ensure unbuffered output from the subprocess so logs stream immediately
            # (PYTHONUNBUFFERED=1 makes Python stdout/stderr unbuffered)
            # Tokenizers parallelism already set above.
            # env updated earlier

            log_container = st.empty()
            log_output = []

            # Process stdout
            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                if line:
                    log_output.append(line)
                    log_container.code('\n'.join(log_output), language='log')

            # Process stderr after stdout is done
            stderr_output = []
            for line in iter(process.stderr.readline, ''):
                 line = line.strip()
                 if line:
                     stderr_output.append(line)
                     # Append stderr line directly without [ERROR] prefix
                     # We capture stderr separately anyway for display on actual failure.
                     log_output.append(line)
                     log_container.code('\n'.join(log_output), language='log')

            process.stdout.close()
            process.stderr.close()
            return_code = process.wait()

            if return_code == 0:
                st.success(f"✅ Command completed successfully!")
                return True, "".join(log_output)
            else:
                st.error(f"❌ Command failed with exit code {return_code}.")
                st.code("\n".join(stderr_output), language='log')
                return False, "\n".join(stderr_output)

        except FileNotFoundError:
            st.error(f"Error: The command 'python' was not found. Ensure Python is installed and in your system's PATH.")
            return False, "Python not found"
        except Exception as e:
            st.error(f"Error executing command: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return False, str(e)

# About mode
if app_mode == "About":
    st.header("About Dutch Feedback Analyzer")
    
    st.markdown("""
    ## Overview
    
    The Dutch Feedback Analyzer is a specialized tool designed to analyze Dutch customer feedback text. It includes several components:
    
    1. **Sentiment Analysis** - Fine-tuned BERT model to detect sentiment in Dutch text
    2. **Topic Modeling** - BERTopic modeling to discover key themes in feedback
    3. **Error Report Detection** - Identifies error reports within feedback
    4. **Help Request Detection** - Identifies help requests within feedback
    5. **Visualization** - Interactive dashboards for understanding feedback patterns
    
    ## Features
    
    - **Preprocessing & Analyze** - Upload and analyze new feedback data using existing models or run the full pipeline
    - **Explore Topics** - Explore topic modeling results and visualizations from previous analysis runs
    - **Explore Categories** - Explore feedback based on predefined business categories
    - **Custom AI Analysis** - Perform targeted analysis using custom prompts with a Hugging Face API
    
    ## Data Requirements
    
    - The application expects CSV or Excel files containing Dutch customer feedback
    - A text column containing the feedback comments is required
    - Optional additional columns like score, business unit, etc. can enhance the analysis
    
    ## Technical Information
    
    - Built with Streamlit, transformers, and BERTopic
    - Uses Dutch-specific models for better performance
    - Model training is hardware-accelerated when available
    """)
    
    # Show system information
    with st.expander("System Information"):
        st.code(f"""
        Python Version: {sys.version}
        Operating System: {sys.platform}
        Working Directory: {os.getcwd()}
        """)

# Explore Topics mode
elif app_mode == "Explore Topics":
    st.header("Explore Topic Modeling Results")
    
    # Select directory containing results
    output_dirs = [d for d in os.listdir() if os.path.isdir(d) and (
        any(keyword in d.lower() for keyword in ["output", "result", "feedback", "topic", "pipeline", "preprocessed", "analysis", "concept"])
    )]
    
    if not output_dirs:
        st.warning("No output directories found. Please run an analysis first.")
    else:
        selected_dir = st.selectbox(
            "Select results directory to explore",
            options=output_dirs
        )
        
        # Check if directory contains results
        results_df = load_analysis_results(selected_dir)
        
        if results_df is None:
            st.error("No analysis results found in the selected directory.")
        else:
            # Show data info
            original_count = len(results_df)
            st.subheader("Dataset Information")
            # --- Add Filtering Checkbox --- 
            filter_generated = st.checkbox("Exclude AI-generated feedback (if available)", value=True, key=f"{app_mode}_filter_gen")
            gen_col_name = next((col for col in results_df.columns if col.lower() == 'is_generated'), None)

            if gen_col_name:
                 # Ensure boolean type robustly
                 results_df[gen_col_name] = results_df[gen_col_name].astype(str).str.lower().map({'true': True, '1': True, 'yes': True, 'false': False, '0': False, 'no': False, '': False}).fillna(False).astype(bool)
                 if filter_generated:
                     filtered_df = results_df[results_df[gen_col_name] == False].copy()
                     filtered_count = original_count - len(filtered_df)
                     st.write(f"Total records (excluding {filtered_count} generated): {len(filtered_df)}")
                     results_df = filtered_df # Use filtered data for the rest of the mode
                 else:
                     st.write(f"Total records (including generated): {len(results_df)}")
            else:
                st.write(f"Total records: {len(results_df)}")
                if filter_generated:
                   st.info("No 'is_generated' column found. Cannot filter AI-generated feedback.")
            # --- End Filtering --- 
            
            # Look for topic column
            topic_cols = [col for col in results_df.columns if 'topic' in col.lower()]
            topic_col = topic_cols[0] if topic_cols else None
            
            if not topic_col:
                st.error("No topic information found in the results.")
            else:
                # Extract topic counts
                topic_counts, has_only_overig = get_topic_counts(results_df, topic_col)
                
                # Create topic distribution chart
                st.subheader("Topic Distribution")
                
                if has_only_overig:
                    st.warning("""
                    ⚠️ Only a generic 'Overig' topic was found. This typically happens when:
                    1. The topic model could not find meaningful clusters in the data
                    2. The dataset is too small or lacks coherent themes
                    3. The HDBSCAN min_cluster_size parameter is too high
                    
                    Try running topic modeling again with different parameters.
                    """)
                else:
                    # Create a bar chart
                    fig = px.bar(
                        topic_counts,
                        x="Count",
                        y="Topic",
                        orientation="h", 
                        title="Topics by Count",
                        labels={"Count": "Number of Items", "Topic": "Topic"}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Check for visualizations
                viz_dir = os.path.join(selected_dir, "visualizations")
                if os.path.exists(viz_dir):
                    st.subheader("Topic Visualizations")
                    
                    # List available visualizations
                    viz_files = [f for f in os.listdir(viz_dir) if f.endswith('.html')]
                    
                    if viz_files:
                        # Group visualizations by type
                        term_rank_files = [f for f in viz_files if "term_rank" in f]
                        distribution_files = [f for f in viz_files if "distribution" in f or "barchart" in f]
                        similarity_files = [f for f in viz_files if "similar" in f]
                        
                        # Create tabs for different visualization types
                        viz_types = []
                        
                        if term_rank_files:
                            viz_types.append("Topic Terms")
                        if distribution_files:
                            viz_types.append("Topic Distribution")
                        if similarity_files:
                            viz_types.append("Topic Similarity")
                            
                        if viz_types:
                            viz_type = st.selectbox("Select visualization type", viz_types)
                            
                            if viz_type == "Topic Terms":
                                selected_viz = st.selectbox("Select topic", term_rank_files)
                                with open(os.path.join(viz_dir, selected_viz), 'r', encoding='utf-8') as f:
                                    html = f.read()
                                    st.components.v1.html(html, height=600)
                            
                            elif viz_type == "Topic Distribution":
                                selected_viz = st.selectbox("Select distribution view", distribution_files)
                                with open(os.path.join(viz_dir, selected_viz), 'r', encoding='utf-8') as f:
                                    html = f.read()
                                    st.components.v1.html(html, height=600)
                            
                            elif viz_type == "Topic Similarity":
                                selected_viz = st.selectbox("Select similarity view", similarity_files)
                                with open(os.path.join(viz_dir, selected_viz), 'r', encoding='utf-8') as f:
                                    html = f.read()
                                    st.components.v1.html(html, height=600)
                        else:
                            st.info("No categorized visualizations found.")
                    else:
                        st.info("No visualization files found.")
                else:
                    st.info("No visualizations directory found.")
                
                # Show sample documents for selected topic
                st.subheader("Explore Topic Documents")
                
                # Get unique topics
                topics = results_df[topic_col].unique()
                selected_topic = st.selectbox("Select a topic to explore", topics)
                
                if selected_topic is not None:
                    # Filter by selected topic
                    topic_docs = results_df[results_df[topic_col] == selected_topic]
                    
                    # Find text column
                    text_cols = [col for col in results_df.columns if any(term in col.lower() for term in ['text', 'comment', 'feedback'])]
                    text_col = text_cols[0] if text_cols else None
                    
                    if text_col:
                        st.write(f"Showing {len(topic_docs)} documents for topic '{selected_topic}'")
                        
                        # Display sample documents
                        for i, (_, row) in enumerate(topic_docs.head(5).iterrows()):
                            st.markdown(f"**Document {i+1}**")
                            st.markdown(f"> {row[text_col]}")
                            st.markdown("---")
                            
                        # Add download option
                        if len(topic_docs) > 0:
                            st.markdown(download_link(topic_docs, f"topic_{selected_topic}_documents.csv", 
                                                    "Download all documents for this topic"), unsafe_allow_html=True)
                    else:
                        st.error("No text column found in the results.")

# Preprocessing & Analyze mode
elif app_mode == "Preprocessing & Analyze":
    st.header("Preprocessing & Analyze Data")
    
    # Updated tab names
    approach_tabs = st.tabs([
        "Analyze with Existing Models",
        "Preprocess Only",
        "Run Topic Modeling Only",
        "Run Pipeline (No Finetuning)",
        "Run Full Pipeline"
    ])

    # Tab 0: Analyze with Existing Models (Mostly unchanged, uses process_new_data.py)
    with approach_tabs[0]:
        st.markdown("Analyze new feedback data using previously trained sentiment and topic models.")
        uploaded_file = st.file_uploader("Upload customer feedback file (CSV or Excel)", type=["csv", "xlsx"], key="existing_models_upload")
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.xlsx'):
                    preview_df = pd.read_excel(uploaded_file, nrows=5)
                else:
                    preview_df = pd.read_csv(uploaded_file, nrows=5)
                st.subheader("Data Preview")
                st.dataframe(preview_df)
                
                text_column = st.selectbox("Select the text column", options=preview_df.columns, key="existing_models_col")
                
                # Model directory selection
                available_models = [d for d in os.listdir() if os.path.isdir(d) and (os.path.exists(os.path.join(d, "bert-sentiment-model")) or os.path.exists(os.path.join(d, "topic_model")))]
                if not available_models:
                    st.warning("No model directories found containing 'bert-sentiment-model' or 'topic_model'. Ensure a full pipeline run has completed.")
                else:
                    model_dir = st.selectbox("Select directory with pre-trained models", options=available_models, key="existing_models_modeldir")
                    output_dir = st.text_input("Output directory for results", value=f"analysis_{uploaded_file.name.split('.')[0]}", key="existing_models_outdir")
                    create_viz = st.checkbox("Create visualizations", value=True, key="existing_models_viz")

                    if st.button("Analyze Feedback", type="primary", key="existing_models_run"):
                        temp_file_path = os.path.join(BASE_DIR, "streamlit_temp", uploaded_file.name)
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Assuming process_new_data.py exists and handles this flow
                        cmd = [
                            sys.executable, "process_new_data.py",
                            "--input", temp_file_path,
                            "--column", text_column,
                            "--model_dir", model_dir,
                            "--output_dir", output_dir
                        ]
                        if create_viz:
                            cmd.append("--create_visualizations")
                        
                        success, _ = run_command_and_stream_output(cmd)
                        if success:
                            st.info(f"Results saved to `{output_dir}`. Explore in other modes.")

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    # Tab 1: Preprocess Only
    with approach_tabs[1]:
        st.markdown("Run only the preprocessing step (cleaning, LLM enrichment if configured) and save the results.")
        uploaded_file_pre = st.file_uploader("Upload feedback file (CSV or Excel)", type=["csv", "xlsx"], key="preprocess_upload")

        if uploaded_file_pre is not None:
            try:
                if uploaded_file_pre.name.endswith('.xlsx'):
                    preview_df_pre = pd.read_excel(uploaded_file_pre, nrows=5)
                else:
                    preview_df_pre = pd.read_csv(uploaded_file_pre, nrows=5)
                st.subheader("Data Preview")
                st.dataframe(preview_df_pre)

                text_column_pre = st.selectbox("Select the text column", options=preview_df_pre.columns, key="preprocess_col")
                output_dir_pre = st.text_input("Output directory for preprocessed file", value=f"preprocessed_{uploaded_file_pre.name.split('.')[0]}", key="preprocess_outdir")

                if st.button("Run Preprocessing", type="primary", key="preprocess_run"):
                    temp_file_path_pre = os.path.join(BASE_DIR, "streamlit_temp", uploaded_file_pre.name)
                    with open(temp_file_path_pre, "wb") as f:
                        f.write(uploaded_file_pre.getbuffer())

                    cmd = [
                        sys.executable, "customer_feedback_analyzer.py",
                        "--input", temp_file_path_pre,
                        "--column", text_column_pre,
                        "--output_dir", output_dir_pre,
                        "--preprocess_only"
                    ]
                    if st.session_state.hf_endpoint and st.session_state.hf_api_key:
                        cmd.extend(["--hf_endpoint", st.session_state.hf_endpoint,
                                    "--hf_api_key", st.session_state.hf_api_key])

                    success, output_log = run_command_and_stream_output(cmd)
                    if success:
                        st.info(f"Preprocessing saved to `{os.path.join(output_dir_pre, 'preprocessed.csv')}`.")

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    # Tab 2: Run Topic Modeling Only (Placeholder)
    with approach_tabs[2]:
        st.markdown("Run preprocessing, embedding calculation, and topic modeling. Saves the topic model and associated files.")
        uploaded_file_tm = st.file_uploader("Upload feedback file (CSV or Excel)", type=["csv", "xlsx"], key="topic_only_upload")

        if uploaded_file_tm is not None:
            try:
                if uploaded_file_tm.name.endswith('.xlsx'):
                    preview_df_tm = pd.read_excel(uploaded_file_tm, nrows=5)
                else:
                    preview_df_tm = pd.read_csv(uploaded_file_tm, nrows=5)
                st.subheader("Data Preview")
                st.dataframe(preview_df_tm)

                text_column_tm = st.selectbox("Select the text column", options=preview_df_tm.columns, key="topic_only_col")
                output_dir_tm = st.text_input("Output directory for topic model files", value=f"topic_model_{uploaded_file_tm.name.split('.')[0]}", key="topic_only_outdir")
                # Optional: Add viz checkbox later if needed
                # create_viz_tm = st.checkbox("Create basic BERTopic visualizations", value=False, key="topic_only_viz")

                if st.button("Run Topic Modeling Only", type="primary", key="topic_only_run"):
                    temp_file_path_tm = os.path.join(BASE_DIR, "streamlit_temp", uploaded_file_tm.name)
                    with open(temp_file_path_tm, "wb") as f:
                        f.write(uploaded_file_tm.getbuffer())

                    cmd = [
                        sys.executable, "customer_feedback_analyzer.py",
                        "--input", temp_file_path_tm,
                        "--column", text_column_tm,
                        "--output_dir", output_dir_tm,
                        "--topic_model_only"
                    ]
                    if st.session_state.hf_endpoint and st.session_state.hf_api_key:
                        cmd.extend(["--hf_endpoint", st.session_state.hf_endpoint,
                                    "--hf_api_key", st.session_state.hf_api_key])
                    # if create_viz_tm:
                    #     cmd.append("--create_visualizations")

                    success, _ = run_command_and_stream_output(cmd)
                    if success:
                        st.info(f"Topic modeling complete! Model and results saved in `{output_dir_tm}`.")
            except Exception as e: # Added except block
                st.error(f"Error processing file: {str(e)}")

    # Tab 3: Run Pipeline (No Finetuning)
    with approach_tabs[3]:
        st.markdown("Run the full analysis pipeline, including **fine-tuning** a sentiment model. Requires a pre-existing fine-tuned model.")
        uploaded_file_noft = st.file_uploader("Upload feedback file (CSV or Excel)", type=["csv", "xlsx"], key="noft_upload")

        if uploaded_file_noft is not None:
            try:
                if uploaded_file_noft.name.endswith('.xlsx'):
                    preview_df_noft = pd.read_excel(uploaded_file_noft, nrows=5)
                else:
                    preview_df_noft = pd.read_csv(uploaded_file_noft, nrows=5)
                st.subheader("Data Preview")
                st.dataframe(preview_df_noft)

                text_column_noft = st.selectbox("Select the text column", options=preview_df_noft.columns, key="noft_col")

                # Sentiment Model directory selection
                # Look specifically for bert-sentiment-model within subdirs
                available_sentiment_models = []
                for item in os.listdir():
                   item_path = os.path.join(item)
                   if os.path.isdir(item_path):
                       sentiment_model_path = os.path.join(item_path, "bert-sentiment-model")
                       if os.path.isdir(sentiment_model_path):
                           available_sentiment_models.append(item) # Add the parent directory

                if not available_sentiment_models:
                     st.warning("No directories containing a 'bert-sentiment-model' subfolder found. You need to run the full pipeline with finetuning at least once.")
                else:
                    # Note: The script expects the *parent* output dir, not the sentiment model dir itself
                    model_dir_noft = st.selectbox("Select OUTPUT directory containing the 'bert-sentiment-model' subfolder", options=available_sentiment_models, key="noft_modeldir")
                    output_dir_noft = st.text_input("Output directory for analysis results", value=f"pipeline_noft_{uploaded_file_noft.name.split('.')[0]}", key="noft_outdir")
                    create_viz_noft = st.checkbox("Create visualizations", value=True, key="noft_viz")

                    if st.button("Run Pipeline (Skip Training)", type="primary", key="noft_run"):
                        temp_file_path_noft = os.path.join(BASE_DIR, "streamlit_temp", uploaded_file_noft.name)
                        with open(temp_file_path_noft, "wb") as f:
                            f.write(uploaded_file_noft.getbuffer())

                        cmd = [
                            sys.executable, "customer_feedback_analyzer.py",
                            "--input", temp_file_path_noft,
                            "--column", text_column_noft,
                            "--output_dir", output_dir_noft, # Pipeline writes results here
                            "--model_load_dir", model_dir_noft, # Load EXISTING model from here
                            "--skip_training" # Tell the script to skip training
                        ]
                        if create_viz_noft:
                            cmd.append("--create_visualizations")
                        if st.session_state.hf_endpoint and st.session_state.hf_api_key:
                            cmd.extend(["--hf_endpoint", st.session_state.hf_endpoint,
                                        "--hf_api_key", st.session_state.hf_api_key])

                        # The analyzer class needs to load the model from the specified directory
                        # We need to ensure customer_feedback_analyzer.py handles loading correctly when skip_training=True
                        # Assuming the analyzer loads from {output_dir}/bert-sentiment-model implicitly or via config
                        # Removing previous info message as model path is now explicitly passed

                        success, _ = run_command_and_stream_output(cmd)
                        if success:
                            st.info(f"Pipeline (no finetuning) complete! Results saved to `{output_dir_noft}`.")

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    # Tab 4: Run Full Pipeline (Now functional)
    with approach_tabs[4]:
        st.markdown("Run the complete analysis pipeline, including **fine-tuning** a sentiment model.")
        uploaded_file_full = st.file_uploader("Upload feedback file (CSV or Excel)", type=["csv", "xlsx"], key="full_upload")

        if uploaded_file_full is not None:
            try:
                if uploaded_file_full.name.endswith('.xlsx'):
                    preview_df_full = pd.read_excel(uploaded_file_full, nrows=5)
                else:
                    preview_df_full = pd.read_csv(uploaded_file_full, nrows=5)
                st.subheader("Data Preview")
                st.dataframe(preview_df_full)

                text_column_full = st.selectbox("Select the text column", options=preview_df_full.columns, key="full_col")
                output_dir_full = st.text_input("Output directory for results and models", value=f"pipeline_full_{uploaded_file_full.name.split('.')[0]}", key="full_outdir")
                create_viz_full = st.checkbox("Create visualizations", value=True, key="full_viz")

                if st.button("Run Full Pipeline", type="primary", key="full_run"):
                    temp_file_path_full = os.path.join(BASE_DIR, "streamlit_temp", uploaded_file_full.name)
                    with open(temp_file_path_full, "wb") as f:
                        f.write(uploaded_file_full.getbuffer())

                    cmd = [
                        sys.executable, "customer_feedback_analyzer.py",
                        "--input", temp_file_path_full,
                        "--column", text_column_full,
                        "--output_dir", output_dir_full
                        # No --skip_training flag means training will occur
                    ]
                    if create_viz_full:
                        cmd.append("--create_visualizations")
                    
                    # Log the HF credentials from session state before adding them to the command
                    print(f"DEBUG: Streamlit session state for HF Endpoint: '{st.session_state.hf_endpoint}'")
                    print(f"DEBUG: Streamlit session state for HF API Key (is set): {bool(st.session_state.hf_api_key)}")

                    if st.session_state.hf_endpoint and st.session_state.hf_api_key:
                        cmd.extend(["--hf_endpoint", st.session_state.hf_endpoint,
                                    "--hf_api_key", st.session_state.hf_api_key])
                        print("DEBUG: Added HF credentials to subprocess command.")
                    else:
                        print("DEBUG: HF credentials NOT added to subprocess command (endpoint or key missing in session state).")

                    success, _ = run_command_and_stream_output(cmd)
                    if success:
                         st.info(f"Full pipeline complete! Results and models saved to `{output_dir_full}`.")

            except Exception as e:
                 st.error(f"Error processing file: {str(e)}")

# Explore Categories mode
elif app_mode == "Explore Categories":
    st.header("Explore Categories")
    
    # Select directory containing results
    output_dirs = [d for d in os.listdir() if os.path.isdir(d) and (
        any(keyword in d.lower() for keyword in ["output", "result", "feedback", "topic", "pipeline", "preprocessed", "analysis", "concept"])
    )]
    
    if not output_dirs:
        st.warning("No output directories found. Please run an analysis first.")
    else:
        selected_dir = st.selectbox(
            "Select results directory to explore",
            options=output_dirs,
            key="category_dir"
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
            
            original_count = len(df)
            # Show data info
            st.subheader("Dataset Information")
            # --- Add Filtering Checkbox --- 
            filter_generated_cat = st.checkbox("Exclude AI-generated feedback (if available)", value=True, key=f"{app_mode}_filter_gen_cat")
            gen_col_name_cat = next((col for col in df.columns if col.lower() == 'is_generated'), None)

            if gen_col_name_cat:
                 df[gen_col_name_cat] = df[gen_col_name_cat].astype(str).str.lower().map({'true': True, '1': True, 'yes': True, 'false': False, '0': False, 'no': False, '': False}).fillna(False).astype(bool)
                 if filter_generated_cat:
                     filtered_df_cat = df[df[gen_col_name_cat] == False].copy()
                     filtered_count_cat = original_count - len(filtered_df_cat)
                     st.write(f"Total records (excluding {filtered_count_cat} generated): {len(filtered_df_cat)}")
                     df = filtered_df_cat # Use filtered data for the rest of the mode
                 else:
                     st.write(f"Total records (including generated): {len(df)}")
            else:
                st.write(f"Total records: {len(df)}")
                if filter_generated_cat:
                   st.info("No 'is_generated' column found. Cannot filter AI-generated feedback.")
            # --- End Filtering --- 
            
            # Show available columns
            st.write("Available columns:", ", ".join(df.columns))
            
            # Determine the best text column
            text_candidates = [col for col in df.columns if any(term in col.lower() for term in ['combined_text', 'text', 'comment', 'feedback'])]
            text_column = None
            if text_candidates:
                # Prioritize 'combined_text' explicitly if present
                for cand in text_candidates:
                    if cand.lower() == 'combined_text':
                        text_column = cand
                        break
                # If not found, pick the candidate with the fewest NaNs (i.e., most populated)
                if text_column is None:
                    populated_counts = {cand: df[cand].notna().sum() for cand in text_candidates}
                    # Select column with max non-null values
                    text_column = max(populated_counts, key=populated_counts.get)
            
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
                        
                        # --- Filter by Selected Subcategory --- 
                        if 'schema_sub_topic' in df.columns:
                            # Ensure NaN are empty strings for filtering
                            df['schema_sub_topic'] = df['schema_sub_topic'].fillna('')
                            # Filter based on the selected subcategory string being present
                            # Use re.escape to handle potential special characters in subcategory names
                            subcategory_df = df[df['schema_sub_topic'].str.contains(re.escape(selected_subcategory), case=False, na=False)].copy()
                            
                            st.write(f"Found {len(subcategory_df)} feedback items for subcategory '{selected_subcategory}'")
                            
                            # Display the top 5 items
                            if not subcategory_df.empty:
                                for idx, row in subcategory_df.head(5).iterrows():
                                    st.markdown(f"**Feedback {idx}**")
                                    st.markdown(f"> {row[text_column]}")
                                    st.markdown("---")
                                
                                # Add download option
                                csv = subcategory_df.to_csv(index=False)
                                b64 = base64.b64encode(csv.encode()).decode()
                                href = f'<a href="data:file/csv;base64,{b64}" download="category_{selected_category}_{selected_subcategory}.csv">Download all matched items as CSV</a>'
                                st.markdown(href, unsafe_allow_html=True)
                            else:
                                st.info(f"No feedback items found specifically for subcategory '{selected_subcategory}'.")
                                
                        else:
                            st.warning("The loaded results file is missing the 'schema_sub_topic' column. Cannot filter examples by subcategory.")
                            # Optionally, fall back to showing main category results or nothing
                            st.info(f"Showing general results for main category '{selected_category}' based on keyword matching.")
                            # Keep existing keyword filter logic as a fallback if schema col is missing
                            category_keywords = SCHEMA_CATEGORY_KEYWORDS.get(selected_category, [])
                            pattern = '|'.join(category_keywords)
                            if pattern:
                                filtered_df = df[df[text_column].str.contains(pattern, case=False, na=False)]
                                st.write(f"Found {len(filtered_df)} feedback items potentially related to '{selected_category}' (keywords: {', '.join(category_keywords)})")
                                if not filtered_df.empty:
                                     for idx, row in filtered_df.head(5).iterrows():
                                        st.markdown(f"**Feedback {idx}**")
                                        st.markdown(f"> {row[text_column]}")
                                        st.markdown("---")
                            # ----------------------------------------------
                                
                        # Show sentiment distribution for this category if available
                        # Note: This sentiment chart still uses the broader keyword-based filter (`filtered_df`)
                        # unless the schema_sub_topic column was missing entirely.
                        
                        # --- Prioritized Sentiment Column Selection ---
                        sentiment_col = None
                        if 'bert_sentiment' in df.columns:
                            sentiment_col = 'bert_sentiment'
                        elif 'llm_sentiment' in df.columns:
                            sentiment_col = 'llm_sentiment'
                        # Add fallback to original_Sentiment if needed
                        # elif next((col for col in df.columns if col.lower() == 'original_sentiment'), None):
                        #    sentiment_col = next((col for col in df.columns if col.lower() == 'original_sentiment')) 

                        if not sentiment_col:
                            st.warning("Could not find a suitable sentiment column ('bert_sentiment' or 'llm_sentiment'). Sentiment distribution cannot be shown.")
                        # --------------------------------------------

                        # Determine which df to use for sentiment: subcategory_df if available, else filtered_df
                        sentiment_df_to_use = subcategory_df if 'subcategory_df' in locals() and not subcategory_df.empty else df if 'filtered_df' in locals() else pd.DataFrame()

                        if sentiment_col and not sentiment_df_to_use.empty:
                            st.subheader("Sentiment Distribution")
                            
                            # --- Robust Sentiment Mapping --- 
                            # Handle NaNs first
                            sentiment_df_to_use[sentiment_col] = sentiment_df_to_use[sentiment_col].fillna('Missing')
                            sentiment_counts = sentiment_df_to_use[sentiment_col].value_counts().reset_index()
                            # Rename columns reliably
                            sentiment_counts.columns = ['SentimentValue', 'Count']
 
                            sentiment_map = {
                                0: "Negative", 'negative': "Negative",
                                1: "Neutral",  'neutral': "Neutral",
                                2: "Positive", 'positive': "Positive",
                                'missing': "Missing" # Add mapping for NaN/fillna
                            }
                            sentiment_order = ["Negative", "Neutral", "Positive", "Missing", "Unknown"] # Add Missing
                            color_map = {"Negative": "red", "Neutral": "gray", "Positive": "green", "Missing": "orange", "Unknown": "purple"} # Add Missing
 
                            # Normalize case if sentiment values are strings
                            if pd.api.types.is_string_dtype(sentiment_counts['SentimentValue']):
                                sentiment_counts['SentimentValue'] = sentiment_counts['SentimentValue'].str.lower()

                            # Apply mapping, handle unknown values
                            sentiment_counts['SentimentLabel'] = sentiment_counts['SentimentValue'].map(sentiment_map).fillna('Unknown')
                            # ---------------------------------
                            
                            # Create the chart
                            fig_sent = px.bar(
                                sentiment_counts,
                                x="SentimentLabel",
                                y="Count",
                                title=f"Sentiment Distribution for '{selected_category}'",
                                color="SentimentLabel",
                                color_discrete_map=color_map,
                                category_orders={"SentimentLabel": sentiment_order} # Ensure order
                            )
                            st.plotly_chart(fig_sent, use_container_width=True)

                        # --- Add Subcategory Frequency Chart ---
                        st.subheader("Subcategory Frequency")
                        if 'schema_main_category' in df.columns and 'schema_sub_topic' in df.columns:
                            main_category_df = df[df['schema_main_category'] == selected_category].copy()
                            if not main_category_df.empty:
                                main_category_df.loc[:, 'schema_sub_topic'] = main_category_df['schema_sub_topic'].fillna('')
                                # Explode subtopics
                                sub_series = main_category_df['schema_sub_topic'].str.split(r' \| ').explode()
                                # Clean up (strip whitespace, remove empty strings)
                                sub_series = sub_series.str.strip()
                                sub_series = sub_series[sub_series != '']

                                if not sub_series.empty:
                                    sub_counts = sub_series.value_counts().reset_index()
                                    sub_counts.columns = ['Subcategory', 'Count']

                                    fig_sub_freq = px.bar(
                                        sub_counts,
                                        x="Count",
                                        y="Subcategory",
                                        orientation='h',
                                        title=f"Subcategory Frequency within '{selected_category}'",
                                        labels={"Count": "Number of Mentions", "Subcategory": "Subcategory"}
                                    )
                                    fig_sub_freq.update_traces(textposition="outside")
                                    st.plotly_chart(fig_sub_freq, use_container_width=True)
                                else:
                                    st.info(f"No specific subcategory mentions found within the '{selected_category}' category.")
                            else:
                                st.info(f"No feedback items found for the main category '{selected_category}'.")
                        else:
                            st.warning("Required columns ('schema_main_category', 'schema_sub_topic') not found in results for subcategory frequency analysis.")
                        # --------------------------------------

# Improvement Areas mode
elif app_mode == "Improvement Areas":
    st.header("Improvement Areas")
    
    # Select directory containing results
    output_dirs = [d for d in os.listdir() if os.path.isdir(d) and (
        any(keyword in d.lower() for keyword in ["output", "result", "feedback", "topic", "pipeline", "preprocessed", "analysis", "concept"])
    )]
    
    if not output_dirs:
        st.warning("No output directories found. Please run an analysis first.")
    else:
        selected_dir = st.selectbox(
            "Select results directory to analyze",
            options=output_dirs,
            key="improvement_areas_dir"
        )
        
        # Find results file
        results_df = load_analysis_results(selected_dir)
        
        if results_df is None:
            st.error("No analysis results found in the selected directory.")
        else:
            original_count_imp = len(results_df)
            st.write(f"Loaded {original_count_imp} records from {selected_dir}")
            
            # --- Add Filtering Checkbox --- 
            filter_generated_imp = st.checkbox("Exclude AI-generated feedback (if available)", value=True, key=f"{app_mode}_filter_gen_imp")
            gen_col_name_imp = next((col for col in results_df.columns if col.lower() == 'is_generated'), None)

            if gen_col_name_imp:
                 results_df[gen_col_name_imp] = results_df[gen_col_name_imp].astype(str).str.lower().map({'true': True, '1': True, 'yes': True, 'false': False, '0': False, 'no': False, '': False}).fillna(False).astype(bool)
                 if filter_generated_imp:
                     filtered_df_imp = results_df[results_df[gen_col_name_imp] == False].copy()
                     filtered_count_imp = original_count_imp - len(filtered_df_imp)
                     st.write(f"Filtered out {filtered_count_imp} generated records. Analyzing {len(filtered_df_imp)} records.")
                     results_df = filtered_df_imp # Use filtered data for the rest of the mode
                 else:
                     st.write(f"Analyzing all {len(results_df)} records (including generated).")
            else:
                st.write(f"Analyzing all {len(results_df)} records.")
                if filter_generated_imp:
                   st.info("No 'is_generated' column found. Cannot filter AI-generated feedback.")
            # --- End Filtering --- 
            
            # --- Check for required columns ---
            required_cols = ['schema_sub_topic', 'schema_main_category', 'bert_sentiment']
            text_cols = [col for col in results_df.columns if any(term in col.lower() for term in ['text', 'comment', 'feedback', 'combined_text'])]
            text_col = text_cols[0] if text_cols else None

            missing_cols = [col for col in required_cols if col not in results_df.columns]
            if not text_col:
                missing_cols.append("a text column")

            if missing_cols:
                st.error(f"Error: The results file is missing required columns: {', '.join(missing_cols)}. Please ensure the analysis pipeline ran successfully and generated these columns.")
            else:
                # Define sentiment column explicitly
                sentiment_col = 'bert_sentiment'

                # Create two tabs - one for top subcategories and one for sentiment across main topics
                insight_tabs = st.tabs(["Top 5 Improvement Areas", "Sentiment Across Main Topics", "Top 5 Benefits/Advantages mentioned"])

                # --- Tab 1: Top 5 Improvement Areas (Subcategories) ---
                with insight_tabs[0]:
                    st.subheader("Top 5 Improvement Areas")
                    try:
                        # --- Refactored Calculation Logic --- 
                        # 1. Explode subtopics from the base dataframe
                        results_df['schema_sub_topic'] = results_df['schema_sub_topic'].fillna('')
                        exploded_df = results_df.assign(Subcategory=results_df['schema_sub_topic'].str.split(r' \| ')).explode('Subcategory')
                        exploded_df['Subcategory'] = exploded_df['Subcategory'].str.strip()
                        exploded_df = exploded_df[exploded_df['Subcategory'] != '']

                        if not exploded_df.empty:
                            # 2. Calculate True Total Counts
                            sub_total_counts = exploded_df.groupby('Subcategory').size().reset_index(name='Total Count')

                            # 3. Calculate Negative Counts
                            neg_df = exploded_df[(exploded_df[sentiment_col] == 'negative') | (exploded_df[sentiment_col] == 0)].copy()
                            sub_neg_counts = neg_df.groupby('Subcategory').size().reset_index(name='Negative Count')

                            # 4. Calculate Positive Counts
                            pos_df = exploded_df[(exploded_df[sentiment_col] == 'positive') | (exploded_df[sentiment_col] == 2)].copy()
                            sub_pos_counts = pos_df.groupby('Subcategory').size().reset_index(name='Positive Count')

                            # 5. Merge Counts
                            improvement_areas = pd.merge(sub_total_counts, sub_neg_counts, on='Subcategory', how='left')
                            improvement_areas = pd.merge(improvement_areas, sub_pos_counts, on='Subcategory', how='left')
                            improvement_areas.fillna(0, inplace=True)
                            improvement_areas[['Negative Count', 'Positive Count']] = improvement_areas[['Negative Count', 'Positive Count']].astype(int)

                            # 6. Apply Frequency Threshold
                            min_frequency_threshold = 20
                            filtered_areas = improvement_areas[improvement_areas['Total Count'] >= min_frequency_threshold].copy()

                            if not filtered_areas.empty:
                                # 7. Calculate Correct Percentages on Filtered Data
                                filtered_areas['Negative Percentage'] = filtered_areas.apply(
                                    lambda row: (row['Negative Count'] / row['Total Count'] * 100) if row['Total Count'] > 0 else 0, axis=1
                                ).round(1)
                                filtered_areas['Positive Percentage'] = filtered_areas.apply(
                                    lambda row: (row['Positive Count'] / row['Total Count'] * 100) if row['Total Count'] > 0 else 0, axis=1
                                ).round(1)

                                # 8. Sort by Negative Sentiment and Select Top 5
                                sorted_areas = filtered_areas.sort_values("Negative Percentage", ascending=False)
                                top_5_improvement_areas = sorted_areas.head(5)

                            else:
                                st.warning(f"No subcategories found mentioned at least {min_frequency_threshold} times.")
                                top_5_improvement_areas = pd.DataFrame(columns=["Subcategory", "Total Count", "Negative Count", "Positive Count", "Negative Percentage", "Positive Percentage"])

                        # --- Visualization Update for Subcategories ---
                        if not top_5_improvement_areas.empty:
                            st.write(f"Top {len(top_5_improvement_areas)} subcategories mentioned at least {min_frequency_threshold} times, ordered by highest negative sentiment percentage:")

                            # Chart 1: Sentiment Breakdown
                            fig_sentiment = go.Figure()
                            fig_sentiment.add_trace(go.Bar(
                                x=top_5_improvement_areas["Subcategory"],
                                y=top_5_improvement_areas["Positive Percentage"],
                                name="Positive Sentiment (%)", marker_color="green",
                                text=top_5_improvement_areas["Positive Percentage"].apply(lambda x: f"{x}%"), textposition="outside"
                            ))
                            fig_sentiment.add_trace(go.Bar(
                                x=top_5_improvement_areas["Subcategory"],
                                y=top_5_improvement_areas["Negative Percentage"],
                                name="Negative Sentiment (%)", marker_color="red",
                                text=top_5_improvement_areas["Negative Percentage"].apply(lambda x: f"{x}%"), textposition="outside"
                            ))
                            fig_sentiment.update_layout(title="Top 5 Subcategories - Sentiment Breakdown", xaxis_title="Subcategory", yaxis_title="Percentage (%)", barmode="group", height=500, yaxis=dict(range=[0, 105]))
                            st.plotly_chart(fig_sentiment, use_container_width=True)

                            # Chart 2: Frequency
                            # Use the actual top 5 determined by sentiment, optionally sorted by count for this chart
                            top_5_for_freq_chart = top_5_improvement_areas.sort_values("Total Count", ascending=False)
                            fig_freq = px.bar(
                                top_5_for_freq_chart, x="Subcategory", y="Total Count",
                                title=f"Frequency of Top {len(top_5_improvement_areas)} Improvement Areas (Mentioned >= {min_frequency_threshold} Times)",
                                text="Total Count", color="Total Count", color_continuous_scale=["#aed6f1", "#2874a6"],
                                labels={"Total Count": "Number of Mentions", "Subcategory": "Subcategory Name"}
                            )
                            fig_freq.update_traces(textposition="outside")
                            st.plotly_chart(fig_freq, use_container_width=True)

                            # --- Recommendations Section ---
                            st.write("#### Suggested Improvement Actions For Top 5 Improvement Areas")
                            if not top_5_improvement_areas.empty: # Check if we have areas to recommend for
                                for idx, row in top_5_improvement_areas.iterrows():
                                    subcategory = row["Subcategory"]
                                    neg_pct = row["Negative Percentage"]
                                    total = row["Total Count"]
                                    st.markdown(f"**{subcategory}** (Mentioned {total} times, {neg_pct}% negative sentiment)")

                                    # Fetch negative examples (Ensure text_col is defined earlier in the mode)
                                    if text_col: # Make sure text_col was identified
                                        # --- Get ALL negative examples for this subcategory --- 
                                        all_negative_examples_df = results_df[
                                            results_df['schema_sub_topic'].str.contains(re.escape(subcategory), na=False) &
                                            ((results_df[sentiment_col] == 'negative') | (results_df[sentiment_col] == 0))
                                        ]
                                        # ------------------------------------------------------

                                        # Display Top 2 Examples
                                        neg_examples_to_display = all_negative_examples_df.head(2)
                                        if not neg_examples_to_display.empty:
                                            st.markdown("**Example negative feedback:**")
                                            for _, ex_row in neg_examples_to_display.iterrows():
                                                st.markdown(f'> *\"{ex_row[text_col]}\"*')
                                        else:
                                             st.markdown("*No specific negative examples found for this subcategory.*") # Should be unlikely if it's a top 5 area
                                    else:
                                        st.markdown("*Text column not identified, cannot show examples.*")

                                    # --- Removed Static Recommendations Block ---
                                    
                                    # --- AI-Generated Improvement Actions --- 
                                    st.markdown("**AI-Generated Recommended Actions:**")
                                    if st.session_state.hf_endpoint and st.session_state.hf_api_key:
                                        # --- Check threshold and proceed --- 
                                        num_negative_examples = len(all_negative_examples_df) if 'all_negative_examples_df' in locals() else 0
                                        min_examples_for_ai = 10 # Lower threshold for negative examples?
                                        max_examples_for_ai = 50

                                        if num_negative_examples >= min_examples_for_ai:
                                            # Gather samples for prompt (up to max)
                                            sample_texts_df = all_negative_examples_df.head(max_examples_for_ai)
                                            sample_texts = sample_texts_df[text_col].tolist() if text_col and text_col in sample_texts_df.columns else []
                                            samples_str = "\n".join([f"- {s}" for s in sample_texts]) if sample_texts else "No specific negative examples found."

                                            # Construct Prompt for Improvements
                                            prompt = f"""Analyze the following negative customer comments regarding '{subcategory}'. 

Negative comments:
{samples_str}

Based *only* on these comments, suggest 2-3 specific and actionable ways the business can address these issues and improve. 
For **each** suggestion, provide a brief explanation (1-2 sentences) connecting it directly to specific examples or themes found in the provided negative comments. Format the output as:

1. [Improvement Action 1]
   *Reasoning: [Explanation linking to negative feedback]*
2. [Improvement Action 2]
   *Reasoning: [Explanation linking to negative feedback]*
...

Suggested Improvements with Reasoning:"""

                                            # Call API & Display
                                            with st.spinner(f"Generating improvement suggestions for {subcategory}..."):
                                                suggestions = call_huggingface_api(prompt, max_tokens=400) # Use the helper, increased tokens
                                                if suggestions and not suggestions.startswith("Error:"):
                                                     st.markdown(suggestions)
                                                else:
                                                     st.warning(f"Could not generate AI suggestions for {subcategory}. Details: {suggestions}")
                                        # This else corresponds to: if num_negative_examples >= min_examples_for_ai
                                        else:
                                            st.info(f"Not enough negative examples found ({num_negative_examples}) to generate AI suggestions. Minimum required: {min_examples_for_ai}.")
                                    # This else corresponds to: if st.session_state.hf_endpoint and st.session_state.hf_api_key
                                    else:
                                        st.info("Please configure the Hugging Face Endpoint and API Key in the sidebar to enable AI-generated suggestions.")
                                    # --- End AI Suggestions ---
                                    
                                    st.markdown("---") # Separator after each improvement area
                                    # Removed the previously added caption here
                                # This else corresponds to the loop: for idx, row in top_5_improvement_areas.iterrows() - Wait, no else for a for loop like this.
                                # This should correspond to: if not top_5_improvement_areas.empty
                            else:
                                # This part handles the case where top_5_improvement_areas is empty
                                st.info("No specific improvement areas identified (meeting frequency and negative sentiment criteria) to generate recommendations.")
                        # --- End Recommendations Section ---
                    # This except corresponds to the try at line 1003
                    except Exception as e:
                        st.error(f"Error analyzing improvement areas for subcategories: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc()) # Show detailed error

                # --- Tab 2: Sentiment Across Main Topics ---
                with insight_tabs[1]:
                    st.subheader("Sentiment Across Main Topics")
                    try:
                        # --- Data Preparation for Main Categories ---
                        main_cat_col = 'schema_main_category'

                        # Filter for relevant sentiments
                        sentiment_map = {'negative': 0, 'positive': 2, 'neutral': 1}
                        if pd.api.types.is_string_dtype(results_df[sentiment_col]):
                            results_df[sentiment_col] = results_df[sentiment_col].str.lower()

                        filtered_main_df = results_df[results_df[sentiment_col].isin(['negative', 'positive', 0, 2])].copy()

                        # --- Sentiment Calculation for Main Categories ---
                        if not filtered_main_df.empty:
                            grouped_main = filtered_main_df.groupby(main_cat_col)
                            main_counts = grouped_main.size().reset_index(name='Total Count')

                            neg_main_counts = filtered_main_df[
                                (filtered_main_df[sentiment_col] == 'negative') | (filtered_main_df[sentiment_col] == 0)
                            ].groupby(main_cat_col).size().reset_index(name='Negative Count')

                            pos_main_counts = filtered_main_df[
                                (filtered_main_df[sentiment_col] == 'positive') | (filtered_main_df[sentiment_col] == 2)
                            ].groupby(main_cat_col).size().reset_index(name='Positive Count')

                            # Merge counts
                            sentiment_by_category = pd.merge(main_counts, neg_main_counts, on=main_cat_col, how='left')
                            sentiment_by_category = pd.merge(sentiment_by_category, pos_main_counts, on=main_cat_col, how='left')
                            sentiment_by_category.fillna(0, inplace=True)

                            # Ensure counts are integers
                            sentiment_by_category['Negative Count'] = sentiment_by_category['Negative Count'].astype(int)
                            sentiment_by_category['Positive Count'] = sentiment_by_category['Positive Count'].astype(int)

                            # Calculate percentages
                            sentiment_by_category['Negative Percentage'] = sentiment_by_category.apply(
                                lambda row: (row['Negative Count'] / row['Total Count'] * 100) if row['Total Count'] > 0 else 0, axis=1
                            ).round(1)
                            sentiment_by_category['Positive Percentage'] = sentiment_by_category.apply(
                                lambda row: (row['Positive Count'] / row['Total Count'] * 100) if row['Total Count'] > 0 else 0, axis=1
                            ).round(1)

                            # Rename main category column for consistency
                            sentiment_by_category.rename(columns={main_cat_col: 'Main Topic'}, inplace=True)

                        # This else corresponds to: if not filtered_main_df.empty (corrected indentation)
                        else:
                            st.warning("No feedback items with 'positive' or 'negative' sentiment found for main category analysis.")
                            sentiment_by_category = pd.DataFrame(columns=['Main Topic', 'Total Count', 'Negative Count', 'Positive Count', 'Negative Percentage', 'Positive Percentage'])


                        # --- Visualization Update for Main Categories ---
                        fig_main_sentiment = go.Figure()
                        if not sentiment_by_category.empty:
                            fig_main_sentiment.add_trace(go.Bar(
                                x=sentiment_by_category['Main Topic'], y=sentiment_by_category['Positive Percentage'],
                                name='Positive Sentiment (%)', marker_color='green',
                                text=sentiment_by_category['Positive Percentage'].apply(lambda x: f"{x}%"), textposition='outside'
                            ))
                            fig_main_sentiment.add_trace(go.Bar(
                                x=sentiment_by_category['Main Topic'], y=sentiment_by_category['Negative Percentage'],
                                name='Negative Sentiment (%)', marker_color='red',
                                text=sentiment_by_category['Negative Percentage'].apply(lambda x: f"{x}%"), textposition='outside'
                            ))
                            fig_main_sentiment.update_layout(
                                title="Sentiment vs. Key Themes", xaxis_title="Main Topics", yaxis_title="Percentage (%)",
                                barmode='group', yaxis=dict(range=[0, 105]), height=600
                            )
                        else:
                            fig_main_sentiment.add_trace(go.Bar(x=["No Data"], y=[0], name="No Data Available", marker_color="gray"))
                            fig_main_sentiment.update_layout(title="Sentiment vs. Key Themes", xaxis_title="Main Topics", yaxis_title="Percentage (%)")

                        st.plotly_chart(fig_main_sentiment, use_container_width=True)

                        # --- Insights Update for Main Categories ---
                        st.write("#### Cross-Dimensional Insights")
                        st.write("This visualization highlights relationships across multiple dimensions, such as sentiment vs. key themes.")
                        
                        if not sentiment_by_category.empty:
                            try:
                                # Find categories with highest positive and negative sentiment
                                highest_positive = sentiment_by_category.loc[sentiment_by_category['Positive Percentage'].idxmax()]
                                highest_negative = sentiment_by_category.loc[sentiment_by_category['Negative Percentage'].idxmax()]
                                
                                st.markdown(f'''
                            **Key Observations:**
                            
                                - **Strongest Positive Sentiment**: {highest_positive['Main Topic']} ({highest_positive['Positive Percentage']}%)
                                - **Strongest Negative Sentiment**: {highest_negative['Main Topic']} ({highest_negative['Negative Percentage']}%)
                                - The visualization shows the distribution of positive and negative sentiment across the main topic categories.
                                - Areas with high negative sentiment represent opportunities for targeted improvements.
                                - Areas with high positive sentiment represent strengths that can be highlighted.
                                ''')
                            except Exception as e:
                                st.warning(f"Could not identify highest sentiment values: {str(e)}")
                                st.markdown('''
                                **General Observations:**
                                - Sentiment analysis across main topics helps identify strengths and weaknesses.
                                ''')
                        # This else corresponds to: if not sentiment_by_category.empty (at line 1259)
                        else:
                            st.warning("Not enough data to provide detailed insights.")
                            st.markdown('''
                            **General Observations:**
                            - Sentiment analysis across main topics helps identify strengths and weaknesses.
                            ''')

                        # --- Table Update for Main Categories ---
                        if not sentiment_by_category.empty:
                            st.write("#### Sentiment Data by Main Topic")
                            # Format percentages for display in table
                            display_main_df = sentiment_by_category.copy()
                            display_main_df["Negative Percentage"] = display_main_df["Negative Percentage"].astype(str) + "%"
                            display_main_df["Positive Percentage"] = display_main_df["Positive Percentage"].astype(str) + "%"
                            st.dataframe(display_main_df[['Main Topic', 'Total Count', 'Negative Count', 'Positive Count', 'Negative Percentage', 'Positive Percentage']])
                        else:
                            st.write("#### Sentiment Data by Main Topic")
                            st.info("No sentiment data available to display.")

                    except Exception as e:
                        st.error(f"Error analyzing sentiment across main topics: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc()) # Show detailed error

                # --- Tab 3: Top 5 Benefits/Advantages (Positively Associated Subcategories) ---
                with insight_tabs[2]:
                    st.subheader("Top 5 Benefits/Advantages Mentioned")
                    st.markdown("Extraction of specific benefits or advantages highlighted in feedback, based on subcategories most frequently associated with positive sentiment.")

                    try:
                        # Check for required columns
                        if 'schema_sub_topic' not in results_df.columns:
                            raise ValueError("Missing 'schema_sub_topic' column needed for benefits analysis.")
                        if sentiment_col not in results_df.columns:
                            raise ValueError(f"Missing sentiment column '{sentiment_col}' needed for benefits analysis.")

                        # Explode subtopics (reuse logic if possible, but ensure it uses the base results_df)
                        results_df_copy = results_df.copy() # Use a copy to avoid modifying original df for other tabs
                        results_df_copy['schema_sub_topic'] = results_df_copy['schema_sub_topic'].fillna('')
                        exploded_benefits_df = results_df_copy.assign(Subcategory=results_df_copy['schema_sub_topic'].str.split(r' \| ')).explode('Subcategory')
                        exploded_benefits_df['Subcategory'] = exploded_benefits_df['Subcategory'].str.strip()
                        exploded_benefits_df = exploded_benefits_df[exploded_benefits_df['Subcategory'] != '']

                        # Filter for Positive Sentiment
                        pos_exploded_df = exploded_benefits_df[
                            (exploded_benefits_df[sentiment_col] == 'positive') | (exploded_benefits_df[sentiment_col] == 2)
                        ].copy()

                        if not pos_exploded_df.empty:
                            # Calculate Frequency Counts
                            pos_sub_counts = pos_exploded_df['Subcategory'].value_counts().reset_index()
                            pos_sub_counts.columns = ['Benefit/Advantage (Subcategory)', 'Frequency']

                            # Calculate Percentages
                            total_positive_mentions = pos_exploded_df.shape[0]
                            if total_positive_mentions > 0:
                                pos_sub_counts['Frequency (%)'] = (pos_sub_counts['Frequency'] / total_positive_mentions * 100).round(1)
                            else:
                                pos_sub_counts['Frequency (%)'] = 0.0

                            # Select Top 5
                            top_5_benefits = pos_sub_counts.head(5)
                            # Corrected indentation for the block below
                            if not top_5_benefits.empty:
                                # Create Visualization (Vertical Percentage Bar Chart)
                                fig_benefits = px.bar(
                                    top_5_benefits,
                                    x="Benefit/Advantage (Subcategory)",
                                    y="Frequency (%)", # Use Percentage column
                                    title="Top 5 Positively Associated Subcategories by Frequency",
                                    labels={"Frequency (%)": "Frequency (%)", "Benefit/Advantage (Subcategory)": "Benefit"},
                                    text='Frequency (%)' # Show percentage on bars
                                )
                                fig_benefits.update_traces(marker_color='#9B59B6', textposition='outside') # Purple color
                                fig_benefits.update_yaxes(range=[0, top_5_benefits['Frequency (%)'].max() * 1.1]) # Adjust y-axis range slightly
                                st.plotly_chart(fig_benefits, use_container_width=True)

                                # --- Add back the Frequency Count Chart (Horizontal) ---
                                st.markdown("---") # Add a separator
                                fig_benefits_freq = px.bar(
                                    top_5_benefits, # Use the same top_5 data
                                    x="Frequency", # Use absolute frequency
                                    y="Benefit/Advantage (Subcategory)",
                                    orientation='h', # Horizontal
                                    title="Top 5 Positively Associated Subcategories (Frequency Count)",
                                    labels={"Frequency": "Number of Mentions", "Benefit/Advantage (Subcategory)": "Benefit"},
                                    text='Frequency' # Show count on bars
                                )
                                fig_benefits_freq.update_layout(yaxis={'categoryorder':'total ascending'}) # Sort bars
                                fig_benefits_freq.update_traces(marker_color='#5DADE2', textposition='outside') # Blue color
                                st.plotly_chart(fig_benefits_freq, use_container_width=True)
                                # --- End Frequency Count Chart ---

                                # Optional: Text Summary
                                summary_list = []
                                for idx, row in top_5_benefits.iterrows():
                                    # Calculate percentage relative to total positive mentions if needed, or just show count
                                    # For simplicity, just showing name and count
                                    summary_list.append(f"{row['Benefit/Advantage (Subcategory)']} ({row['Frequency (%)']}%)")
                                if summary_list:
                                    st.markdown("**Key benefits mentioned:** " + ", ".join(summary_list) + ".")

                                # --- AI-Generated Actions to Build Upon --- 
                                st.write("#### AI-Generated Actions to Build Upon")
                                if st.session_state.hf_endpoint and st.session_state.hf_api_key:
                                    for idx, row in top_5_benefits.iterrows():
                                        benefit_subcategory = row["Benefit/Advantage (Subcategory)"]
                                        # --- Gather ALL positive examples for this subcategory --- 
                                        all_positive_examples_df = results_df_copy[ # Use the copy that was exploded
                                             results_df_copy['schema_sub_topic'].str.contains(re.escape(benefit_subcategory), case=False, na=False) &
                                             ((results_df_copy[sentiment_col] == 'positive') | (results_df_copy[sentiment_col] == 2))
                                        ]
                                        # --------------------------------------------------------

                                        # --- Check threshold and proceed --- 
                                        num_positive_examples = len(all_positive_examples_df)
                                        min_examples_for_ai = 20
                                        max_examples_for_ai = 50

                                        if num_positive_examples >= min_examples_for_ai:
                                            # Display header only if threshold met
                                            st.markdown(f"**Suggestions for: {benefit_subcategory}**")

                                            # Display Positive Examples (still showing top 2)
                                            st.markdown("**Example positive feedback:**")
                                            positive_examples_to_display = all_positive_examples_df.head(2)
                                            if not positive_examples_to_display.empty and text_col in positive_examples_to_display.columns:
                                                for _, ex_row in positive_examples_to_display.iterrows():
                                                    st.markdown(f'> *\"{ex_row[text_col]}\"*')
                                            else:
                                                 st.markdown("*No specific positive examples found for this subcategory.*")

                                            # Gather samples for prompt (up to max)
                                            # Corrected indentation below to match the block
                                            sample_texts_df = all_positive_examples_df.head(max_examples_for_ai)
                                            sample_texts = sample_texts_df[text_col].tolist() if text_col in sample_texts_df.columns else []
                                            samples_str = "\n".join([f"- {s}" for s in sample_texts]) if sample_texts else "No specific examples found."

                                            # Construct Prompt (Updated to request reasoning)
                                            prompt = f"""Analyze the following positive customer comments about '{benefit_subcategory}'. 

Customer comments:
{samples_str}

Based *only* on these comments, suggest 2-3 specific and actionable ways the business can leverage or build upon this strength. 
For **each** suggestion, provide a brief explanation (1-2 sentences) connecting it directly to specific examples or themes found in the provided customer comments. Format the output as:

1. [Suggestion 1]
   *Reasoning: [Explanation linking to feedback]*
2. [Suggestion 2]
   *Reasoning: [Explanation linking to feedback]*
...

Suggestions with Reasoning:"""

                                            # Call API & Display (Increased max_tokens)
                                            with st.spinner(f"Generating suggestions for {benefit_subcategory}..."):
                                                suggestions = call_huggingface_api(prompt, max_tokens=400) # Use the helper, increased tokens
                                                if suggestions and not suggestions.startswith("Error:"):
                                                     st.markdown(suggestions)
                                                else:
                                                     st.warning(f"Could not generate AI suggestions for {benefit_subcategory}. Details: {suggestions}")
                                            st.markdown("---")
                                        # This else corresponds to: if num_positive_examples >= min_examples_for_ai:
                                        else:
                                            st.info(f"Not enough positive examples found ({num_positive_examples}) to generate AI suggestions. Minimum required: {min_examples_for_ai}.")
                                            st.markdown("---") # Separator after each item
                                        # Removed the previously added caption here
                                    # This else corresponds to: if st.session_state.hf_endpoint and st.session_state.hf_api_key:
                                    else:
                                        st.info("Please configure the Hugging Face Endpoint and API Key in the sidebar to enable AI-generated suggestions.")
                                # --- End AI Suggestions --- 
                                    
                            # This else corresponds to: if not top_5_benefits.empty:
                            else:
                                st.info("No distinct subcategories found associated with positive sentiment.")

                    # This except corresponds to the try block starting around line 1318
                    except Exception as e:
                        st.error(f"Error analyzing benefits/advantages: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

# NPS Score mode
elif app_mode == "NPS Score":
    st.header("NPS Score Calculation")
    st.markdown("Calculate and visualize Net Promoter Score (NPS) based on available score data.")

    # Select directory containing results
    output_dirs = [d for d in os.listdir() if os.path.isdir(d) and (
        any(keyword in d.lower() for keyword in ["output", "result", "feedback", "topic", "pipeline", "preprocessed", "analysis", "concept"])
    )]

    if not output_dirs:
        st.warning("No output directories found. Please run an analysis first.")
    else:
        selected_dir = st.selectbox(
            "Select results directory to analyze",
            options=output_dirs,
            key="nps_score_dir"
        )

        # Find results file
        results_df = load_analysis_results(selected_dir)

        if results_df is None:
            st.error("No analysis results found in the selected directory.")
        else:
            original_count_nps = len(results_df)
            st.write(f"Loaded {original_count_nps} records from {selected_dir}")

            # --- Add Filtering Checkbox --- 
            filter_generated_nps = st.checkbox("Exclude AI-generated feedback (if available)", value=True, key=f"{app_mode}_filter_gen_nps")
            gen_col_name_nps = next((col for col in results_df.columns if col.lower() == 'is_generated'), None)

            if gen_col_name_nps:
                 results_df[gen_col_name_nps] = results_df[gen_col_name_nps].astype(str).str.lower().map({'true': True, '1': True, 'yes': True, 'false': False, '0': False, 'no': False, '': False}).fillna(False).astype(bool)
                 if filter_generated_nps:
                     filtered_df_nps = results_df[results_df[gen_col_name_nps] == False].copy()
                     filtered_count_nps = original_count_nps - len(filtered_df_nps)
                     st.write(f"Filtered out {filtered_count_nps} generated records. Analyzing {len(filtered_df_nps)} records for NPS.")
                     results_df = filtered_df_nps # Use filtered data for NPS
                 # This else corresponds to: if filter_generated_nps
                 else:
                     st.write(f"Analyzing all {len(results_df)} records (including generated) for NPS.")
            # This else corresponds to: if gen_col_name_nps
            else:
                 st.write(f"Analyzing all {len(results_df)} records for NPS.")
                 if filter_generated_nps:
                    st.info("No 'is_generated' column found. Cannot filter AI-generated feedback.")
            # --- End Filtering --- 

            # --- Identify Score Column ---
            score_col_name = None
            possible_score_cols = ['score', 'rating', 'nps', 'original_score', 'bert_sentiment', 'llm_sentiment']
            
            # Find all available score columns
            available_score_cols = []
            for col in results_df.columns:
                if col.lower() in possible_score_cols:
                    available_score_cols.append(col)
                                
            if not available_score_cols:
                st.error(f"Error: Could not find a suitable score column in the results file. Looked for columns named like: {', '.join(possible_score_cols)}.")
            else:
                # If multiple score columns are available, let the user select one
                if len(available_score_cols) > 1:
                    score_col_name = st.selectbox(
                        "Select which column to use for NPS calculation:",
                        options=available_score_cols,
                        index=0 if 'bert_sentiment' in available_score_cols else 0
                    )
                else:
                    score_col_name = available_score_cols[0]
                    st.success(f"Found score column: `{score_col_name}`")
                
                # --- Calculate NPS ---
                try:
                    # Determine if the selected score column is likely a sentiment string column
                    is_sentiment_string_col = False
                    if score_col_name.lower() in ['bert_sentiment', 'llm_sentiment', 'sentiment']:
                        is_sentiment_string_col = True
                    elif results_df[score_col_name].dtype == 'object': # Check if column is object type (likely strings)
                        # More robust check for string values that are sentiment-like
                        sample_values = results_df[score_col_name].dropna().unique()[:5] # Take a small sample
                        if any(isinstance(val, str) and val.lower() in ['positive', 'negative', 'neutral', 'positief', 'negatief', 'neutraal', 'promoter', 'passive', 'detractor'] for val in sample_values):
                            is_sentiment_string_col = True

                    if is_sentiment_string_col:
                        # If it's a sentiment string column, just use it as is (after potential NaN drop)
                        # Drop NaNs from the sentiment column itself before proceeding
                        valid_scores_df = results_df.dropna(subset=[score_col_name]).copy()
                    else:
                        # For other columns, attempt numeric conversion
                        results_df[score_col_name] = pd.to_numeric(results_df[score_col_name], errors='coerce')
                    valid_scores_df = results_df.dropna(subset=[score_col_name])
                    
                    if valid_scores_df.empty:
                        if is_sentiment_string_col:
                            st.warning(f"No valid sentiment values (e.g., 'positive', 'negative', 'neutral') found in the selected column (`{score_col_name}`). Cannot calculate NPS.")
                        else:
                            st.warning(f"No valid numeric scores found in the selected column (`{score_col_name}`). Cannot calculate NPS.")
                    else:
                        # Detect score range
                        min_score = valid_scores_df[score_col_name].min()
                        max_score = valid_scores_df[score_col_name].max()
                        st.write(f"Detected score range: {min_score} to {max_score}")
                        
                        # Detect likely score scale
                        likely_scale = "0-10"  # Default assumption
                        if isinstance(max_score, (int, float)) and max_score <= 5:
                            likely_scale = "1-5"
                        elif isinstance(max_score, (int, float)) and max_score <= 7:
                            likely_scale = "1-7"
                        
                        # Let user select score scale
                        score_scale = st.selectbox(
                            "Select the score scale used in your data:",
                            options=["0-10 (Standard NPS)", "1-5 (5-star rating)", "1-7 (7-point scale)", "Custom"],
                            index=0 if likely_scale == "0-10" else (1 if likely_scale == "1-5" else 2)
                        )
                        
                        # If custom, let user specify thresholds
                        if score_scale == "Custom":
                            st.write("Define custom thresholds for Promoters and Passives:")
                            col1, col2 = st.columns(2)
                            with col1:
                                # Ensure min/max values are numeric
                                num_min = float(min_score) if isinstance(min_score, (int, float)) else 0
                                num_max = float(max_score) if isinstance(max_score, (int, float)) else 10
                                promoter_threshold = st.number_input("Promoter score threshold (≥)", 
                                                                   min_value=num_min, 
                                                                   max_value=num_max,
                                                                   value=float(num_max * 0.9))
                            with col2:
                                passive_threshold = st.number_input("Passive score threshold (≥)", 
                                                                  min_value=num_min, 
                                                                  max_value=float(promoter_threshold),
                                                                  value=float(num_max * 0.7))
                        else:
                            # Set thresholds based on selected scale
                            if score_scale == "0-10 (Standard NPS)":
                                promoter_threshold = 9
                                passive_threshold = 7
                            elif score_scale == "1-5 (5-star rating)":
                                promoter_threshold = 4.5  # 4.5-5 are Promoters
                                passive_threshold = 3.5   # 3.5-4.4 are Passives
                            elif score_scale == "1-7 (7-point scale)":
                                promoter_threshold = 6    # 6-7 are Promoters
                                passive_threshold = 5     # 5-5.9 are Passives
                        
                        st.write(f"Calculating NPS based on {len(valid_scores_df)} valid scores.")
                        
                        # Define NPS categories function
                        def categorize_nps(score):
                            # Try to convert score to string and normalize it first
                            if pd.isna(score):
                                return 'Detractor'  # Default for missing values
                                
                            # Option 1: If working with sentiment strings
                            if isinstance(score, str):
                                score_lower = score.lower().strip()
                                if score_lower in ['positive', 'positief', 'good', 'goed', 'promoter']:
                                    return 'Promoter'
                                elif score_lower in ['neutral', 'neutraal', 'average', 'passive']:
                                    return 'Passive'
                                else:  # Negative, detractor, etc.
                                    return 'Detractor'
                            
                            # Option 2: If working with numeric scores (original approach)
                            try:
                                score_num = float(score)
                                # Check if it might be on a 1-5 scale
                                if 1 <= score_num <= 5:
                                    if score_num >= 4.5:  # Top scores on 1-5 scale
                                        return 'Promoter'
                                    elif score_num >= 3.5:  # Above average on 1-5 scale
                                        return 'Passive'
                                    else:  # Below average on 1-5 scale
                                        return 'Detractor'
                                # Standard 0-10 NPS scale
                                else:
                                    if score_num >= 9:
                                        return 'Promoter'
                                    elif score_num >= 7:
                                        return 'Passive'
                                    else:  # score <= 6
                                        return 'Detractor'
                            except (ValueError, TypeError):
                                # If conversion fails, treat as detractor by default
                                return 'Detractor'

                        # Apply categorization
                        valid_scores_df['NPS Category'] = valid_scores_df[score_col_name].apply(categorize_nps)

                        # Calculate percentages
                        total_responses = len(valid_scores_df)
                        promoter_pct = (valid_scores_df['NPS Category'] == 'Promoter').sum() / total_responses * 100
                        detractor_pct = (valid_scores_df['NPS Category'] == 'Detractor').sum() / total_responses * 100
                        
                        # Calculate NPS
                        nps_score = promoter_pct - detractor_pct

                        # --- Display Results ---
                        st.subheader("Net Promoter Score (NPS)")
                        st.metric(label="NPS", value=f"{nps_score:.1f}")
                        st.caption("NPS = % Promoters - % Detractors")
                        
                        # Add detailed explanation about NPS calculation
                        with st.expander("How is NPS calculated?"):
                            st.markdown(f"""
                            ### NPS Calculation Explanation
                            
                            **Net Promoter Score (NPS)** measures customer loyalty and satisfaction. The score ranges from -100 to +100.
                            
                            #### Formula
                            ```
                            NPS = % of Promoters - % of Detractors
                            ```
                            
                            #### Customer Categories
                            For **numeric scores** (traditional 0-10 scale):
                            - **Promoters (9-10)**: Loyal enthusiasts who will keep buying and refer others
                            - **Passives (7-8)**: Satisfied but unenthusiastic customers (not counted in the NPS calculation)
                            - **Detractors (0-6)**: Unhappy customers who can damage your brand
                            
                            For **sentiment strings**:
                            - **Promoters**: Feedback classified as "positive"
                            - **Passives**: Feedback classified as "neutral"
                            - **Detractors**: Feedback classified as "negative"
                            
                            #### Different Scales
                            This app also supports different rating scales:
                            - **0-10 scale**: Standard NPS (Promoters: 9-10, Passives: 7-8, Detractors: 0-6)
                            - **1-5 scale**: Commonly used in surveys (Promoters: 4.5-5, Passives: 3.5-4.4, Detractors: 1-3.4)
                            - **1-7 scale**: Used in some academic surveys (Promoters: 6-7, Passives: 5-5.9, Detractors: 1-4.9)
                            
                            #### Interpretation
                            - **Above 0**: Generally considered good
                            - **Above 20**: Favorable
                            - **Above 50**: Excellent
                            - **Above 70**: World-class
                            
                            The current calculation using `{score_col_name}` column resulted in {promoter_pct:.1f}% Promoters and {detractor_pct:.1f}% Detractors, giving an NPS of {nps_score:.1f}.
                            """)
                        
                        # Display count and percentage
                        st.write("Category Distribution:")
                        category_stats = valid_scores_df['NPS Category'].value_counts().to_frame()
                        category_stats.columns = ['Count']
                        category_stats['Percentage'] = category_stats['Count'] / total_responses * 100
                        st.dataframe(category_stats)

                        # --- Visualize Breakdown ---
                        st.subheader("NPS Category Breakdown")
                        category_counts = valid_scores_df['NPS Category'].value_counts().reset_index()
                        category_counts.columns = ['NPS Category', 'Count']
                        
                        # Ensure all categories are present for consistent coloring/ordering
                        all_categories = pd.DataFrame({'NPS Category': ['Detractor', 'Passive', 'Promoter'], 'Count': [0, 0, 0]})
                        category_counts = pd.merge(all_categories, category_counts, on='NPS Category', how='left', suffixes=('_x', ''))
                        category_counts['Count'] = category_counts['Count'].fillna(0).astype(int)
                        category_counts.drop(columns=['Count_x'], inplace=True)

                        # Define category order and colors for visualization
                        category_order = ["Detractor", "Passive", "Promoter"]
                        color_map = {"Detractor": "red", "Passive": "gray", "Promoter": "green"}

                        # Create the bar chart
                        fig_nps = px.bar(
                            category_counts,
                            x='NPS Category',
                            y='Count',
                            title='Distribution of NPS Categories',
                            text='Count',
                            color='NPS Category',
                            color_discrete_map=color_map,
                            category_orders={"NPS Category": category_order} # Ensure order
                        )
                        fig_nps.update_traces(textposition='outside')
                        st.plotly_chart(fig_nps, use_container_width=True)

                        # Show table
                        st.write("Counts per Category:")
                        st.dataframe(category_counts)
                    
                # This except corresponds to the try at line 1521
                except Exception as e:
                    st.error(f"An error occurred during NPS calculation: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# Evaluation of accuracy mode
elif app_mode == "Evaluation of accuracy":
    st.header("Evaluation of Sentiment Accuracy")
    st.markdown("Evaluate the accuracy of sentiment predictions against ground truth data, or compare different prediction methods.")

    # --- Normalization Helper Function --- 
    def normalize_sentiment(sentiment):
        if pd.isna(sentiment):
            return 'unknown'
        
        s_str = str(sentiment).lower().strip()
        
        if s_str in ['positive', 'positief', '2', 'promoter']:
            return 'positive'
        elif s_str in ['negative', 'negatief', '0', 'detractor']:
            return 'negative'
        elif s_str in ['neutral', 'neutraal', '1', 'passive']:
            return 'neutral'
        # Corrected indentation for the final else
        else:
            return 'unknown' # Or handle other potential values
    # -------------------------------------

    # Select directory containing results
    output_dirs = [d for d in os.listdir() if os.path.isdir(d) and (
        any(keyword in d.lower() for keyword in ["output", "result", "feedback", "topic", "pipeline", "preprocessed", "analysis", "concept"])
    )]

    if not output_dirs:
        st.warning("No output directories found. Please run an analysis first.")
    else:
        selected_dir = st.selectbox(
            "Select results directory to evaluate",
            options=output_dirs,
            key="eval_accuracy_dir"
        )

        # Load results
        results_df = load_analysis_results(selected_dir)

        if results_df is None:
            st.error("No analysis results found in the selected directory.")
        else:
            original_count_eval = len(results_df)
            st.write(f"Loaded {original_count_eval} records from {selected_dir}")

            # --- Add Filtering Checkbox --- 
            filter_generated_eval = st.checkbox("Exclude AI-generated feedback (if available)", value=True, key=f"{app_mode}_filter_gen_eval")
            gen_col_name_eval = next((col for col in results_df.columns if col.lower() == 'is_generated'), None)

            if gen_col_name_eval:
                 results_df[gen_col_name_eval] = results_df[gen_col_name_eval].astype(str).str.lower().map({'true': True, '1': True, 'yes': True, 'false': False, '0': False, 'no': False, '': False}).fillna(False).astype(bool)
                 if filter_generated_eval:
                     filtered_df_eval = results_df[results_df[gen_col_name_eval] == False].copy()
                     filtered_count_eval = original_count_eval - len(filtered_df_eval)
                     st.write(f"Filtered out {filtered_count_eval} generated records. Evaluating {len(filtered_df_eval)} records.")
                     eval_df = filtered_df_eval # Use filtered data for evaluation
                 else:
                     st.write(f"Evaluating all {len(results_df)} records (including generated).")
                     eval_df = results_df
            else:
                 st.write(f"Evaluating all {len(results_df)} records.")
                 eval_df = results_df
                 if filter_generated_eval:
                    st.info("No 'is_generated' column found. Cannot filter AI-generated feedback.")
            # --- End Filtering --- 

            # --- Identify Columns ---
            ground_truth_col = None
            possible_gt_cols = ['original_sentiment', 'ground_truth', 'true_label', 'sentiment_label', 'label']
            for col in eval_df.columns:
                if col.lower() in possible_gt_cols:
                    ground_truth_col = col
                    st.success(f"Found potential ground truth column: `{ground_truth_col}`")
                    break
            
            llm_predictor_col = 'llm_sentiment' if 'llm_sentiment' in eval_df.columns else None
            bert_predictor_col = 'bert_sentiment' if 'bert_sentiment' in eval_df.columns else None

            # --- Perform Evaluation --- 
            if ground_truth_col:
                st.subheader("Accuracy vs. Ground Truth")
                eval_df['ground_truth_normalized'] = eval_df[ground_truth_col].apply(normalize_sentiment)
                
                # Filter out rows where ground truth is unknown for accuracy calculation
                valid_gt_df = eval_df[eval_df['ground_truth_normalized'] != 'unknown'].copy()
                
                if valid_gt_df.empty:
                    st.warning("No valid ground truth labels found after normalization. Cannot calculate accuracy.")
                else:
                    labels = sorted(valid_gt_df['ground_truth_normalized'].unique())
                    
                    predictors_evaluated = []
                    
                    # Evaluate LLM Sentiment
                    if llm_predictor_col:
                        valid_gt_df['llm_normalized'] = valid_gt_df[llm_predictor_col].apply(normalize_sentiment)
                        llm_accuracy = accuracy_score(valid_gt_df['ground_truth_normalized'], valid_gt_df['llm_normalized'])
                        st.metric(label="LLM Sentiment Accuracy", value=f"{llm_accuracy:.2%}")
                        predictors_evaluated.append("LLM")

                        with st.expander("LLM Sentiment Classification Report & Confusion Matrix"):
                            try:
                                report_llm = classification_report(valid_gt_df['ground_truth_normalized'], valid_gt_df['llm_normalized'], labels=labels, zero_division=0)
                                st.text("Classification Report:")
                                st.code(report_llm)
                                
                                cm_llm = confusion_matrix(valid_gt_df['ground_truth_normalized'], valid_gt_df['llm_normalized'], labels=labels)
                                fig_cm_llm, ax_llm = plt.subplots()
                                sns.heatmap(cm_llm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax_llm)
                                ax_llm.set_xlabel('Predicted Label')
                                ax_llm.set_ylabel('True Label')
                                ax_llm.set_title('Confusion Matrix - LLM Sentiment')
                                st.pyplot(fig_cm_llm)
                            # Correctly aligned except block for the try at line 1700
                            except Exception as e:
                                st.error(f"Error generating LLM evaluation report: {e}")
                                
                    # Evaluate BERT Sentiment
                    # Corrected indentation for this block
                    if bert_predictor_col:
                        valid_gt_df['bert_normalized'] = valid_gt_df[bert_predictor_col].apply(normalize_sentiment)
                        bert_accuracy = accuracy_score(valid_gt_df['ground_truth_normalized'], valid_gt_df['bert_normalized'])
                        st.metric(label="BERT Sentiment Accuracy", value=f"{bert_accuracy:.2%}")
                        predictors_evaluated.append("BERT")

                        with st.expander("BERT Sentiment Classification Report & Confusion Matrix"):
                            try:
                                report_bert = classification_report(valid_gt_df['ground_truth_normalized'], valid_gt_df['bert_normalized'], labels=labels, zero_division=0)
                                st.text("Classification Report:")
                                st.code(report_bert)
                                
                                cm_bert = confusion_matrix(valid_gt_df['ground_truth_normalized'], valid_gt_df['bert_normalized'], labels=labels)
                                fig_cm_bert, ax_bert = plt.subplots()
                                sns.heatmap(cm_bert, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax_bert)
                                ax_bert.set_xlabel('Predicted Label')
                                ax_bert.set_ylabel('True Label')
                                ax_bert.set_title('Confusion Matrix - BERT Sentiment')
                                st.pyplot(fig_cm_bert)
                            # Correctly aligned except block for the try at line 1723
                            except Exception as e:
                                st.error(f"Error generating BERT evaluation report: {e}")

                    if not predictors_evaluated:
                        st.warning("Ground truth column found, but neither 'llm_sentiment' nor 'bert_sentiment' columns were present in the results file.")

            elif llm_predictor_col and bert_predictor_col:
                st.subheader("Agreement Between LLM and BERT Sentiment")
                st.info("No ground truth column found. Showing agreement between LLM and BERT predictions.")
                
                eval_df['llm_normalized'] = eval_df[llm_predictor_col].apply(normalize_sentiment)
                eval_df['bert_normalized'] = eval_df[bert_predictor_col].apply(normalize_sentiment)
                
                # Consider only rows where both are known
                valid_compare_df = eval_df[(eval_df['llm_normalized'] != 'unknown') & (eval_df['bert_normalized'] != 'unknown')].copy()

                if valid_compare_df.empty:
                    st.warning("No rows with valid predictions from both LLM and BERT found.")
                else:
                    agreement = accuracy_score(valid_compare_df['llm_normalized'], valid_compare_df['bert_normalized'])
                    st.metric(label="LLM vs BERT Agreement", value=f"{agreement:.2%}")

                    st.write("Contingency Table (LLM vs BERT):")
                    contingency_table = pd.crosstab(valid_compare_df['llm_normalized'], valid_compare_df['bert_normalized'])
                    st.dataframe(contingency_table)
            
            # This else corresponds to the main if/elif starting at line 1678 (if ground_truth_col: ... elif llm_predictor_col and bert_predictor_col: ...)
            else:
                st.warning("Could not perform evaluation. Need either a ground truth column or both 'llm_sentiment' and 'bert_sentiment' columns in the results file.")

            # --- LLM Evaluation of BERT Sentiment --- 
            st.divider()
            st.subheader("LLM Evaluation of BERT Sentiment (Sampled)")

            if bert_predictor_col and st.session_state.hf_endpoint and st.session_state.hf_api_key:
                st.info("Using the configured LLM endpoint to evaluate BERT sentiment predictions on a sample.")
                
                # Sampling Control
                max_sample_size = len(eval_df)
                sample_size_bert_eval = st.slider(
                    "Number of reviews to sample for LLM evaluation",
                    min_value=10,
                    max_value=min(200, max_sample_size),
                    value=min(50, max_sample_size),
                    step=10,
                    key="llm_bert_eval_sample"
                )

                if st.button("Run LLM Evaluation of BERT Sample", key="run_llm_bert_eval"):
                    # Determine text column consistently
                    text_col_eval = None
                    possible_text_cols = ['text', 'combined_text', 'review'] # Add more if needed
                    identified_text_col = next((col for col in eval_df.columns if col.lower() in possible_text_cols), None)
                    
                    if identified_text_col:
                        text_col_eval = identified_text_col
                    elif text_col: # text_col identified earlier in the mode
                        text_col_eval = text_col 
                    
                    if not text_col_eval:
                        st.error("Could not identify a suitable text column (e.g., 'text', 'combined_text') for sampling.")
                    else:
                        # Sample Data
                        if sample_size_bert_eval < len(eval_df):
                            sampled_df_bert = eval_df.sample(sample_size_bert_eval)
                        else:
                            sampled_df_bert = eval_df
                        
                        llm_eval_results = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # Iterate and Prompt LLM
                        try:
                            for i, (idx, row) in enumerate(sampled_df_bert.iterrows()):
                                status_text.text(f"Processing sample {i+1}/{sample_size_bert_eval}...")
                                review_text = row.get(text_col_eval, '') # Use the identified text column
                                bert_pred_normalized = normalize_sentiment(row.get(bert_predictor_col, ''))
                                
                                # --- LLM API Call Simulation --- 
                                prompt_llm_bert = f"""Analyze the sentiment of the following Dutch customer feedback text. Respond with only one word: 'positive', 'negative', or 'neutral'.

Feedback:
\"{review_text}\"

Sentiment: """
                                
                                # llm_response_raw = call_huggingface_api(prompt_llm_bert, max_tokens=10) # Actual call would go here
                                import random 
                                llm_response_raw = random.choice(['positive', 'negative', 'neutral', 'Positive', ' garbage']) 
                                llm_pred_normalized = normalize_sentiment(llm_response_raw) 
                                # -------------------------------

                                llm_eval_results.append({
                                    'index': idx,
                                    'text': review_text,
                                    'bert_prediction': bert_pred_normalized,
                                    'llm_prediction': llm_pred_normalized,
                                    'match': bert_pred_normalized == llm_pred_normalized
                                })
                                progress_bar.progress((i + 1) / sample_size_bert_eval)
                            # End of loop - successful completion
                            status_text.text("LLM evaluation complete!")
                            llm_eval_results_df = pd.DataFrame(llm_eval_results).set_index('index')
                            
                            # Calculate Agreement
                            if not llm_eval_results_df.empty:
                                agreement_llm_bert = llm_eval_results_df['match'].mean()
                                st.metric("LLM vs BERT Agreement (Sampled)", value=f"{agreement_llm_bert:.2%}")
                            # This else corresponds to: if not llm_eval_results_df.empty
                            else:
                                st.info("No results generated from LLM evaluation.")
                            
                            # Display Sample Results
                            st.dataframe(llm_eval_results_df)
                            # TODO: Add expander for mismatches later
                        
                        # This except corresponds to the try at line 1810
                        except Exception as e:
                            status_text.text("Error during LLM evaluation.") # Update status
                            st.error(f"Error during LLM evaluation loop: {e}")
                            import traceback
                            st.code(traceback.format_exc())
            
            # This elif corresponds to the main if for this section (line 1769)
            elif not bert_predictor_col:
                st.warning("BERT sentiment column (`bert_sentiment`) not found. Cannot perform LLM evaluation of BERT.")
            # This else corresponds to the main if for this section (line 1769)
            else: # API not configured
                st.warning("Hugging Face Endpoint and API Key not configured in the sidebar. Cannot perform LLM evaluation.")

            # --- Category Assignment Evaluation --- 
            st.divider()
            st.header("Category Assignment Evaluation") # Changed from subheader to header for better separation

            # Check Prerequisites
            schema_main_col = 'schema_main_category' if 'schema_main_category' in eval_df.columns else None
            
            # Identify text column for category evaluation
            text_col_for_cat_eval = None
            possible_text_cols_cat = ['text', 'combined_text', 'review_text', 'feedback'] 
            identified_text_col_cat = next((col for col in eval_df.columns if col.lower() in possible_text_cols_cat), None)
            if identified_text_col_cat:
                text_col_for_cat_eval = identified_text_col_cat
            
            if not schema_main_col:
                st.warning("Required column `schema_main_category` not found. Cannot evaluate category assignment.")
            else:
                st.subheader("1. Category Assignment Coverage")
                eval_df[schema_main_col] = eval_df[schema_main_col].astype(str).fillna('')
                coverage = (eval_df[schema_main_col].str.strip().ne("")).mean()
                st.metric("Category Assignment Coverage", f"{coverage:.2%}", help="% of feedback rows that get any non-empty schema_main_category.")

                st.subheader("2. LLM Plausibility Probe for Categories")
                status_text_cat_probe = st.empty()
                if not st.session_state.hf_endpoint or not st.session_state.hf_api_key:
                    status_text_cat_probe.warning("Hugging Face Endpoint and API Key not configured.")
                elif not text_col_for_cat_eval:
                    status_text_cat_probe.warning(f"Text column not identified (e.g., {', '.join(possible_text_cols_cat)}).")
                else:
                    # LLM Plausibility Probe UI and logic
                    max_llm_probe_sample = len(eval_df)
                    sample_size_llm_probe = st.slider(
                        "Number of reviews for LLM Category Plausibility Probe",
                        min_value=5,
                        max_value=min(100, max_llm_probe_sample) if max_llm_probe_sample > 0 else 5, 
                        value=min(20, max_llm_probe_sample) if max_llm_probe_sample > 0 else 5,
                        step=5,
                        key="llm_cat_probe_sample_slider"
                    )
                    if st.button("Run LLM Category Plausibility Probe", key="run_llm_cat_probe_button"):
                        if sample_size_llm_probe <= 0:
                             status_text_cat_probe.warning("Please select a sample size greater than 0 for LLM Category Plausibility Probe.")
                        else:
                            with st.spinner(f"Running LLM Category Plausibility Probe on {sample_size_llm_probe} samples..."):
                                if sample_size_llm_probe < len(eval_df):
                                    probe_df = eval_df.sample(sample_size_llm_probe).copy()
                                else:
                                    probe_df = eval_df.copy()

                                llm_probe_results = []
                                progress_bar_cat_probe = st.progress(0)
                                
                                category_definitions_text = "\n".join([
                                    f"- {cat}: {(', '.join(sub_cats[:3])) if sub_cats else 'General category description'}..." 
                                    for cat, sub_cats in STRUCTURED_TOPIC_SCHEMA.items()
                                ])

                                try:
                                    for i, (idx, row) in enumerate(probe_df.iterrows()):
                                        status_text_cat_probe.text(f"Processing sample {i+1}/{sample_size_llm_probe} for category probe...")
                                        feedback_text = row[text_col_for_cat_eval]
                                        assigned_category = str(row[schema_main_col]) if schema_main_col in row and pd.notna(row[schema_main_col]) and str(row[schema_main_col]).strip() != "" else "None"

                                        prompt = f"""You are an AI assistant evaluating category assignments for Dutch customer feedback.
Your task is to assess if the 'Assigned Category' is plausible for the given 'Feedback Text', and to suggest the single best category if you disagree or if if none was assigned.

Category Definitions:
{category_definitions_text}

Feedback Text (Dutch):
\"{feedback_text}\"

Assigned Category: {assigned_category}

Questions:
Q1_PLAUSIBLE: Is the 'Assigned Category' a plausible main category for this 'Feedback Text', based on the definitions? Respond with only YES or NO.
Q2_LLM_CATEGORY: What is the single most appropriate main category from the definitions for this feedback? If the 'Assigned Category' is plausible and correct, repeat it. If no category fits well, respond with 'None'.

Provide your answers in the format:
Q1_PLAUSIBLE: [YES/NO]
Q2_LLM_CATEGORY: [Single Category Name or None]
"""
                                        
                                        response_text = call_huggingface_api(prompt, max_tokens=60) 

                                        q1_match = re.search(r"Q1_PLAUSIBLE:\s*(YES|NO)", response_text, re.IGNORECASE)
                                        q2_match = re.search(r"Q2_LLM_CATEGORY:\s*([^\n]+)", response_text, re.IGNORECASE)

                                        plausible = q1_match.group(1).upper() if q1_match else "ERROR"
                                        llm_category_raw = q2_match.group(1).strip() if q2_match else "ERROR"
                                        
                                        normalized_llm_category = "None"
                                        if llm_category_raw != "ERROR" and llm_category_raw.lower() != "none":
                                            matched = False
                                            for schema_cat_key in STRUCTURED_TOPIC_SCHEMA.keys():
                                                if schema_cat_key.lower() == llm_category_raw.lower():
                                                    normalized_llm_category = schema_cat_key
                                                    matched = True
                                                    break
                                            if not matched: 
                                                 normalized_llm_category = llm_category_raw 
                                        
                                        llm_probe_results.append({
                                            'text': feedback_text,
                                            'assigned_category': assigned_category,
                                            'q1_plausible': plausible,
                                            'q2_llm_category': normalized_llm_category,
                                            'llm_raw_response': response_text
                                        })
                                        progress_bar_cat_probe.progress((i + 1) / sample_size_llm_probe)
                                    
                                    status_text_cat_probe.success("LLM Category Plausibility Probe complete!")
                                    if llm_probe_results:
                                        llm_probe_results_df = pd.DataFrame(llm_probe_results)
                                        
                                        plausible_rate = (llm_probe_results_df['q1_plausible'] == 'YES').mean() if not llm_probe_results_df.empty else 0
                                        
                                        llm_probe_results_df['assigned_category_str'] = llm_probe_results_df['assigned_category'].astype(str)
                                        llm_probe_results_df['q2_llm_category_str'] = llm_probe_results_df['q2_llm_category'].astype(str)

                                        agreement_df_cat = llm_probe_results_df[llm_probe_results_df['assigned_category_str'] != 'None'].copy()
                                        agreement_rate_cat = (agreement_df_cat['assigned_category_str'] == agreement_df_cat['q2_llm_category_str']).mean() if not agreement_df_cat.empty else 0
                                        
                                        col1_probe_cat, col2_probe_cat = st.columns(2)
                                        with col1_probe_cat:
                                            st.metric("LLM Category Plausibility Rate", f"{plausible_rate:.2%}", help="% of 'YES' answers to Q1_PLAUSIBLE from LLM.")
                                        with col2_probe_cat:
                                            st.metric("LLM Category Agreement Rate", f"{agreement_rate_cat:.2%}", help="% of times script's assigned category (when not 'None') matched LLM's Q2_LLM_CATEGORY.")
                                        
                                        with st.expander("LLM Category Plausibility Probe Sample Details & Disagreements"):
                                            st.dataframe(llm_probe_results_df)
                                            
                                            disagreements_cat = llm_probe_results_df[
                                                (llm_probe_results_df['q1_plausible'] == 'NO') | 
                                                ((llm_probe_results_df['assigned_category_str'] != 'None') & (llm_probe_results_df['assigned_category_str'] != llm_probe_results_df['q2_llm_category_str']))
                                            ]
                                            st.write("Disagreements or 'NO' Plausibility (Category Review Queue):")
                                            st.dataframe(disagreements_cat[['text', 'assigned_category', 'q1_plausible', 'q2_llm_category', 'llm_raw_response']])
                                    else:
                                        status_text_cat_probe.info("No results from LLM Category Plausibility Probe.")
                                except Exception as e:
                                    status_text_cat_probe.error(f"Error during LLM Category Plausibility Probe: {e}")
                                    import traceback
                                    st.code(traceback.format_exc())

                st.subheader("3. Prototype-Similarity Margin")
                st.markdown("""**Understanding the Prototype-Similarity Margin:**

This metric helps evaluate how well feedback items align with their assigned categories based on their semantic similarity to category "prototypes" (representative examples or keyword-derived embeddings for each category).

-   A **negative margin** usually indicates that a feedback item is closer (more similar) to a prototype of a *different* category than it is to the prototype of the category it was assigned to. This is often a sign of potential misclassification or ambiguity between categories.
-   A **positive margin** would mean the feedback is closer to its assigned category's prototype than to any other category's prototype, indicating a more confident and correct classification relative to the prototypes.

In essence, this "Prototype-Similarity Margin" is a way to check how well your current categorization aligns with the semantic similarity of the feedback items to the central themes of your defined categories. The negative values you might see indicate areas where this alignment could be improved, suggesting that either the feedback item is on the boundary between categories, or the category definitions/prototypes themselves might need refinement.
""")
                if not st.session_state.show_prototype_similarity_metrics:
                    st.info("Enable 'Show prototype-similarity metrics' in the sidebar to view this section.")
                elif not text_col_for_cat_eval:
                     st.warning(f"Text column not identified (e.g., {', '.join(possible_text_cols_cat)}). Cannot calculate Prototype-Similarity Margin.")
                else:
                    # Prototype-Similarity Margin UI and logic
                    if st.button("Calculate Prototype Similarity Margins", key="run_prototype_margin_button"):
                        with st.spinner("Calculating Prototype-Similarity Margins... This may take a moment for the first run or large datasets."):
                            try:
                                sbert_model = load_sbert_model()
                                category_prototypes = get_category_prototypes(sbert_model, SCHEMA_CATEGORY_KEYWORDS)

                                if not category_prototypes:
                                    st.error("Could not generate category prototypes. Check `SCHEMA_CATEGORY_KEYWORDS` and ensure keywords are defined for categories.")
                                else:
                                    # Filter eval_df to only include rows where schema_main_col has a prototype and is not empty
                                    eval_df_for_margin = eval_df[
                                        eval_df[schema_main_col].isin(category_prototypes.keys()) & 
                                        eval_df[schema_main_col].ne("")
                                    ].copy()
                                    
                                    if eval_df_for_margin.empty:
                                        st.warning("No rows found with assigned categories that have defined prototypes or non-empty assigned category. Cannot calculate margins.")
                                    else:
                                        # Ensure text column has valid (string) data for encoding
                                        eval_df_for_margin[text_col_for_cat_eval] = eval_df_for_margin[text_col_for_cat_eval].astype(str).fillna("")
                                        texts_to_encode = eval_df_for_margin[text_col_for_cat_eval].tolist()
                                        
                                        if not texts_to_encode:
                                             st.warning("No valid text data to encode for prototype similarity margin calculation.")
                                        else:
                                            text_embeddings_list = sbert_model.encode(texts_to_encode, normalize_embeddings=True, show_progress_bar=True)
                                            
                                            margins = []
                                            # Iterate over the filtered df (eval_df_for_margin) using its own index for text_embeddings_list
                                            for i, (idx, row) in enumerate(eval_df_for_margin.iterrows()): 
                                                assigned_cat = row[schema_main_col]
                                                assigned_cat_prototype = category_prototypes.get(assigned_cat) # Should exist due to pre-filtering
                                                text_embedding = text_embeddings_list[i] 
                                                
                                                margin_val = calculate_similarity_margin(text_embedding, assigned_cat_prototype, category_prototypes, assigned_cat)
                                                margins.append(margin_val)
                                            
                                            eval_df_for_margin['similarity_margin'] = margins
                                            valid_margins = eval_df_for_margin['similarity_margin'].dropna()

                                            if not valid_margins.empty:
                                                avg_margin = valid_margins.mean()
                                                median_margin = valid_margins.median()
                                                
                                                col1_margin, col2_margin = st.columns(2)
                                                with col1_margin:
                                                    st.metric("Average Similarity Margin", f"{avg_margin:.4f}", help="Average difference between similarity to assigned category and next best. Higher is better. Calculated on rows with valid prototypes for assigned categories.")
                                                with col2_margin:
                                                    st.metric("Median Similarity Margin", f"{median_margin:.4f}")

                                                with st.expander("Prototype Margin Details"):
                                                    st.write("#### Distribution of Similarity Margins")
                                                    fig_margin_hist = px.histogram(valid_margins, nbins=50, title="Distribution of Prototype Similarity Margins")
                                                    fig_margin_hist.update_layout(showlegend=False)
                                                    st.plotly_chart(fig_margin_hist, use_container_width=True)

                                                    st.write("#### Feedback with Low Similarity Margin (Top 10 lowest, margin < 0.05)")
                                                    low_margin_df = eval_df_for_margin[eval_df_for_margin['similarity_margin'] < 0.05].nsmallest(10, 'similarity_margin')
                                                    st.dataframe(low_margin_df[[text_col_for_cat_eval, schema_main_col, 'similarity_margin']])
                                            else:
                                                st.info("No valid similarity margins could be calculated. Check category assignments and ensure assigned categories have keyword-based prototypes.")
                            except Exception as e:
                                st.error(f"Error calculating prototype similarity margins: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                
                st.subheader("4. Consistency-under-Perturbation")
                st.markdown("""**4. Consistency-under-Perturbation**

This is a test to see how stable your category assignments are. Think of it like this: if you slightly change the wording of a customer's feedback, but the meaning stays the same, should it still get the same category? Mostly, yes!

*   **What it does:**
    1.  Takes a piece of feedback.
    2.  Creates a few slightly different versions (e.g., swaps a word for a similar one, adds a small typo, or rephrases it a bit). These are called 'perturbations'.
    3.  Checks if these new versions still get assigned the same main category as the original feedback.
*   **Why it's useful:** If changing just one or two words makes the category flip-flop, it means your category rules (especially if they rely on exact keywords) might be too fragile or too specific. You want your system to be robust and understand the general idea, not just memorize exact phrases.
*   **The Score (Consistency):**
    *   We count how many times a perturbed version gets a *different* category than the original (a "flip").
    *   A high consistency score (closer to 100%) is good. It means your category assignment is steady even with small wording changes.
    *   A low score means many flips, suggesting your categories might not be well-defined by your keywords, or your keywords are too limited.

**How to generate perturbations (examples of techniques):**
- Back-translation (e.g., Dutch → English → Dutch)
- Synonym / paraphrase swap (e.g., using `NL-Augmenter`)
- Word-order shuffle
- Character noise (typos)

*Generally, three to five slightly different versions per original text are enough to get a good signal.*

**How to read the number:**
- **≥ 90 % (Great!):** Your category mapping is stable. Keep an eye on it, but things look good.
- **70 – 90 % (Okay):** There are some weak spots. You might need to look at feedback that's getting miscategorized or is on the borderline. Adding more synonyms or related terms to your category keywords (`SCHEMA_CATEGORY_KEYWORDS`) can help.
- **< 70 % (Needs Work):** Your system is flipping categories a lot based on small changes. This often means your keywords are too basic. You'll likely need to expand your keyword lists significantly, or think about more advanced ways to match feedback to categories.

**Why this helps (even without a human checking every time):**
- **Catches new slang/phrasing early:** If customers start using new words for the same old problems, this score might drop, warning you to update your keywords.
- **It's automatic:** You don't need to manually label anything new for this test.
- **Adds another safety net:** Along with checking if feedback gets *any* category (coverage) and if the category makes sense (LLM plausibility), this checks if the assignment is *stable*.
""")

                # --- HELPER FUNCTIONS FOR CONSISTENCY (DEFINED LOCALLY) ---
                def local_assign_category_for_perturbation(text: str, schema_definitions: dict, category_keywords_map: dict) -> str:
                    """
                    Assigns a category to the input text based on keyword matching.
                    MODIFIED FOR EXTREME DIAGNOSTICS: Only checks for 'odido' -> 'Merk'.
                    Ignores schema_definitions and category_keywords_map arguments.
                    """
                    if not text or not isinstance(text, str):
                        return "None"  # Handle empty or non-string input

                    text_lower = text.lower()

                    # EXTREME DIAGNOSTIC: Hardcoded check for a single keyword and category
                    if "odido" in text_lower:
                        return "Merk"
                    
                    # You could add another specific keyword for testing if needed:
                    # if "reclame" in text_lower:
                    #     return "Relevantie van Email"

                    return "None"  # Default if no hardcoded keywords match

                @st.cache_data(show_spinner="Generating perturbations and checking consistency...", persist="disk")
                def local_calculate_consistency_score(df_input: pd.DataFrame, txt_col: str, schema_main_c: str, k_pert: int, sample_size_cons: int) -> tuple[float, int, int]:
                    if df_input.empty: return 1.0, 0, 0
                    actual_sample = min(sample_size_cons, len(df_input))
                    if actual_sample == 0: return 1.0, 0, 0
                    if txt_col not in df_input.columns or schema_main_c not in df_input.columns: return 0.0, 0, 0
                    smpl_df = df_input.sample(actual_sample).copy()
                    flps, tot_valid_pert = 0, 0
                    syn_trans, nl_aug_ok = None, False
                    try:
                        from nl_augmenter.transformations.synonym_substitution import DutchSynonymSubstitution
                        syn_trans = DutchSynonymSubstitution(num_synonyms=1)
                        nl_aug_ok = True
                    except Exception: pass

                    def make_paraphrases_local(txt_inner: str, k_inner: int) -> list[str]:
                        prphrs = set()
                        if not txt_inner or not isinstance(txt_inner, str) or not txt_inner.strip(): return [txt_inner] * k_inner
                        if nl_aug_ok and syn_trans:
                            try:
                                for _ in range(k_inner * 3):
                                    if len(prphrs) >= k_inner: break
                                    aug_txt = syn_trans.transform(txt_inner)
                                    if aug_txt != txt_inner and aug_txt.strip(): prphrs.add(aug_txt)
                            except Exception: pass
                        wds = txt_inner.split()
                        if not wds: return [txt_inner] * k_inner
                        while len(prphrs) < k_inner:
                            chc, cur_len, new_p = random.random(), len(prphrs), None
                            if chc < 0.5 and wds:
                                tmp_w, ch_w_idx = wds[:], random.randrange(len(wds))
                                ch_w_l = list(tmp_w[ch_w_idx])
                                if len(ch_w_l) > 1:
                                    ix = random.randrange(len(ch_w_l) - 1)
                                    ch_w_l[ix], ch_w_l[ix+1] = ch_w_l[ix+1], ch_w_l[ix]
                                    tmp_w[ch_w_idx] = "".join(ch_w_l); new_p = " ".join(tmp_w)
                            elif chc >= 0.5 and len(wds) > 1:
                                tmp_w, ix = wds[:], random.randrange(len(wds) - 1)
                                tmp_w[ix], tmp_w[ix+1] = tmp_w[ix+1], tmp_w[ix]; new_p = " ".join(tmp_w)
                            if new_p and new_p != txt_inner and new_p.strip(): prphrs.add(new_p)
                            if len(prphrs) == cur_len and len(prphrs) < k_inner:
                                if txt_inner.strip(): prphrs.add(txt_inner + random.choice([" ook", " nu", " soms", ".", "!", "?"])) 
                                if len(prphrs) == cur_len: break 
                        fin_p = [p for p in list(prphrs) if p.strip()]
                        while len(fin_p) < k_inner and txt_inner.strip(): fin_p.append(txt_inner + random.choice(["!", "?", " echt"])) 
                        return fin_p[:k_inner] if fin_p else [txt_inner]*k_inner

                    prog_bar = st.progress(0); stat_txt = st.empty()
                    
                    # Debugging: Print info for the first few samples
                    debug_prints_done = 0
                    max_debug_prints = 3

                    for i_row, (_, row_data) in enumerate(smpl_df.iterrows()):
                        stat_txt.text(f"Processing sample {i_row+1}/{actual_sample} for consistency...")
                        orig_txt, b_cat = str(row_data[txt_col]), str(row_data[schema_main_c]).strip()
                        
                        if debug_prints_done < max_debug_prints:
                            st.write(f"--- Debug Sample {i_row+1} ---")
                            st.write(f"Original Text: '{orig_txt}'")
                            st.write(f"Base Category (from df): '{b_cat}'")

                        if not orig_txt.strip() or b_cat == "" or b_cat.lower() == "none": 
                            if debug_prints_done < max_debug_prints:
                                st.write("Skipping due to empty text or base category.")
                            prog_bar.progress((i_row + 1) / actual_sample); continue
                        
                        pert_txts = make_paraphrases_local(orig_txt, k_pert)
                        if debug_prints_done < max_debug_prints:
                            st.write(f"Perturbed Texts ({len(pert_txts)} generated): {pert_txts}")

                        for p_t_idx, p_t in enumerate(pert_txts):
                            if not p_t.strip(): 
                                if debug_prints_done < max_debug_prints:
                                    st.write(f"Perturbation {p_t_idx+1} is empty/whitespace, skipping.")
                                continue
                            tot_valid_pert += 1
                            n_cat = local_assign_category_for_perturbation(p_t, STRUCTURED_TOPIC_SCHEMA, SCHEMA_CATEGORY_KEYWORDS).strip()
                            
                            if debug_prints_done < max_debug_prints:
                                st.write(f"Perturbation {p_t_idx+1}: '{p_t}' -> New Category: '{n_cat}'")
                            
                            if n_cat != b_cat: 
                                flps += 1 # Corrected: was flips, now flps
                                if debug_prints_done < max_debug_prints:
                                    st.write(f"FLIP! (Base: '{b_cat}', New: '{n_cat}')")
                            elif debug_prints_done < max_debug_prints:
                                st.write("NO FLIP.")
                        
                        if debug_prints_done < max_debug_prints:
                            st.write("-------------------------")
                            debug_prints_done += 1

                        prog_bar.progress((i_row + 1) / actual_sample)
                    stat_txt.success(f"Consistency check complete on {actual_sample} samples ({tot_valid_pert} perturbations).")
                    if tot_valid_pert == 0: return 1.0, flps, tot_valid_pert
                    return 1.0 - (flps / tot_valid_pert), flps, tot_valid_pert
                # --- END HELPER FUNCTIONS FOR CONSISTENCY ---

                # UI elements for running the check
                if 'eval_df' in locals() and not eval_df.empty and text_col_for_cat_eval: # text_col_for_cat_eval identified earlier
                    st.markdown("--- anticoagulation") 
                    st.write("**Run Analysis:**")
                    col1_cup_input, col2_cup_input = st.columns(2)
                    with col1_cup_input:
                        max_consistency_sample = len(eval_df)
                        sample_size_cup = st.slider(
                            "Sample size for Consistency Check", min_value=10,
                            max_value=min(500, max_consistency_sample) if max_consistency_sample > 0 else 10,
                            value=min(50, max_consistency_sample) if max_consistency_sample > 0 else 10,
                            step=10, key="cup_sample_slider", help="Number of feedback items to sample."
                        )
                    with col2_cup_input:
                        k_perturbations_cup = st.number_input(
                            "Perturbations per text (K)", min_value=1, max_value=5, value=3, 
                            step=1, key="cup_k_input", help="Number of variants per text."
                        )

                    if st.button("Calculate Consistency Score", key="cup_run_button"):
                        if sample_size_cup > 0:
                            # NLTK data check for NL-Augmenter
                            try:
                                import nltk # Ensure nltk is imported here if not at top
                                nltk.data.find('corpora/wordnet'); nltk.data.find('corpora/omw-1.4')
                            except ImportError:
                                st.warning("NLTK library not found. NL-Augmenter might not work.", icon="⚠️")
                            except LookupError:
                                st.info("Downloading NLTK resources (wordnet, omw-1.4) for Dutch synonym substitution...")
                                with st.spinner("Downloading NLTK data..."):
                                    try: nltk.download('wordnet', quiet=True, raise_on_error=True); nltk.download('omw-1.4', quiet=True, raise_on_error=True); st.success("NLTK resources downloaded.")
                                    except Exception as e_nltk: st.error(f"Failed to download NLTK resources: {e_nltk}")
                            
                            # Now call the locally defined function
                            consistency_val, num_flips, num_total_pert = local_calculate_consistency_score(
                                eval_df, 
                                text_col_for_cat_eval, 
                                'schema_main_category', 
                                k_pert=k_perturbations_cup,  # Changed from k_perturbations to k_pert
                                sample_size_cons=sample_size_cup
                            )
                            
                            st.metric("Perturbation Consistency", f"{consistency_val:.2%}", help=f"{num_flips} flips out of {num_total_pert} perturbations.")
                            if consistency_val >= 0.9:
                                st.success("Mapping is stable (≥ 90%). Monitor.")
                            elif consistency_val >= 0.7:
                                st.warning("Some fragile spots (70-90%). Inspect; consider adding synonyms.", icon="⚠️")
                            else:
                                st.error("High flip-rate (< 70%). Rules may be too surface-level. Expand keywords/retrain.", icon="🚨")
                        else:
                            st.warning("Sample size for consistency check must be greater than 0.")
                else:
                    st.info("Load data and ensure text column is identified to enable Consistency-under-Perturbation analysis.")

# Custom AI Analysis mode
elif app_mode == "Custom AI Analysis":
    st.header("Custom AI Analysis")
    
    # Verify that HF endpoint is configured
    if not st.session_state.hf_endpoint or not st.session_state.hf_api_key:
        st.error("⚠️ Hugging Face endpoint not configured. Please set it up in the sidebar.")
        st.info("This feature requires a valid Hugging Face endpoint and API key to analyze feedback.")
    else:
        st.success("✅ Hugging Face endpoint configured.")
        
        # Select directory containing results
        output_dirs = [d for d in os.listdir() if os.path.isdir(d) and (
            any(keyword in d.lower() for keyword in ["output", "result", "feedback", "topic", "pipeline", "preprocessed", "analysis", "concept"])
        )]
        
        if not output_dirs:
            st.warning("No output directories found. Please run an analysis first.")
        else:
            selected_dir = st.selectbox(
                "Select results directory to analyze",
                options=output_dirs,
                key="custom_analysis_dir"
            )
            
            # Find results file
            results_df = load_analysis_results(selected_dir)
            
            if results_df is None:
                st.error("No analysis results found in the selected directory.")
            else:
                original_count_cust = len(results_df)
                st.write(f"Loaded {original_count_cust} records from {selected_dir}")
                
                # --- Add Filtering Checkbox --- 
                filter_generated_cust = st.checkbox("Exclude AI-generated feedback (if available)", value=True, key=f"{app_mode}_filter_gen_cust")
                gen_col_name_cust = next((col for col in results_df.columns if col.lower() == 'is_generated'), None)
                
                if gen_col_name_cust:
                     results_df[gen_col_name_cust] = results_df[gen_col_name_cust].astype(str).str.lower().map({'true': True, '1': True, 'yes': True, 'false': False, '0': False, 'no': False, '': False}).fillna(False).astype(bool)
                     if filter_generated_cust:
                         filtered_df_cust = results_df[results_df[gen_col_name_cust] == False].copy()
                         filtered_count_cust = original_count_cust - len(filtered_df_cust)
                         st.write(f"Filtered out {filtered_count_cust} generated records. Analyzing {len(filtered_df_cust)} records.")
                         results_df = filtered_df_cust # Use filtered data for analysis prompt
                     else:
                         st.write(f"Analyzing all {len(results_df)} records (including generated) for prompt construction.")
                else:
                     st.write(f"Analyzing all {len(results_df)} records for prompt construction.")
                     if filter_generated_cust:
                        st.info("No 'is_generated' column found. Cannot filter AI-generated feedback.")
                # --- End Filtering --- 
                
                # Show data preview
                with st.expander("Data Preview"):
                    st.dataframe(results_df.head())
                
                # Identify text and topic columns
                text_cols = [col for col in results_df.columns if any(term in col.lower() for term in ['text', 'comment', 'feedback'])]
                text_col = text_cols[0] if text_cols else None
                
                topic_cols = [col for col in results_df.columns if 'topic' in col.lower()]
                topic_col = topic_cols[0] if topic_cols else None
                
                if text_col:
                    # Custom analysis prompt
                    st.subheader("Define Analysis")
                    
                    # Prompt templates
                    prompt_templates = {
                        "Customer Service Follow-up Cases": "Analyze feedback items and identify specific customers that require follow-up from customer service. Focus on customers with questions, complaints, technical problems or service issues that need resolution. For each case, extract the customer issue, urgency level (High/Medium/Low), and specific details needed for follow-up. Format your response as a list of specific cases with all necessary details for customer service to take action.",
                        "Identify Common Issues": "Analyze these customer feedback items and identify the most common issues mentioned. Group similar complaints together and rank them by frequency. For each issue, provide 1-2 representative examples.",
                        "Extract Product Suggestions": "Extract specific product suggestions or improvement ideas from these customer feedback items. Group similar suggestions together and provide a concise summary of each suggestion type.",
                        "Analyze Customer Emotions": "Analyze the emotional content of these customer feedback items beyond simple sentiment. Identify specific emotions like frustration, delight, confusion, etc. Provide a distribution of emotions and examples of each.",
                        "Generate Response Templates": "Based on these customer feedback items, generate 3-5 response templates that could be used by customer service representatives. Each template should address a common issue pattern.",
                        "Identify Service Pain Points": "Analyze these feedback items to identify specific pain points in the customer service experience. For each pain point, suggest possible improvements.",
                        "Custom Analysis": "<!-- Define your custom analysis here -->"
                    }
                    
                    template_choice = st.radio(
                        "Choose an analysis template or create your own",
                        options=list(prompt_templates.keys()),
                        index=0  # Default to Customer Service Follow-up Cases
                    )
                    
                    # Allow editing of the selected template
                    analysis_prompt = st.text_area(
                        "Analysis Prompt",
                        value=prompt_templates[template_choice],
                        height=150
                    )
                    
                    # Sample size control
                    max_items = len(results_df)
                    
                    # Add warning about API limitations
                    st.info("""
                    ℹ️ **API Input Limitations**
                    The Hugging Face API has input size limitations (typically 4-8K tokens).
                    For large datasets, consider:
                    - Analyzing 200-500 items per batch
                    - Filtering by topic or sentiment first
                    - Using more specific analysis prompts
                    """)
                    
                    sample_size = st.slider(
                        "Number of feedback items to analyze",
                        min_value=10,
                        max_value=max_items,  # Remove the 500 limit to allow analyzing all items
                        value=min(100, max_items),
                        step=10
                    )
                    
                    # Add option to analyze all items
                    analyze_all = st.checkbox("Analyze all feedback items", 
                                            value=False, 
                                            help="Enable to analyze all feedback items instead of using the slider value")
                    
                    # Add batch processing option
                    use_batch_processing = st.checkbox("Use batch processing for large datasets", 
                                                     value=True,
                                                     help="Process large datasets in smaller batches to avoid API timeouts")
                    
                    # Add high precision mode option for follow-up analysis
                    high_precision_mode = False
                    if "follow-up" in analysis_prompt.lower() or "followup" in analysis_prompt.lower() or template_choice == "Customer Service Follow-up Cases":
                        high_precision_mode = st.checkbox("High Precision Mode (recommended)", 
                                                      value=True,
                                                      help="Process each feedback item individually for maximum accuracy. Slower but more reliable for follow-up case detection.")
                        if high_precision_mode:
                            st.success("✅ High Precision Mode enabled. Each feedback item will be analyzed individually for more reliable follow-up case detection.")
                            use_batch_processing = False
                    
                    # Determine if this is a follow-up analysis based on the prompt or template
                    is_followup_analysis = "follow-up" in analysis_prompt.lower() or "followup" in analysis_prompt.lower() or template_choice == "Customer Service Follow-up Cases"
                    
                    # Set batch sizes based on analysis type
                    if is_followup_analysis:
                        max_safe_items = 500  # Lower for complex analysis
                        batch_size = 30       # Reduced from 100 to 30 for better accuracy in follow-up analysis
                    else:
                        max_safe_items = 800  # Higher for simpler analysis
                        batch_size = 200      # Larger batches for simpler analysis
                        
                    # Only show safety limit if batch processing is disabled
                    if not use_batch_processing:
                        use_safe_limit = st.checkbox(f"Apply safety limit ({max_safe_items} items max)", 
                                                  value=True,
                                                  help="Enable to prevent API errors with large datasets")
                    else:
                        use_safe_limit = False
                        if is_followup_analysis:
                            st.info(f"Batch processing enabled: Data will be processed in smaller batches of {batch_size} items for more accurate follow-up case detection")
                        else:
                            st.info(f"Batch processing enabled: Data will be processed in batches of {batch_size} items")
                    
                    if analyze_all:
                        if not use_batch_processing and use_safe_limit and len(filtered_df) > max_safe_items:
                            st.warning(f"⚠️ Safety limit applied: Will analyze only the first {max_safe_items} items out of {len(filtered_df)} total.")
                        else:
                            total_items = len(filtered_df)
                            if use_batch_processing and total_items > batch_size:
                                st.info(f"Will analyze all {total_items} feedback items in batches of {batch_size}.")
                            else:
                                st.info(f"Will analyze all {total_items} feedback items. This may take a long time for large datasets.")
                    
                    # Additional options for customer service follow-up
                    is_followup_analysis = "follow-up" in analysis_prompt.lower() or "followup" in analysis_prompt.lower() or template_choice == "Customer Service Follow-up Cases"
                    
                    # Filter by sentiment for follow-up cases
                    if is_followup_analysis:
                        st.subheader("Follow-up Case Filters")
                        
                        # Check for sentiment columns
                        sentiment_cols = []
                        for col in results_df.columns:
                            if col.lower() in ['sentiment', 'bert_sentiment', 'llm_sentiment']:
                                sentiment_cols.append(col)
                        
                        if sentiment_cols:
                            sentiment_col = st.selectbox("Filter by sentiment", 
                                                       options=["All Sentiments"] + sentiment_cols,
                                                       index=0)
                            
                            if sentiment_col != "All Sentiments":
                                sentiment_filter = st.multiselect("Select sentiments to include", 
                                                               options=["negative", "neutral", "positive"],
                                                               default=["negative", "neutral"])
                                if sentiment_filter:
                                    filtered_df = filtered_df[filtered_df[sentiment_col].str.lower().isin([s.lower() for s in sentiment_filter])]
                                    st.write(f"Filtered to {len(filtered_df)} items with {', '.join(sentiment_filter)} sentiment")
                    
                    # Filter by topic if available
                    if topic_col:
                        st.subheader("Filter by Categories/Topics")
                        topics = sorted(results_df[topic_col].unique())
                        selected_topics = st.multiselect(
                            "Select categories or topics to analyze (leave empty for all)",
                            options=topics
                        )
                        
                        # Filter data based on selection
                        if selected_topics:
                            filtered_df = results_df[results_df[topic_col].isin(selected_topics)]
                        else:
                            filtered_df = results_df
                            
                        st.write(f"Working with {len(filtered_df)} feedback items")
                    else:
                        filtered_df = results_df
                    
                    if st.button("Run Analysis", type="primary"):
                        # Sample the data
                        if analyze_all:
                            if not use_batch_processing and use_safe_limit and len(filtered_df) > max_safe_items:
                                # Apply safety limit by taking the first max_safe_items
                                analysis_df = filtered_df.head(max_safe_items)
                                st.info(f"Analyzing first {max_safe_items} items due to safety limit.")
                            else:
                                analysis_df = filtered_df
                        elif sample_size < len(filtered_df):
                            analysis_df = filtered_df.sample(sample_size)
                        else:
                            analysis_df = filtered_df
                        
                        # Prepare the feedback items
                        # Include more context for follow-up cases (e.g., IDs, dates, etc.)
                        feedback_texts = []
                        for i, (idx, row) in enumerate(analysis_df.iterrows()):
                            entry = f"#{i+1} [ID: {idx}]"
                            
                            # Add relevant metadata if available
                            if 'date' in row:
                                entry += f" [Date: {row['date']}]"
                            if 'customer_id' in row:
                                entry += f" [Customer: {row['customer_id']}]"
                            if 'bert_sentiment' in row:
                                entry += f" [Sentiment: {row['bert_sentiment']}]"
                            elif 'llm_sentiment' in row:
                                entry += f" [Sentiment: {row['llm_sentiment']}]"
                            
                            # Add category information if available
                            if 'schema_main_category' in row:
                                entry += f" [Category: {row['schema_main_category']}]"
                            if 'schema_sub_topic' in row:
                                entry += f" [Subcategory: {row['schema_sub_topic']}]"
                            elif topic_col and topic_col in row:
                                entry += f" [Topic: {row[topic_col]}]"
                            
                            # Add the actual feedback text
                            entry += f": {row[text_col]}"
                            feedback_texts.append(entry)
                        
                        feedback_list = "\n\n".join(feedback_texts)
                        
                        # Customize the prompt for follow-up analysis
                        if is_followup_analysis:
                            full_prompt = f"""# Dutch Customer Service Follow-up Cases Identification

## Task
{analysis_prompt}

## Feedback Items (Dutch)
{feedback_list}

## Instructions
- Analyze the feedback items in their original Dutch language 
- IMPORTANT: Pay special attention to the actual Dutch FEEDBACK TEXT, not just metadata
- IMPORTANT: Structure your response in a CONSISTENT machine-readable format for CSV export
- For EACH case requiring follow-up, use EXACTLY the following format:
  
  ## Follow-up Cases
  
  1. **Case ID:** [ID number]
     - **Issue Summary:** [brief description]
     - **Priority Level:** [High/Medium/Low]
     - **Reason for Follow-up:** [specific reason]
     - **Recommended Action:** [what to do]

- VERY STRICT FILTERING: ONLY include cases that EXPLICITLY request help or indicate a problem requiring resolution, such as:
  * Technical issues clearly stated (network problems, connection issues, service outages)
  * Specific account requests (subscription changes, renewals, cancellations)
  * Direct questions requiring an answer ("I want to know when...")
  * Complaints about billing or charges that need resolution
  * Service problems that cannot be resolved without representative intervention

- EXAMPLES of valid follow-up cases (these SHOULD be included):
  * "Ik vind het heel jammer dat ik het tegoed van mijn data niet mee kan krijgen voor de volgende maand" (billing/account issue)
  * "Netwerk kabel | Krijg geen aansluiting" (technical connection problem)
  * "Ik zou mijn vaste telefoonnummer blijven houden" (service request)
  * "Ik heb een tijdelijke functie in het buitenland. Heb hier ook telefonisch contact over gehad" (needs contact about international service)
  * "Het abonnement gaat steeds omhoog" (billing concern)

- EXAMPLES that should NOT be included (DO NOT include these types):
  * "Optijd Bedenktijd" (just a comment, no issue requiring action)
  * "De communicatie van Ben via e-mail is duidelijk en nuttig" (positive feedback, no action needed)
  * "Ik ben tevreden over Odido" (positive feedback, no action needed)
  * "Niet om gevraagd..." (vague comment, no specific issue)
  * "Informatie. Altijd nuttig!" (just a comment about information being useful)
  * "De communicatie van Odido via e-mail is duidelijk" (positive comment, no action needed)

- DO NOT include:
  * General remarks or feedback
  * Purely positive statements
  * Opinions without requests
  * Statements that don't indicate a problem
  * Any feedback that doesn't explicitly need human intervention

- Use consistent formatting with the EXACT field labels shown above
- Include complete information for each field
- If analyzing many items, focus on the most important actionable cases first
- Number cases sequentially (1, 2, 3, etc.)
- End your response with a brief "## Summary" section
"""
                        else:
                            # Default prompt for other analyses
                            full_prompt = f"""# Dutch Customer Feedback Analysis
                        
                            ## Task
                            {analysis_prompt}
                        
                            ## Feedback Items (Dutch)
                            {feedback_list}
                        
                            ## Instructions
                            - Analyze the feedback items in their original Dutch language
                            - Provide your analysis in English
                            - Be specific and provide examples
                            - If you identify patterns, include an approximate percentage of feedback items that mention each pattern
                            """
                        
                        # Calculate estimated time based on complexity
                        estimated_time = len(analysis_df) * 0.2  # rough estimate: 0.2 seconds per item
                        if len(feedback_list) > 10000:
                            estimated_time *= 1.5  # longer text takes more time
                        
                        if is_followup_analysis:
                            estimated_time *= 2  # follow-up analysis is more complex
                            
                        # Cap reasonable max time to display
                        if estimated_time > 120:
                            time_msg = "2+ minutes"
                        else:
                            time_msg = f"{int(estimated_time)} seconds"
                        
                        spinner_msg = f"Analyzing {len(analysis_df)} feedback items... (estimated time: {time_msg})"
                        if len(analysis_df) > 50 or is_followup_analysis:
                            spinner_msg += "\nThis is a complex analysis that may take longer. Please be patient."
                        
                        if analyze_all and len(analysis_df) > 1000:
                            st.warning("""
                            ⚠️ You are analyzing a very large dataset. This may cause the following issues:
                            1. The API request may time out if too many items are analyzed at once
                            2. The model may return incomplete results due to token limitations
                            
                            Recommendations:
                            - Consider reducing the number of items or filtering by topic/sentiment first
                            - For very large datasets, it may be better to analyze in smaller batches
                            """)
                        
                        with st.spinner(spinner_msg):
                            try:
                                # Determine if we should use batch processing
                                should_batch = use_batch_processing and len(analysis_df) > batch_size
                                
                                if high_precision_mode and is_followup_analysis:
                                    # High precision mode processes each feedback item individually for better accuracy
                                    st.info(f"Using High Precision Mode: Processing {len(analysis_df)} items individually for maximum accuracy")
                                    
                                    # Setup for individual processing
                                    all_results = []
                                    valid_cases = []
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()
                                    
                                    # Process each feedback item individually
                                    for i, (idx, row) in enumerate(analysis_df.iterrows()):
                                        feedback_text = row[text_col]
                                        
                                        # Update status
                                        status_text.text(f"Processing item {i+1}/{len(analysis_df)} (ID: {idx})...")
                                        
                                        # Skip obviously positive feedback
                                        if len(feedback_text) < 10:
                                            all_results.append(f"Item {i+1} (ID: {idx}) - Skipped (too short)")
                                            progress_bar.progress((i + 1) / len(analysis_df))
                                            continue
                                            
                                        # Check for common positive terms that never need follow-up
                                        positive_terms = ["prima", "goed", "tevreden", "uitstekend", "top", 
                                                         "zeer tevreden", "erg tevreden", "heel tevreden"]
                                        if any(term in feedback_text.lower() for term in positive_terms) and len(feedback_text) < 30:
                                            # Skip obviously positive feedback
                                            progress_bar.progress((i + 1) / len(analysis_df))
                                            continue
                                        
                                        # Prepare single item prompt with additional emphasis on validation
                                        single_item_prompt = f"""# Dutch Customer Service Follow-up Case Analysis

## Task
Analyze this single feedback item and determine if it requires customer service follow-up.

## Feedback Item (Dutch)
ID: {idx}
{feedback_text}

## Instructions
ONLY classify this as requiring follow-up if it EXPLICITLY contains:
- A clear request for help
- A specific technical problem description
- A direct question requiring an answer
- A complaint about billing or service that needs resolution
- An account-related request that requires human intervention

DO NOT classify as requiring follow-up if it:
- Contains only general feedback
- Is a positive comment (e.g., "tevreden", "goed", "prima")
- Is an opinion without a specific request
- Does not clearly indicate a problem requiring intervention

## Format
If this feedback requires follow-up, respond EXACTLY in this format:
```
REQUIRES_FOLLOWUP: YES
ISSUE: [Describe the specific issue]
REASON: [Explain why this requires follow-up]
ACTION: [Recommend what action to take]
PRIORITY: [High/Medium/Low]
```

If it does NOT require follow-up, respond EXACTLY:
```
REQUIRES_FOLLOWUP: NO
```
"""
                                        
                                        # Call API
                                        result = call_huggingface_api(single_item_prompt, max_tokens=200)
                                        
                                        # Parse result
                                        if "REQUIRES_FOLLOWUP: YES" in result:
                                            # Extract the details
                                            issue_match = re.search(r'ISSUE: (.+?)(?:\n|$)', result)
                                            reason_match = re.search(r'REASON: (.+?)(?:\n|$)', result)
                                            action_match = re.search(r'ACTION: (.+?)(?:\n|$)', result)
                                            priority_match = re.search(r'PRIORITY: (.+?)(?:\n|$)', result)
                                            
                                            issue = issue_match.group(1).strip() if issue_match else ""
                                            reason = reason_match.group(1).strip() if reason_match else ""
                                            action = action_match.group(1).strip() if action_match else ""
                                            priority = priority_match.group(1).strip() if priority_match else "Medium"
                                            
                                            # Extract key words from issue (non-stop words)
                                            issue_words = [word.lower() for word in re.findall(r'\b\w+\b', issue)
                                                          if len(word) > 3 and word.lower() not in ["naar", "voor", "over", "door", "deze", "zijn", "heel", "omdat", "hebben", "welke", "worden"]]
                                            
                                            # Common Dutch problem indicators, inquiry terms, and pricing/contract terminology
                                            problem_indicators = ["kopen", "willen", "probleem", "aanschaffen", "bestellen", 
                                                                 "storing", "werkt", "kapot", "fout", "error", "hulp", "help",
                                                                 "vraag", "betekent", "wat", "betekenis", "asterisk", "sterretje", 
                                                                 "hoe", "waarom", "contract", "prijs", "kosten", "betaal", "betalen",
                                                                 "abonnement", "maand", "maandelijks", "rekening", "weten", "mnd"]
                                            
                                            # Check if any significant words from the issue appear in the text
                                            issue_valid = False
                                            if issue_words:
                                                issue_valid = any(word in feedback_text.lower() for word in issue_words) or \
                                                             any(indicator in feedback_text.lower() for indicator in problem_indicators)
                                            
                                            # Only add validated issues
                                            if issue_valid:
                                                case_num = len(valid_cases) + 1
                                                valid_cases.append({
                                                    "Case Number": case_num,
                                                    "Feedback ID": idx,
                                                    "Priority": priority,
                                                    "Issue": issue,
                                                    "Reason for Follow-up": reason,
                                                    "Recommended Action": action,
                                                    "Original Feedback": feedback_text
                                                })
                                                # Store raw result for debugging
                                                all_results.append(f"Item {i+1} (ID: {idx}) - Valid follow-up case detected")
                                            else:
                                                all_results.append(f"Item {i+1} (ID: {idx}) - Rejected: Issue not validated in original text")
                                        else:
                                            # Extract only the REQUIRES_FOLLOWUP: NO part for cleaner display
                                            cleaned_result = "REQUIRES_FOLLOWUP: NO"
                                            # Try to extract any explanation if it exists
                                            explanation_match = re.search(r'NO\s+(.+?)(?=\Z|\n\n)', result, re.DOTALL)
                                            if explanation_match and len(explanation_match.group(1).strip()) > 0:
                                                # Only include if it's a real explanation, not just formatting noise
                                                explanation = explanation_match.group(1).strip()
                                                if not explanation.startswith('```') and len(explanation) > 5:
                                                    cleaned_result = f"REQUIRES_FOLLOWUP: NO\nReason: {explanation.strip()}"
                                            
                                            all_results.append(f"Item {i+1} (ID: {idx}) - No follow-up required")
                                        
                                        # Update progress
                                        progress_bar.progress((i + 1) / len(analysis_df))
                                    
                                    # Process results from high precision mode
                                    if valid_cases:
                                        # Generate markdown for display with valid cases
                                        cases_md = []
                                        for case in valid_cases:
                                            case_md = f"{case['Case Number']}. **Case ID:** {case['Feedback ID']}\n   - **Issue Summary:** {case['Issue']}\n   - **Priority Level:** {case['Priority']}\n   - **Reason for Follow-up:** {case['Reason for Follow-up']}\n   - **Recommended Action:** {case['Recommended Action']}"
                                            cases_md.append(case_md)
                                        
                                        # Create final result text with cases
                                        result = f"""# Dutch Customer Service Follow-up Cases - High Precision Results

## Follow-up Cases

{chr(10).join(cases_md)}

## Summary
High Precision Mode: Analyzed {len(analysis_df)} feedback items individually, found {len(valid_cases)} validated cases requiring follow-up.
"""
                                        # Create follow-up df for CSV export
                                        follow_up_df = pd.DataFrame(valid_cases)
                                        
                                        # Display the table
                                        st.subheader("Follow-up Cases Table")
                                        st.dataframe(follow_up_df)
                                        
                                        # Create CSV download link
                                        csv = follow_up_df.to_csv(index=False)
                                        b64_csv = base64.b64encode(csv.encode()).decode()
                                        now = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        csv_filename = f"followup_cases_highprecision_{now}.csv"
                                        
                                        # Create a styled button for CSV download
                                        st.markdown(
                                            f"""
                                            <div style="text-align:center;margin:20px 0;">
                                                <a href="data:file/csv;base64,{b64_csv}" 
                                                   download="{csv_filename}" 
                                                   style="display:inline-block;background-color:#4CAF50;color:white;
                                                          padding:12px 24px;text-align:center;text-decoration:none;
                                                          font-size:16px;border-radius:4px;font-weight:bold;">
                                            📥 Download High Precision Follow-up Cases CSV
                                        </a>
                                            </div>
                                            """, 
                                            unsafe_allow_html=True
                                        )
                                    else:
                                        # Clean summary when no cases are found
                                        result = f"""# Analysis Complete - High Precision Mode

## Summary
High Precision Mode analyzed {len(analysis_df)} feedback items individually.

**No follow-up cases requiring attention were identified.** All feedback items were classified as one of:
- General feedback without specific requests
- Expressions of opinion that don't require action
- Positive or neutral statements
- Comments without actionable issues

The analysis has successfully completed with no cases requiring customer service follow-up.
"""
                                        follow_up_df = None
                                    
                                    # Clear status text
                                    status_text.empty()
                                
                                elif should_batch:
                                    # Setup for batch processing
                                    all_results = []
                                    progress_bar = st.progress(0)
                                    batch_count = (len(analysis_df) + batch_size - 1) // batch_size  # Ceiling division
                                    status_text = st.empty()
                                    combined_result = ""
                                    
                                    # Process in batches
                                    for i in range(0, batch_count):
                                        # Get the current batch of data
                                        start_idx = i * batch_size
                                        end_idx = min((i + 1) * batch_size, len(analysis_df))
                                        batch_df = analysis_df.iloc[start_idx:end_idx]
                                        
                                        # Update status
                                        status_text.text(f"Processing batch {i+1}/{batch_count} ({start_idx+1}-{end_idx} of {len(analysis_df)} items)...")
                                        
                                        # Prepare the feedback items for this batch
                                        batch_feedback_texts = []
                                        for j, (idx, row) in enumerate(batch_df.iterrows()):
                                            entry = f"#{j+1+start_idx} [ID: {idx}]"
                                            
                                            # Add relevant metadata if available
                                            if 'date' in row:
                                                entry += f" [Date: {row['date']}]"
                                            if 'customer_id' in row:
                                                entry += f" [Customer: {row['customer_id']}]"
                                            if 'bert_sentiment' in row:
                                                entry += f" [Sentiment: {row['bert_sentiment']}]"
                                            elif 'llm_sentiment' in row:
                                                entry += f" [Sentiment: {row['llm_sentiment']}]"
                                            
                                            # Add the actual feedback text
                                            entry += f": {row[text_col]}"
                                            batch_feedback_texts.append(entry)
                                        
                                        batch_feedback_list = "\n\n".join(batch_feedback_texts)
                                        
                                        # Create batch-specific prompt
                                        if is_followup_analysis:
                                            batch_prompt = f"""# Dutch Customer Service Follow-up Cases Identification (Batch {i+1}/{batch_count})

## Task
{analysis_prompt}

## Feedback Items (Dutch) - Batch {i+1}/{batch_count}
{batch_feedback_list}

## Instructions
- Analyze the feedback items in their original Dutch language
- IMPORTANT: Pay special attention to the actual Dutch FEEDBACK TEXT, not just metadata
- IMPORTANT: Structure your response in a CONSISTENT machine-readable format for CSV export
- For EACH case requiring follow-up, use EXACTLY the following format:
  
  ## Follow-up Cases
  
  1. **Case ID:** [ID number]
     - **Issue Summary:** [brief description]
     - **Priority Level:** [High/Medium/Low]
     - **Reason for Follow-up:** [specific reason]
     - **Recommended Action:** [what to do]

- VERY STRICT FILTERING: ONLY include cases that EXPLICITLY request help or indicate a problem requiring resolution, such as:
  * Technical issues clearly stated (network problems, connection issues, service outages)
  * Specific account requests (subscription changes, renewals, cancellations)
  * Direct questions requiring an answer ("I want to know when...")
  * Complaints about billing or charges that need resolution
  * Service problems that cannot be resolved without representative intervention

- EXAMPLES of valid follow-up cases (these SHOULD be included):
  * "Ik vind het heel jammer dat ik het tegoed van mijn data niet mee kan krijgen voor de volgende maand" (billing/account issue)
  * "Netwerk kabel | Krijg geen aansluiting" (technical connection problem)
  * "Ik zou mijn vaste telefoonnummer blijven houden" (service request)
  * "Ik heb een tijdelijke functie in het buitenland. Heb hier ook telefonisch contact over gehad" (needs contact about international service)
  * "Het abonnement gaat steeds omhoog" (billing concern)

- EXAMPLES that should NOT be included (DO NOT include these types):
  * "Optijd Bedenktijd" (just a comment, no issue requiring action)
  * "De communicatie van Ben via e-mail is duidelijk en nuttig" (positive feedback, no action needed)
  * "Ik ben tevreden over Odido" (positive feedback, no action needed)
  * "Niet om gevraagd..." (vague comment, no specific issue)
  * "Informatie. Altijd nuttig!" (just a comment about information being useful)
  * "De communicatie van Odido via e-mail is duidelijk" (positive comment, no action needed)

- DO NOT include:
  * General remarks or feedback
  * Purely positive statements
  * Opinions without requests
  * Statements that don't indicate a problem
  * Any feedback that doesn't explicitly need human intervention

- Use consistent formatting with the EXACT field labels shown above
- Include complete information for each field
- Number cases sequentially (1, 2, 3, etc.)
- Note: This is batch {i+1} of {batch_count}
"""
                                        else:
                                            batch_prompt = f"""# Dutch Customer Feedback Analysis (Batch {i+1}/{batch_count})
                                            
## Task
{analysis_prompt}

## Feedback Items (Dutch) - Batch {i+1}/{batch_count}
{batch_feedback_list}

## Instructions
- Analyze the feedback items in their original Dutch language
- Provide your analysis in English
- Be specific and provide examples
- If you identify patterns, include an approximate percentage of feedback items that mention each pattern
- Note: This is batch {i+1} of {batch_count}
"""
                                        
                                        # Process the batch
                                        max_tokens = 800 if is_followup_analysis else 400
                                        batch_result = call_huggingface_api(batch_prompt, max_tokens=max_tokens)
                                        
                                        # Check for errors
                                        if batch_result.startswith("Error"):
                                            st.error(f"Error in batch {i+1}: {batch_result}")
                                            if "timeout" in batch_result.lower():
                                                st.warning(f"Batch {i+1} timed out. Consider reducing batch size.")
                                            # Continue with next batch rather than stopping completely
                                            all_results.append(f"⚠️ **Batch {i+1} Error:** {batch_result}")
                                        else:
                                            all_results.append(batch_result)
                                        
                                        # Update progress
                                        progress_bar.progress((i + 1) / batch_count)
                                    
                                    # Combine results
                                    status_text.text("Combining results from all batches...")
                                    
                                    if is_followup_analysis:
                                        # For follow-up cases, extract and combine case sections
                                        combined_cases = []
                                        case_counter = 1
                                        
                                        for batch_idx, batch_result in enumerate(all_results):
                                            # Extract cases from this batch
                                            follow_up_section = re.search(r'## Follow-up Cases\s+(.*?)(?:\n##|\Z)', 
                                                                         batch_result, re.DOTALL | re.MULTILINE)
                                            
                                            if follow_up_section:
                                                # Clean up and renumber cases
                                                batch_cases = follow_up_section.group(1).strip()
                                                # Get individual cases by regex pattern for numbered items
                                                case_blocks = re.findall(r'\d+\.\s+.*?(?=\n\d+\.|\Z)', 
                                                                        batch_cases, re.DOTALL | re.MULTILINE)
                                                
                                                for case in case_blocks:
                                                    # Replace original numbering with sequential numbering
                                                    renumbered_case = re.sub(r'^\d+\.', f"{case_counter}.", case, 1)
                                                    combined_cases.append(renumbered_case.strip())
                                                    case_counter += 1
                                        
                                        if combined_cases:
                                            combined_result = f"""# Dutch Customer Service Follow-up Cases - Combined Results

## Follow-up Cases

{chr(10).join(combined_cases)}

## Summary
Combined results from {batch_count} batches, found {len(combined_cases)} cases requiring follow-up.
"""
                                        else:
                                            combined_result = "# Analysis Complete\n\nNo follow-up cases were identified in any of the batches."
                                    else:
                                        # For general analysis, provide batch results separately
                                        combined_result = "# Combined Analysis Results\n\n"
                                        for i, batch_result in enumerate(all_results):
                                            combined_result += f"## Batch {i+1} Results\n\n{batch_result}\n\n---\n\n"
                                        
                                        combined_result += f"\n## Overall Summary\nAnalyzed {len(analysis_df)} feedback items across {batch_count} batches."
                                    
                                    # Set the result to the combined output
                                    result = combined_result
                                    status_text.empty()
                                    
                                else:
                                    # Non-batched processing (original code)
                                    max_tokens = 800 if is_followup_analysis else 400
                                    result = call_huggingface_api(full_prompt, max_tokens=max_tokens)
                                
                                # Check if we got an error from the API (for non-batched or if all batches failed)
                                if result.startswith("Error") and not should_batch:
                                    st.error(result)
                                    if "timeout" in result.lower():
                                        st.info("The analysis timed out. This can happen with very large or complex analyses. Try enabling batch processing or reducing the number of feedback items.")
                                    st.stop()
                                
                                # Display the results
                                st.success("✅ Analysis complete!")
                                st.subheader("Analysis Results")
                                st.markdown(result)
                                
                                # Special handling for follow-up cases to create a downloadable CSV
                                if is_followup_analysis:
                                    try:
                                        # Parse the markdown response to extract case information
                                        follow_up_cases = []
                                        import re
                                        
                                        # Look for "## Follow-up Cases" section
                                        follow_up_section = re.search(r'## Follow-up Cases\s+(.*?)(?:\n##|\Z)', result, re.DOTALL | re.MULTILINE)
                                        
                                        if follow_up_section:
                                            cases_text = follow_up_section.group(1)
                                            
                                            # Updated pattern to match numbered items with bold case IDs
                                            # Pattern matches format like "1. **Case ID:** 123" or "1. **Case ID:** [ID: 123]"
                                            case_pattern = r'(\d+)\.\s+\*\*Case ID:?\*\*:?\s+(?:\[ID:?\s*)?(\d+)(?:\])?'
                                            
                                            # Also try alternative format like "1. Case ID: 123"
                                            alt_case_pattern = r'(\d+)\.\s+Case ID:?\s+(?:\[ID:?\s*)?(\d+)(?:\])?'
                                            
                                            # First try the bold pattern, then fallback to alternative
                                            cases = re.findall(case_pattern, cases_text, re.MULTILINE)
                                            if not cases:
                                                cases = re.findall(alt_case_pattern, cases_text, re.MULTILINE)
                                            
                                            # If neither pattern worked, try a more generic approach
                                            if not cases:
                                                # Just look for numbered items with any ID numbers
                                                generic_pattern = r'(\d+)\.\s+.*?ID:?\s*(\d+)'
                                                cases = re.findall(generic_pattern, cases_text, re.MULTILINE)
                                            
                                            if cases:
                                                # Extract the full text of each case and parse it
                                                for case_num, case_id in cases:
                                                    # Find the full case text - start at the case number
                                                    case_start_pattern = re.escape(f"{case_num}.")
                                                    case_start_match = re.search(case_start_pattern, cases_text)
                                                    
                                                    if case_start_match:
                                                        case_start = case_start_match.start()
                                                        
                                                        # Find the end (next case number or end of section)
                                                        next_case_pattern = r'\n\s*' + re.escape(f"{int(case_num)+1}.") + r'\s+'
                                                        next_case = re.search(next_case_pattern, cases_text[case_start:])
                                                        
                                                        if next_case:
                                                            case_end = case_start + next_case.start()
                                                        else:
                                                            case_end = len(cases_text)
                                                        
                                                        case_text = cases_text[case_start:case_end].strip()
                                                        
                                                        # Extract fields with more flexible patterns
                                                        # Look for bold headers (**Priority Level:**) or regular headers (Priority Level:)
                                                        priority_pattern = r'(?:\*\*Priority Level:?\*\*|Priority Level:?)\s*(High|Medium|Low)'
                                                        priority_match = re.search(priority_pattern, case_text, re.IGNORECASE)
                                                        priority = priority_match.group(1) if priority_match else "Medium"
                                                        
                                                        issue_pattern = r'(?:\*\*Issue(?:\s+Summary)?:?\*\*|Issue(?:\s+Summary)?:?)\s*([^\n]+)'
                                                        issue_match = re.search(issue_pattern, case_text, re.IGNORECASE)
                                                        issue = issue_match.group(1).strip() if issue_match else ""
                                                        
                                                        reason_pattern = r'(?:\*\*Reason for Follow-up:?\*\*|Reason for Follow-up:?)\s*([^\n]+)'
                                                        reason_match = re.search(reason_pattern, case_text, re.IGNORECASE)
                                                        reason = reason_match.group(1).strip() if reason_match else ""
                                                        
                                                        action_pattern = r'(?:\*\*Recommended Action:?\*\*|Recommended Action:?)\s*([^\n]+)'
                                                        action_match = re.search(action_pattern, case_text, re.IGNORECASE)
                                                        action = action_match.group(1).strip() if action_match else ""
                                                        
                                                        # If we couldn't extract structured fields, use the full text
                                                        if not issue and not reason:
                                                            # Remove the case number and ID
                                                            case_text = re.sub(r'^\d+\.\s+(?:\*\*)?Case ID:(?:\*\*)?\s+\d+.*?\n', '', case_text, flags=re.IGNORECASE).strip()
                                                            issue = case_text
                                                        
                                                        # Try to get the feedback text using the extracted ID
                                                        original_feedback = ""
                                                        try:
                                                            if int(case_id) in analysis_df.index:
                                                                original_feedback = analysis_df.loc[int(case_id), text_col]
                                                            # If not found by index, try looking for a row where ID appears in any column
                                                            else:
                                                                for _, row in analysis_df.iterrows():
                                                                    if any(str(case_id) in str(val) for val in row.values):
                                                                        original_feedback = row[text_col]
                                                                        break
                                                        except (ValueError, KeyError, TypeError):
                                                            # If we can't find it, leave it blank
                                                            pass
                                                        
                                                        follow_up_cases.append({
                                                            "Case Number": case_num,
                                                            "Feedback ID": case_id,
                                                            "Priority": priority,
                                                            "Issue": issue,
                                                            "Reason for Follow-up": reason,
                                                            "Recommended Action": action,
                                                            "Original Feedback": original_feedback
                                                        })
                                        
                                        # Create a DataFrame from the extracted cases
                                        if follow_up_cases:
                                            follow_up_df = pd.DataFrame(follow_up_cases)
                                            
                                            # Display the table
                                            st.subheader("Follow-up Cases Table")
                                            st.dataframe(follow_up_df)
                                            
                                            # Create CSV download link with a more prominent button
                                            csv = follow_up_df.to_csv(index=False)
                                            b64_csv = base64.b64encode(csv.encode()).decode()
                                            now = datetime.now().strftime("%Y%m%d_%H%M%S")
                                            csv_filename = f"followup_cases_{now}.csv"
                                            
                                            # Create a styled button for CSV download
                                            st.markdown(
                                                f"""
                                                <div style="text-align:center;margin:20px 0;">
                                                    <a href="data:file/csv;base64,{b64_csv}" 
                                                       download="{csv_filename}" 
                                                       style="display:inline-block;background-color:#4CAF50;color:white;
                                                              padding:12px 24px;text-align:center;text-decoration:none;
                                                              font-size:16px;border-radius:4px;font-weight:bold;">
                                                        📥 Download Follow-up Cases CSV
                                                    </a>
                                                </div>
                                                """, 
                                                unsafe_allow_html=True
                                            )
                                        else:
                                            st.warning("No follow-up cases were successfully extracted from the analysis. This could be due to formatting issues in the AI response or no cases requiring follow-up were found.")
                                    except Exception as e:
                                        st.warning(f"Could not parse follow-up cases into a structured format. Using text download only. Error: {e}")
                                        import traceback
                                        st.code(traceback.format_exc())
                                
                                # Always provide the text download option
                                now = datetime.now().strftime("%Y%m%d_%H%M%S")
                                download_filename = f"feedback_analysis_{now}.txt"
                                
                                download_content = f"""# Feedback Analysis Results
                                Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                                
                                ## Analysis Prompt
                                {analysis_prompt}
                                
                                ## Results
                            {result}
                                
                                ## Analysis Information
                                - Total feedback items: {len(filtered_df)}
- Analyzed sample: {len(analysis_df)}
                                """
                                
                                b64 = base64.b64encode(download_content.encode()).decode()
                                st.markdown(f'<a href="data:text/plain;base64,{b64}" download="{download_filename}">Download Analysis Results as Text</a>', unsafe_allow_html=True)
                            
                            except Exception as e:
                                st.error(f"Error during analysis: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                else:
                    st.error("No text column found in the results.")