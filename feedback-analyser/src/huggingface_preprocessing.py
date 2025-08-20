import pandas as pd
import numpy as np
import json
import requests
import time
import os
from tqdm.auto import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("huggingface_preprocessing.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("Huggingface Preprocessing")

# Track endpoint initialization state
ENDPOINT_INITIALIZED = False
INIT_RETRIES = 5  # Default number of initialization retries

def generate_comment_for_missing(score, reason, business_unit):
    """
    Generate a generic comment for missing comment based on score and other metadata.
    
    Args:
        score: Customer score (2.0 = negative, 8.0 = positive)
        reason: Reason field from feedback (e.g., 'Brand', 'MAIL', etc.)
        business_unit: Business unit field (e.g., 'B2C - ODIDO - MOBILE_FIXED')
        
    Returns:
        A generated comment that can be used for training
    """
    # Extract brand from business unit
    brand = "Odido"  # Default
    if "BEN" in business_unit:
        brand = "Ben"
    elif "TELE2" in business_unit:
        brand = "Tele2"
    elif "T-MOBILE" in business_unit:
        brand = "T-Mobile"
    
    if score == 2.0:  # Negative
        if reason == "Brand":
            return f"Ik ben niet tevreden over {brand} als merk."
        elif reason == "MAIL":
            return f"Ik vind de mails van {brand} niet relevant voor mij."
        elif reason == "ANDERS":
            return f"Ik ben ontevreden over de dienstverlening van {brand}."
        else:
            return f"Ik ben niet tevreden over {brand}."
    
    elif score == 8.0:  # Positive
        if reason == "Brand":
            return f"Ik ben erg tevreden over {brand} als merk."
        elif reason == "MAIL":
            return f"De communicatie van {brand} via e-mail is duidelijk en nuttig."
        elif reason == "ANDERS":
            return f"Ik ben tevreden over de dienstverlening van {brand}."
        else:
            return f"Ik ben tevreden over {brand}."
    
    else:  # Neutral or unknown
        return f"Geen specifieke mening over {brand}."

def analyze_sentiment_hf(text, endpoint_url, api_key, retries=15, initial_delay=5):
    """
    Analyze sentiment of a text using Hugging Face endpoint.
    
    Args:
        text: Text to analyze
        endpoint_url: Hugging Face endpoint URL
        api_key: Hugging Face API key
        retries: Number of retries
        initial_delay: Initial delay between retries in seconds
        
    Returns:
        Dict with sentiment and confidence
    """
    # Default result
    result = {"sentiment": "neutraal", "confidence": 0.5}
    
    global ENDPOINT_INITIALIZED
    
    # Prepare prompt for the model - MODIFIED FOR BETTER ACCURACY
    prompt = f"""
Analyseer het sentiment van de volgende Nederlandse klantfeedback. Geef alleen het sentiment terug als één woord: positief, negatief, of neutraal. Let goed op de context en de algehele boodschap, inclusief klachten, problemen, gebroken beloftes, en de wens om over te stappen.

Hier zijn enkele voorbeelden:
Tekst: "Alles werkt perfect, heel tevreden."
Sentiment: positief
Tekst: "Beloftes worden niet nagekomen."
Sentiment: negatief
Tekst: "Niemand gezien, heel weekend zonder tv, ze hebben schijt aan je, wil andere provider."
Sentiment: negatief
Tekst: "Ik heb een vraag over mijn factuur."
Sentiment: neutraal

Analyseer nu de volgende tekst:
Tekst: \"{text}\"
Sentiment:"""
    
    # Ensure proper API key format - it should start with 'hf_'
    if api_key and not api_key.startswith("hf_"):
        api_key = f"hf_{api_key}"  # Add prefix if missing
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "inputs": prompt
    }
    
    # Log headers and payload for debugging (omit actual API key)
    logger.info(f"Sending request to: {endpoint_url}")
    logger.info(f"Headers: {{'Content-Type': 'application/json', 'Authorization': 'Bearer hf_***'}}")
    
    # Special handling for first-time initialization
    if not ENDPOINT_INITIALIZED:
        logger.info("First request to endpoint - initializing...")
        logger.info("This may take several minutes as the model loads for the first time")
        
        global INIT_RETRIES
        init_retries = INIT_RETRIES if 'INIT_RETRIES' in globals() else 5
        
        # Try multiple times with increasing delays to let the endpoint initialize
        for init_attempt in range(init_retries):  # Use INIT_RETRIES instead of hard-coded 5
            try:
                init_delay = 30 + (init_attempt * 30)  # Start with 30s, then 60s, 90s, etc.
                logger.info(f"Initialization attempt {init_attempt+1}/{init_retries} (will wait {init_delay}s for endpoint)")
                
                # Send a wake-up request
                init_response = requests.post(
                    endpoint_url, 
                    headers=headers, 
                    json={"inputs": "Test request to initialize endpoint"}, 
                    timeout=60
                )
                
                logger.info(f"Initialization response status: {init_response.status_code}")
                
                if init_response.status_code == 200:
                    logger.info("Endpoint successfully initialized!")
                    ENDPOINT_INITIALIZED = True
                    break
                else:
                    logger.info(f"Endpoint not ready yet. Waiting {init_delay} seconds...")
                    time.sleep(init_delay)
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Initialization request error: {str(e)}")
                logger.info(f"Waiting {init_delay} seconds before retry...")
                time.sleep(init_delay)
    
    # Try to get response with retries and exponential backoff
    for attempt in range(retries):
        try:
            # Calculate exponential backoff delay
            delay = initial_delay * (2 ** attempt) if attempt > 0 else initial_delay
            delay = min(delay, 120)  # Cap at 120 seconds max delay
            
            response = requests.post(endpoint_url, headers=headers, json=payload, timeout=60)
            
            # Log response status for debugging
            logger.info(f"Response status: {response.status_code}")
            
            # Handle different response codes
            if response.status_code == 200:
                # Extract the sentiment from the response
                try:
                    full_text = response.json()[0]["generated_text"]
                    # Remove the prompt and extract only the model's reply after the last 'Sentiment:'
                    logger.info(f"Full response text: {full_text}")
                    if "Sentiment:" in full_text:
                        # Split on the last occurrence to get the generated sentiment label
                        generated = full_text.split("Sentiment:")[-1].strip()
                    else:
                        generated = full_text.strip()
                    logger.info(f"Model output after prompt split: {generated}")
                    # Lowercase for keyword matching
                    response_lower = generated.lower()
                    
                    if "positief" in response_lower:
                        result = {"sentiment": "positief", "confidence": 0.8}
                    elif "negatief" in response_lower:
                        result = {"sentiment": "negatief", "confidence": 0.8}
                    else:
                        result = {"sentiment": "neutraal", "confidence": 0.6}
                    
                    ENDPOINT_INITIALIZED = True  # Mark as initialized for future requests
                    return result
                except (IndexError, KeyError, json.JSONDecodeError) as e:
                    logger.warning(f"Error parsing response (attempt {attempt+1}/{retries}): {str(e)}")
                    logger.warning(f"Raw response: {response.text[:500]}")
            
            elif response.status_code == 401:
                logger.warning(f"Authentication error (attempt {attempt+1}/{retries}): Unauthorized - check your API key")
            
            elif response.status_code == 403:
                logger.warning(f"Authorization error (attempt {attempt+1}/{retries}): Forbidden - no access to this endpoint")
            
            elif response.status_code == 404:
                logger.warning(f"Endpoint error (attempt {attempt+1}/{retries}): Not Found - check your endpoint URL")
            
            elif response.status_code == 503:
                logger.warning(f"Service Unavailable (attempt {attempt+1}/{retries}): Endpoint is still initializing")
                logger.info(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)
                continue
            
            else:
                logger.warning(f"Error (attempt {attempt+1}/{retries}): Status code {response.status_code} - {response.text[:200]}")
            
            if attempt < retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error (attempt {attempt+1}/{retries}): {str(e)}")
            if attempt < retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
    
    # If all attempts fail, use score-based approach instead
    logger.warning(f"All sentiment analysis attempts failed, defaulting to score-based sentiment")
    return result

def map_sentiment_to_score(sentiment):
    """
    Map sentiment string to a numeric score between -1 and 1
    
    Args:
        sentiment: Sentiment string (positief, negatief, neutraal)
        
    Returns:
        Float between -1 and 1
    """
    if sentiment == "positief":
        return 0.8
    elif sentiment == "negatief":
        return -0.8
    else:  # neutraal or gemengd
        return 0.0

def map_numeric_score_to_sentiment(score):
    """
    Map numeric customer score to sentiment
    
    Args:
        score: Customer score (e.g., 2.0 or 8.0)
        
    Returns:
        Sentiment string and score
    """
    if score == 8.0:
        return "positief", 0.8
    elif score == 2.0:
        return "negatief", -0.8
    else:
        return "neutraal", 0.0

def preprocess_feedback(
    input_file, 
    output_file, 
    hf_endpoint_url=None,
    hf_api_key=None,
    batch_size=50, 
    force_reprocess=False,
    use_api_for_missing=False,  # Whether to use the HF API for missing comments or use the generated ones
    offline_mode=False,  # New parameter to skip API calls completely
    max_init_wait=300,  # Maximum seconds to wait for endpoint initialization
    init_retries=5  # Number of retries for initialization
):
    """
    Preprocess Dutch customer feedback using Hugging Face for sentiment analysis.
    Handle missing comments by using the score to determine sentiment.
    
    Args:
        input_file: Path to Excel file containing feedback
        output_file: Path to save preprocessed data
        hf_endpoint_url: Hugging Face endpoint URL (optional in offline mode)
        hf_api_key: Hugging Face API key (optional in offline mode)
        batch_size: Number of rows to process in each batch
        force_reprocess: Whether to force reprocessing if output file exists
        use_api_for_missing: Whether to use the API for generated comments or just assign sentiment
        offline_mode: Whether to skip API calls and use score-based sentiment for all data
        max_init_wait: Maximum seconds to wait for endpoint initialization
        init_retries: Number of retries for initialization
        
    Returns:
        DataFrame with preprocessed data
    """
    # Check if endpoint and API key are provided when not in offline mode
    if not offline_mode and (not hf_endpoint_url or not hf_api_key):
        logger.warning("Hugging Face endpoint URL or API key not provided. Switching to offline mode.")
        offline_mode = True
    
    # Check if output file already exists
    if os.path.exists(output_file) and not force_reprocess:
        logger.info(f"Loading preprocessed data from {output_file}")
        return pd.read_csv(output_file)
    
    # Load data
    logger.info(f"Loading data from {input_file}")
    
    # Handle different file extensions
    if input_file.endswith('.xlsx') or input_file.endswith('.xls'):
        df = pd.read_excel(input_file)
    elif input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    else:
        logger.error(f"Unsupported file format: {input_file}")
        raise ValueError(f"Unsupported file format: {input_file}. Use .xlsx, .xls, or .csv")
        
    logger.info(f"Loaded {len(df)} feedback entries")
    
    # Log processing mode
    if offline_mode:
        logger.info("Running in offline mode - using score-based sentiment for all entries")
    else:
        logger.info(f"Using Hugging Face endpoint for sentiment analysis: {hf_endpoint_url}")
    
    # Store all processed data
    all_processed_data = []
    
    # Test API connection if not in offline mode
    if not offline_mode:
        logger.info("Testing API connection...")
        
        # Set the global initialization retries
        global INIT_RETRIES
        INIT_RETRIES = init_retries
        logger.info(f"Setting endpoint initialization retries to {INIT_RETRIES}")
        
        # Set a start time to track initialization time
        start_time = time.time()
        
        # Try to initialize the endpoint
        test_result = analyze_sentiment_hf("Dit is een test bericht", hf_endpoint_url, hf_api_key)
        
        # Check elapsed time
        elapsed_time = time.time() - start_time
        
        # If initialization took more than max_init_wait or failed
        if elapsed_time > max_init_wait:
            logger.warning(f"Endpoint initialization exceeded max wait time of {max_init_wait} seconds")
            logger.warning("Switching to offline mode to proceed with processing")
            offline_mode = True
        elif test_result["sentiment"] == "neutraal" and test_result["confidence"] == 0.5:
            logger.warning("API test failed to return meaningful results")
            logger.warning("Switching to offline mode to proceed with processing")
            offline_mode = True
        else:
            logger.info(f"API test successful after {elapsed_time:.1f} seconds: {test_result}")
            logger.info("Continuing with online mode using Hugging Face endpoint")
    
    # Process in batches
    for i in tqdm(range(0, len(df), batch_size), desc="Processing feedback batches"):
        batch = df.iloc[i:i+batch_size]
        
        for idx, row in batch.iterrows():
            # Extract necessary fields
            comment = row.get('Comment', None)
            score = row.get('Score', None)
            reason = row.get('Reason', '')
            business_unit = row.get('Business Unit', '')
            
            # Skip entries with missing score
            if pd.isna(score):
                continue
                
            # Handle missing comments
            if pd.isna(comment) or comment is None or comment == '':
                # Generate a comment based on score and metadata
                generated_comment = generate_comment_for_missing(score, reason, business_unit)
                
                # Decide how to process the generated comment
                if not offline_mode and use_api_for_missing:
                    # Use the Hugging Face API to analyze the generated comment
                    try:
                        sentiment_result = analyze_sentiment_hf(generated_comment, hf_endpoint_url, hf_api_key)
                        sentiment = sentiment_result["sentiment"]
                        sentiment_score = map_sentiment_to_score(sentiment)
                    except Exception as e:
                        logger.error(f"Error analyzing generated comment: {str(e)}")
                        # Fallback to using the score
                        sentiment, sentiment_score = map_numeric_score_to_sentiment(score)
                else:
                    # Directly map the numeric score to sentiment
                    sentiment, sentiment_score = map_numeric_score_to_sentiment(score)
                
                # Create entry with generated comment - CHANGED: use empty string instead of "Algemeen"
                entry_data = {
                    "original_index": idx,
                    "intent": reason if not pd.isna(reason) and reason else "",
                    "text_summary": generated_comment,
                    "sentiment": sentiment,
                    "sentiment_score": sentiment_score,
                    "named_entities": [],
                    "is_generated": True  # Flag to indicate this comment was generated
                }
                
            else:
                # Process real comments
                if not offline_mode:
                    try:
                        # Use Hugging Face to analyze sentiment
                        sentiment_result = analyze_sentiment_hf(comment, hf_endpoint_url, hf_api_key)
                        sentiment = sentiment_result["sentiment"]
                        sentiment_score = map_sentiment_to_score(sentiment)
                        
                        # Create entry - CHANGED: use empty string instead of "Algemeen"
                        entry_data = {
                            "original_index": idx,
                            "intent": reason if not pd.isna(reason) and reason else "",
                            "text_summary": comment,
                            "sentiment": sentiment,
                            "sentiment_score": sentiment_score,
                            "named_entities": [],
                            "is_generated": False
                        }
                    except Exception as e:
                        logger.error(f"Error processing comment: {str(e)}")
                        # Fallback to using the score
                        sentiment, sentiment_score = map_numeric_score_to_sentiment(score)
                        
                        entry_data = {
                            "original_index": idx,
                            "intent": reason if not pd.isna(reason) and reason else "",
                            "text_summary": comment,
                            "sentiment": sentiment,
                            "sentiment_score": sentiment_score,
                            "named_entities": [],
                            "is_generated": False
                        }
                else:
                    # Offline mode - use score
                    sentiment, sentiment_score = map_numeric_score_to_sentiment(score)
                    
                    entry_data = {
                        "original_index": idx,
                        "intent": reason if not pd.isna(reason) and reason else "",
                        "text_summary": comment,
                        "sentiment": sentiment,
                        "sentiment_score": sentiment_score,
                        "named_entities": [],
                        "is_generated": False
                    }
            
            # Copy original fields
            for col in df.columns:
                entry_data[f"original_{col}"] = row[col]
            
            # Add to results
            all_processed_data.append(entry_data)
        
        # Save intermediate results periodically
        if len(all_processed_data) % 500 == 0 and all_processed_data:
            temp_df = pd.DataFrame(all_processed_data)
            temp_df.to_csv(f"{output_file}_intermediate_{len(all_processed_data)}.csv", index=False)
    
    # Create final DataFrame
    processed_df = pd.DataFrame(all_processed_data)
    
    # Create sentiment labels for BERT training (0=negative, 1=neutral, 2=positive)
    processed_df['sentiment_label'] = processed_df['sentiment'].map({
        'positief': 2, 
        'neutraal': 1, 
        'gemengd': 1,
        'negatief': 0
    })
    
    # Add combined text for context - CHANGED: conditionally format based on whether intent is empty
    processed_df['combined_text'] = processed_df.apply(
        lambda row: row['text_summary'] if row['intent'] == "" 
        else f"{row['intent']}: {row['text_summary']}", 
        axis=1
    )
    
    # Save to CSV
    processed_df.to_csv(output_file, index=False)
    logger.info(f"Preprocessing complete. Generated {len(processed_df)} entries from feedback data.")
    
    # Report statistics
    generated_count = processed_df['is_generated'].sum()
    logger.info(f"Generated comments: {generated_count} ({generated_count/len(processed_df):.1%})")
    logger.info(f"Sentiment distribution:\n{processed_df['sentiment'].value_counts()}")
    
    return processed_df

def process_reviews_with_hf(df: pd.DataFrame, review_column: str, hf_endpoint_url: str, hf_api_key: str, batch_size: int = 50) -> pd.DataFrame:
    """Light-weight wrapper expected by CustomerFeedbackAnalyzer.

    Given an in-memory DataFrame it calls the Hugging Face endpoint to obtain
    a single-word sentiment (positief/negatief/neutraal) for each row and
    appends the following columns:
        • sentiment
        • sentiment_score (numeric ‑1 … 1)
        • sentiment_label (0/1/2)
        • combined_text (copy of review_column with NaNs replaced)
        • is_generated (False for all rows – the analyzer may overwrite later)

    It re-uses `analyze_sentiment_hf` for each batch so the retry/initialisation
    logic remains centralised.
    """
    if review_column not in df.columns:
        raise ValueError(f"Column '{review_column}' not found in DataFrame")

    processed_df = df.copy()
    processed_df[review_column] = processed_df[review_column].fillna("")

    sentiments = []
    scores = []
    label_map = {"positief": 2, "neutraal": 1, "negatief": 0}

    for start in tqdm(range(0, len(processed_df), batch_size), desc="HF sentiment batches"):
        batch_texts = processed_df.iloc[start:start+batch_size][review_column].tolist()
        for txt in batch_texts:
            if not txt:
                sentiments.append("neutraal")
                scores.append(0.0)
                continue
            res = analyze_sentiment_hf(txt, hf_endpoint_url, hf_api_key)
            sentiments.append(res["sentiment"])
            scores.append(map_sentiment_to_score(res["sentiment"]))

    processed_df["sentiment"] = sentiments
    processed_df["sentiment_score"] = scores
    processed_df["sentiment_label"] = processed_df["sentiment"].map(label_map).fillna(1).astype(int)
    processed_df["combined_text"] = processed_df[review_column].fillna("")
    if "is_generated" not in processed_df.columns:
        processed_df["is_generated"] = False

    return processed_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess Dutch customer feedback with Hugging Face")
    parser.add_argument("--input", required=True, help="Path to Excel file with feedback")
    parser.add_argument("--output", default="preprocessed_feedback.csv", help="Path to save processed data")
    parser.add_argument("--endpoint", help="Hugging Face endpoint URL")
    parser.add_argument("--api_key", help="Hugging Face API key")
    parser.add_argument("--batch_size", type=int, default=50, help="Number of entries to process per batch")
    parser.add_argument("--use_api_for_missing", action="store_true", help="Use HF API for generated comments")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if output exists")
    parser.add_argument("--offline_mode", action="store_true", help="Skip API calls completely and use score-based sentiment")
    
    args = parser.parse_args()
    
    # Run preprocessing
    processed_df = preprocess_feedback(
        args.input, 
        args.output, 
        args.endpoint,
        args.api_key,
        args.batch_size,
        args.force,
        args.use_api_for_missing,
        args.offline_mode
    ) 