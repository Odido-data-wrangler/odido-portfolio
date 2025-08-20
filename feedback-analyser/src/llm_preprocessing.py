import pandas as pd
import json
from openai import OpenAI
from tqdm.auto import tqdm
import time
import os

def preprocess_reviews(input_file, output_file, api_key, review_column, batch_size=50):
    """
    Preprocess Dutch customer reviews using an LLM to extract structured data.
    
    Args:
        input_file: Path to the Excel/CSV file containing reviews
        output_file: Path to save the preprocessed data
        api_key: OpenAI API key
        review_column: Name of the column containing review text
        batch_size: Number of reviews to process in each batch
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Load data
    print(f"Loading data from {input_file}")
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    else:
        df = pd.read_excel(input_file)
    
    print(f"Loaded {len(df)} reviews")
    
    # Define function calling schema
    functions = [
        {
            "name": "extract_review_data",
            "description": "Extract structured data from Dutch customer review text",
            "parameters": {
                "type": "object",
                "properties": {
                    "intents": {
                        "type": "array",
                        "description": "List of different intents/topics mentioned in the review",
                        "items": {
                            "type": "object",
                            "properties": {
                                "intent": {
                                    "type": "string",
                                    "description": "The specific intent or aspect being discussed in Dutch (e.g., Product, Klantenservice, Aankoop/Levering, Merk Perceptie)"
                                },
                                "text_summary": {
                                    "type": "string",
                                    "description": "A concise summary in Dutch of this specific part of the review"
                                },
                                "sentiment": {
                                    "type": "string",
                                    "enum": ["positief", "negatief", "neutraal", "gemengd"],
                                    "description": "The sentiment for this specific intent in Dutch"
                                },
                                "sentiment_score": {
                                    "type": "number",
                                    "description": "Sentiment score from -1 (very negative) to 1 (very positive)"
                                },
                                "named_entities": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Any named entities mentioned in Dutch (product names, features, specific services, etc.)"
                                }
                            },
                            "required": ["intent", "text_summary", "sentiment", "sentiment_score"]
                        }
                    }
                },
                "required": ["intents"]
            }
        }
    ]
    
    all_processed_data = []
    
    # Process reviews in batches
    for i in tqdm(range(0, len(df), batch_size), desc="Verwerken van recensie batches"):
        batch = df.iloc[i:i+batch_size]
        
        for idx, row in batch.iterrows():
            review_text = row[review_column]
            
            # Skip empty reviews
            if pd.isna(review_text) or not isinstance(review_text, str) or review_text.strip() == "":
                continue
            
            # Prepare prompt
            prompt = f"""
            Analyseer deze Nederlandse klantrecensie en extraheer gestructureerde data. Splits de recensie op in verschillende intenties of aspecten die worden besproken.
            Voor elke intentie:
            1. Identificeer het specifieke onderwerp/aspect (Product, Klantenservice, Aankoop/Levering, Merk Perceptie, of een ander onderwerp)
            2. Geef een beknopte samenvatting voor dat specifieke aspect in het Nederlands
            3. Bepaal het sentiment (positief, negatief, neutraal, of gemengd) en ken een sentiment score toe (-1 tot 1)
            4. Extraheer alle benoemde entiteiten (productnamen, specifieke functies, etc.)
            
            Klantrecensie: "{review_text}"
            """
            
            # Process with retries
            max_retries = 3
            retry_count = 0
            success = False
            
            while not success and retry_count < max_retries:
                try:
                    # Call OpenAI API
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",  # Or another appropriate model
                        messages=[{"role": "user", "content": prompt}],
                        functions=functions,
                        function_call={"name": "extract_review_data"}
                    )
                    
                    # Parse function output
                    function_response = json.loads(response.choices[0].message.function_call.arguments)
                    
                    # Add original metadata
                    for intent in function_response["intents"]:
                        intent_data = intent.copy()
                        intent_data["original_index"] = idx
                        intent_data["original_text"] = review_text
                        
                        # Add other metadata from original row
                        for col in df.columns:
                            if col != review_column:
                                intent_data[f"original_{col}"] = row[col]
                        
                        all_processed_data.append(intent_data)
                    
                    success = True
                    
                except Exception as e:
                    retry_count += 1
                    print(f"Fout bij het verwerken van recensie (poging {retry_count}/{max_retries}): {str(e)}")
                    time.sleep(2)  # Wait before retry
            
            if not success:
                print(f"Kon recensie niet verwerken na {max_retries} pogingen")
        
        # Save intermediate results periodically
        if len(all_processed_data) % 500 == 0 and all_processed_data:
            temp_df = pd.DataFrame(all_processed_data)
            temp_df.to_csv(f"{output_file}_intermediate_{len(all_processed_data)}.csv", index=False)
    
    # Create final DataFrame and save
    processed_df = pd.DataFrame(all_processed_data)
    
    # Create sentiment labels for BERT training
    processed_df['sentiment_label'] = processed_df['sentiment'].map({
        'positief': 2, 
        'neutraal': 1, 
        'gemengd': 1,  # Considering mixed as neutral for simplicity
        'negatief': 0
    })
    
    # Add text column that combines intent and summary for better context
    processed_df['combined_text'] = processed_df.apply(
        lambda row: row['text_summary'] if row['intent'] == "" 
        else f"{row['intent']}: {row['text_summary']}", 
        axis=1
    )
    
    # Save to file
    processed_df.to_csv(output_file, index=False)
    print(f"Voorverwerking voltooid. Er zijn {len(processed_df)} intentie rijen gegenereerd uit de originele recensies.")
    
    return processed_df

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verwerk Nederlandse klantrecensies met een LLM")
    parser.add_argument("--input", required=True, help="Pad naar het Excel/CSV bestand met recensies")
    parser.add_argument("--output", default="preprocessed_reviews.csv", help="Pad om de verwerkte data op te slaan")
    parser.add_argument("--api_key", required=True, help="OpenAI API key")
    parser.add_argument("--column", required=True, help="Naam van de kolom met recensietekst")
    parser.add_argument("--batch_size", type=int, default=50, help="Aantal recensies om in elke batch te verwerken")
    
    args = parser.parse_args()
    
    # Run preprocessing
    processed_df = preprocess_reviews(args.input, args.output, args.api_key, args.column, args.batch_size) 