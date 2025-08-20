"""
Configuration loader for the Dutch Feedback Analyzer application.

This module provides functions to load configuration data from JSON files,
making it easy to manage schemas and keywords separately from the main code.
"""

import json
import os
from typing import Dict, List
from pathlib import Path


def get_config_path(filename: str) -> str:
    """
    Get the full path to a configuration file.
    
    Args:
        filename: Name of the configuration file
        
    Returns:
        Full path to the configuration file
    """
    # Get the directory containing this script
    current_dir = Path(__file__).parent
    # Navigate to the config directory
    config_dir = current_dir.parent / "config"
    return str(config_dir / filename)


def load_topic_schema() -> Dict[str, List[str]]:
    """
    Load the structured topic schema from the configuration file.
    
    Returns:
        Dictionary containing topic categories and their subcategories
    """
    config_path = get_config_path("topic_schema.json")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Topic schema configuration file not found at {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in topic schema configuration: {e}")


def load_category_keywords() -> Dict[str, List[str]]:
    """
    Load the category keywords mapping from the configuration file.
    
    Returns:
        Dictionary containing category names and their associated keywords
    """
    config_path = get_config_path("category_keywords.json")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Category keywords configuration file not found at {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in category keywords configuration: {e}")


def validate_configurations() -> bool:
    """
    Validate that both configuration files exist and have consistent category names.
    
    Returns:
        True if configurations are valid, False otherwise
    """
    try:
        topic_schema = load_topic_schema()
        category_keywords = load_category_keywords()
        
        # Check that both configs have the same category keys
        topic_categories = set(topic_schema.keys())
        keyword_categories = set(category_keywords.keys())
        
        if topic_categories != keyword_categories:
            print(f"Warning: Mismatch between topic schema and category keywords.")
            print(f"Topic schema categories: {topic_categories}")
            print(f"Category keywords categories: {keyword_categories}")
            return False
            
        return True
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Configuration validation failed: {e}")
        return False


# For backward compatibility, provide the default configurations
def get_default_topic_schema() -> Dict[str, List[str]]:
    """
    Get default topic schema as fallback if configuration files are not available.
    
    Returns:
        Default topic schema dictionary
    """
    return {
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


def get_default_category_keywords() -> Dict[str, List[str]]:
    """
    Get default category keywords as fallback if configuration files are not available.
    
    Returns:
        Default category keywords dictionary
    """
    return {
        "Relevantie van Email": ["reclame", "spam", "overbodig", "ik wil dit niet", "waarom mail je", "stoppen met mail"],
        "Inhoud van Email": ["gegevens kloppen niet", "naam onjuist", "prijs niet zichtbaar", "wat kost", "onduidelijk"],
        "Prijs / Korting": ["te duur", "aanbod", "geen korting voor bestaande klanten", "korting te laag", "goedkoper"],
        "Merk": ["slechte naam", "odido", "brand", "merk", "imago", "reputatie", "identiteit", "naam"],
        "Klantenservice": ["service", "contact", "telefonisch contact", "winkel medewerker", "personeel", "wachttijd"],
        "Kwaliteit van Product": ["netwerk", "verbinding", "uitvallen", "valt uit", "bereik", "geen verbinding", "dekking"]
    } 