# System Patterns: Dutch Feedback Analyzer

## 1. System Architecture
The system follows a modular, pipeline-oriented architecture designed for processing and analyzing customer feedback. Key components include:

1.  **Data Ingestion Layer:** Accepts `.csv` or `.xlsx` files containing customer feedback. Managed by individual scripts and the Streamlit UI.
2.  **Preprocessing Module (`huggingface_preprocessing.py`, integrated parts in `customer_feedback_analyzer.py`):
    *   Text cleaning (special characters, normalization).
    *   Handling of missing comments: Generates placeholder comments based on available data (e.g., score, reason, business unit) and flags them (`is_generated=True`).
    *   Optional LLM interaction (via Hugging Face endpoint) for sentiment assignment to generated comments or other enrichment tasks during preprocessing.
    *   Offline mode for preprocessing relies on score-to-sentiment mapping.
3.  **Sentiment Analysis Module (`bert_sentiment_trainer.py`, `customer_feedback_analyzer.py`):
    *   Fine-tunes a Dutch BERT model (e.g., `bert-base-dutch-cased`) for sentiment classification (positive, negative, neutral).
    *   Applies the fine-tuned model to predict sentiment on preprocessed feedback.
4.  **Topic Modeling Module (`customer_feedback_analyzer.py` using BERTopic):
    *   Generates embeddings (e.g., using `paraphrase-multilingual-MiniLM-L12-v2`).
    *   Performs dimensionality reduction (UMAP) and clustering (HDBSCAN).
    *   Represents topics using c-TF-IDF.
    *   Typically filters out `is_generated=True` feedback before topic modeling.
5.  **Categorization/Schema Mapping Module (logic within `customer_feedback_analyzer.py` and utilized by `streamlit_app.py`):
    *   Maps identified topics or raw feedback to a `STRUCTURED_TOPIC_SCHEMA` using keyword matching defined in `SCHEMA_CATEGORY_KEYWORDS`.
    *   Assigns `schema_main_category` and `schema_sub_topic`.
6.  **Visualization Module (`visualizations.py`, `sentiment_visualizations.py`, Plotly in `streamlit_app.py`):
    *   Generates various static (HTML, PNG) and interactive (Plotly via Streamlit) visualizations for sentiment distributions, topic breakdowns, NPS scores, etc.
7.  **Orchestration Scripts (`run_pipeline.py`, `process_new_data.py`, `process_real_feedback.py`):
    *   Coordinate the execution of preprocessing, model training/application, and analysis steps based on command-line arguments.
8.  **Interactive UI (`streamlit_app.py`):
    *   Provides a web-based graphical user interface for all major functionalities.
    *   Manages user inputs, triggers backend script execution (via `subprocess`), and displays results dynamically.
    *   Integrates with Hugging Face endpoints for AI-driven features (custom analysis, suggestions).

## 2. Key Technical Decisions
-   **Dutch Language Focus:** Utilization of Dutch-specific pre-trained models (e.g., `bert-base-dutch-cased` for BERT, multilingual sentence transformers for BERTopic) to maximize NLP task performance.
-   **Domain-Specific Sentiment:** Fine-tuning BERT for sentiment analysis on the target domain's feedback data rather than relying solely on general-purpose sentiment models.
-   **Unsupervised Topic Discovery:** Employing BERTopic for its ability to discover topics without predefined labels, suitable for exploratory analysis.
-   **Structured Topic Schema:** Implementing a predefined schema (`STRUCTURED_TOPIC_SCHEMA`) and keyword-based mapping (`SCHEMA_CATEGORY_KEYWORDS`) to translate discovered topics into business-relevant categories, bridging the gap between unsupervised topics and business understanding.
-   **Hugging Face Endpoint Integration:** Leveraging external LLMs via Hugging Face Inference Endpoints for advanced tasks like AI-driven suggestions in the Streamlit app, and optionally for preprocessing tasks. This allows for incorporating powerful models without local hosting burdens.
-   **Streamlit for UI:** Choosing Streamlit to rapidly develop an interactive and user-friendly web application for data analysis and visualization.
-   **Handling AI-Generated Feedback:** Implementing a clear distinction (`is_generated` flag) between actual customer comments and system-generated placeholders for missing feedback. This is crucial for ensuring the integrity of analyses like topic modeling, which should primarily focus on genuine customer input.
-   **Modular Scripting:** Separating functionalities into different Python scripts (e.g., `huggingface_preprocessing.py`, `customer_feedback_analyzer.py`, `run_pipeline.py`) to promote reusability and maintainability.
-   **Offline Capability:** Providing an `--offline_mode` that allows the system to function without Hugging Face API calls, typically by simplifying certain steps (e.g., mapping scores directly to sentiment).

## 3. Design Patterns & Principles
-   **Pipeline Processing:** Data flows through a sequence of distinct processing stages (preprocess -> sentiment -> topic modeling -> categorize -> visualize).
-   **Modularity:** Core functionalities are encapsulated in separate modules/scripts, facilitating independent development, testing, and updates.
-   **Configuration over Hardcoding:** Key parameters, such as Hugging Face endpoint URLs, API keys, file paths, and model parameters, are configurable via command-line arguments or UI inputs, enhancing flexibility.
-   **Separation of Concerns:** The Streamlit UI (`streamlit_app.py`) is largely a presentation and control layer, while the core NLP and data processing logic resides in backend scripts (`customer_feedback_analyzer.py`, etc.).
-   **Graceful Degradation (Offline Mode):** The system can operate with reduced functionality (e.g., simpler sentiment assignment) if external API dependencies are unavailable or not configured.

## 4. Component Relationships & Data Flow
1.  **User Interaction (Streamlit):** `streamlit_app.py` captures user inputs (file uploads, column selections, analysis choices, API configs).
2.  **Subprocess Execution:** For analysis tasks, `streamlit_app.py` constructs and executes command-line calls to orchestrator scripts (e.g., `customer_feedback_analyzer.py`, `process_new_data.py`) using `subprocess.Popen`.
3.  **Orchestrator Scripts (`run_pipeline.py`, etc.):** These scripts manage the overall workflow, calling specialized modules:
    *   Invoke `huggingface_preprocessing.py` or its internal logic for data preparation.
    *   Invoke `customer_feedback_analyzer.py` (or its classes/functions) for model training (BERT, BERTopic) and application (sentiment prediction, topic assignment).
4.  **Core Analyzer (`customer_feedback_analyzer.py`):** Contains the primary classes and functions for:
    *   BERT model fine-tuning and inference.
    *   BERTopic model training and topic assignment.
    *   Logic for schema mapping and other analytical transformations.
5.  **Preprocessing (`huggingface_preprocessing.py`):** Handles initial data cleaning, generation of comments for missing entries, and interaction with HF endpoints if configured.
6.  **Data Persistence:**
    *   Intermediate and final datasets (e.g., `preprocessed.csv`, `feedback_with_bert_sentiment.csv`, `final_analysis_results.csv`) are saved to disk in the specified output directory.
    *   Trained models (`bert-sentiment-model/`, `topic_model`) are also saved to disk.
7.  **Visualization (`visualizations.py`, `sentiment_visualizations.py`, Streamlit plotting):
    *   Orchestrator scripts or the Streamlit app use these modules to generate plots from the analysis results.
    *   Streamlit app displays interactive Plotly charts directly.
8.  **Results Loading (Streamlit):** The `load_analysis_results` function in `streamlit_app.py` reads previously generated CSV files from output directories to populate exploration modes.

**Data Flow Summary:**
Raw Data (CSV/XLSX) -> Preprocessing (`huggingface_preprocessing.py`) -> Preprocessed Data -> Sentiment Analysis (`customer_feedback_analyzer.py`) -> Data with Sentiment -> Topic Modeling (`customer_feedback_analyzer.py`) -> Data with Topics & Sentiment -> Schema Mapping -> Final Analyzed Data (`final_analysis_results.csv`) -> Visualizations / Streamlit Exploration. 