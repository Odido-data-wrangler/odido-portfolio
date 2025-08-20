# Tech Context: Dutch Feedback Analyzer

## 1. Core Technologies & Libraries
-   **Python:** Primary programming language (version 3.12 mentioned in venv).
-   **Streamlit:** For building the interactive web application UI (`streamlit_app.py`).
-   **Pandas:** For data manipulation and analysis (e.g., reading CSV/Excel, DataFrame operations).
-   **NumPy:** For numerical operations, often used in conjunction with Pandas.
-   **Hugging Face Transformers:** For accessing and fine-tuning pre-trained BERT models (e.g., `bert-base-dutch-cased`) for sentiment analysis.
-   **BERTopic:** For performing topic modeling. This library itself uses:
    *   `sentence-transformers` (e.g., `paraphrase-multilingual-MiniLM-L12-v2`) for generating document embeddings.
    *   `UMAP` for dimensionality reduction.
    *   `HDBSCAN` for clustering.
    *   `scikit-learn` for TF-IDF (c-TF-IDF).
-   **Scikit-learn:** For machine learning utilities, including evaluation metrics (`accuracy_score`, `classification_report`, `confusion_matrix`).
-   **Plotly & Plotly Express:** For creating interactive charts and visualizations displayed in Streamlit.
-   **Matplotlib & Seaborn:** Used for generating static plots, particularly confusion matrices in the Streamlit app's evaluation mode.
-   **Requests:** For making HTTP requests to the Hugging Face Inference Endpoint.
-   **NLTK:** (Presence inferred from `nltk` in `feedback-analysis-env`) Potentially used for basic NLP tasks like tokenization or stemming if not entirely superseded by Hugging Face tokenizers, though its specific use isn't detailed in the primary READMEs for core flows.

## 2. Development Environment & Setup
-   **Virtual Environment:** A Python virtual environment is used (e.g., `feedback-analysis-env/`, `.venv/`).
    *   Activation: `source feedback-analysis-env/bin/activate` (example).
-   **Requirements Files:**
    *   `requirements.txt`: Likely contains dependencies for the core pipeline scripts.
    *   `streamlit_requirements.txt`: Contains dependencies specifically for the Streamlit application.
    *   `uv.lock` and `pyproject.toml` suggest use of `uv` or `Poetry`/`PDM` for more robust dependency management, with `uv.lock` being a lock file.
-   **IDE:** `.vscode/` directory indicates use of Visual Studio Code.
-   **Version Control:** `.git/` directory and `.gitignore` file indicate the project is managed with Git.

## 3. Key Scripts & Entry Points
-   **Streamlit App:** `streamlit run streamlit_app.py`
-   **Full Pipeline:** `python run_pipeline.py --input <path> --column <name> --output_dir <dir> [options]`
-   **Process New Data (Existing Models):** `python process_new_data.py --input <path> --column <name> --model_dir <dir> --output_dir <dir> [options]`
-   **Process "Real" Feedback:** `python process_real_feedback.py --input_file <path> --output_dir <dir> [options]`
-   **Regenerate Visualizations:** `python regenerate_visualizations.py --results_file <path> --topic_model_file <path> --output_dir <dir> [options]`
-   **Synthetic Data Generation:** `python synthetic_data.py [--num_reviews <N>] [--output_file <path>]`
-   **Core Logic:** `customer_feedback_analyzer.py` (houses main analysis classes/functions), `huggingface_preprocessing.py` (specialized preprocessing).

## 4. Data Storage & Formats
-   **Input Data:** `.csv` or `.xlsx` files.
-   **Output Data & Artifacts:**
    *   Processed data: `.csv` files (e.g., `preprocessed_feedback.csv`, `final_analysis_results.csv`).
    *   Trained models: Saved in specified directories (e.g., `output_dir/bert-sentiment-model/`, `output_dir/topic_model`). BERTopic model is typically a single file, BERT model consists of multiple files (config, weights).
    *   Visualizations: `.html` (interactive Plotly/BERTopic plots), `.png` (static plots).
-   **Temporary Files:** `streamlit_temp/` directory used by the Streamlit app for temporary storage of uploaded files.

## 5. External Services & APIs
-   **Hugging Face Inference Endpoint:**
    *   Used by `streamlit_app.py` for AI-driven suggestions and custom analysis.
    *   Optionally used by `huggingface_preprocessing.py` (via `process_real_feedback.py` or `run_pipeline.py`) for sentiment analysis of generated comments or other LLM-based preprocessing, if not in `--offline_mode`.
    *   Requires configuration of an endpoint URL (`hf_endpoint`) and API key (`hf_api_key`) in the Streamlit sidebar or via command-line arguments.

## 6. Technical Constraints & Considerations
-   **Dutch Language Models:** Performance is dependent on the availability and quality of Dutch-specific NLP models.
-   **Computational Resources:** Training/fine-tuning BERT models and running BERTopic on large datasets can be computationally intensive. The `CLAUDE.md` mentions model training is hardware-accelerated when available.
-   **API Rate Limits/Costs:** If using a Hugging Face Inference Endpoint, be mindful of potential rate limits or costs associated with API calls, especially for large-scale processing.
-   **Dependency Management:** Ensuring consistency of Python packages and their versions across development and deployment is crucial. The presence of `uv.lock` is a good sign for reproducible environments.
-   **Error Handling:** Scripts incorporate `try-except` blocks for robustness. The Streamlit app streams command output and displays errors.
-   **Offline Functionality:** The `--offline_mode` provides a fallback for environments without internet access or API keys, but with potentially reduced accuracy or feature set for some preprocessing steps. 