## Code-overzicht

- `evaluation/configuration/benchmark.toml` en `experiment.toml`
  - Sturen datasetnaam, run_name, evaluatielijst en doelwaarden.
- `evaluation/configuration/general.toml`
  - LLM-as-judge templates (relevance, completeness, accuracy, helpfulness, language_quality, intent_recognition).
- `evaluation/evallib.py`
  - Laadt templates en voert LLM-judge aan; schrijft scores naar Langfuse via dataset runs.
- `evaluation/evaluation_integration.py`
  - Helpers om evaluaties te draaien en als Langfuse-scores vast te leggen.
- `evaluation/create_datasets.py`
  - Maakt Langfuse datasets op basis van testcases + TOML-config.
- `evaluation/experiment.py`
  - Orkestreert benchmark-run (V0) en experiment-run (V1) en logt naar Langfuse.
- `evaluation/nova_external_test.py`
  - Externe evaluatieflow via websocket + LLM-judge; als referentie (secrets verwijderd).
