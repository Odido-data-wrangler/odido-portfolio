## Evaluation library & dataset tools (showcase)

Bestanden:
- `evaluation/evallib.py`
  - Laadt evaluator‑templates uit `evaluation/configuration/general.toml`
  - Azure OpenAI LLM‑as‑judge (JSON output) → `Feedback(score, comment)`
  - Dataset runs in Langfuse (`item.run(...)`) met `score_trace` per metric
  - Helpers: `create_benchmark`, `run_experiment`, `run_nova_dataset_experiment`
- `evaluation/create_datasets.py`
  - Maakt Langfuse datasets vanuit `tests/data/test_cases.json` o.b.v. TOML config
  - CLI subcommands voor benchmark/experiment/custom (showcase, niet bedoeld om hier te draaien)

Koppeling naar screenshots:
- Datasets‑overzicht → uit `create_nova_dataset_from_test_cases(...)`
- Run waterfall → uit `run_nova_dataset_experiment(...)` met `root_span.score_trace`
- Scores tabellen/hist → de metrics die per item zijn gelogd (relevance/completeness/accuracy/helpfulness/language_quality/intent_recognition)


