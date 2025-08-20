## Evaluatiemetrics (LLM-as-judge)
- relevance, completeness, accuracy, helpfulness, language_quality, intent_recognition
Schaal: 0.0–1.0. Logging via Langfuse `score_trace`.

## Extra signalen
- sources_count (context coverage proxy)
- nova_success (1/0)
- abstain/clarify (1/0 indien van toepassing)

## Rapportage
- Macro-avg per metric per run (benchmark vs experiment)
- Win-rate: experiment > benchmark per metric
