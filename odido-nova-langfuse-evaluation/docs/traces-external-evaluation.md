

### Bestand
- `evaluation/nova_external_test.py` (code‑drop, secrets verwijderd in repo)

### Wat dit script aantoont
- WebSocket‑integratie met Nova (self‑contained):
  - Bouwt een sessie op, handelt `input_required`/`confirm_customer`, verzamelt `response_streaming`, `context_retrieved`, `final_result`.
- Langfuse observability end‑to‑end:
  - Start een trace (`start_as_current_span`), logt input/metadata (question, conversation_id, evaluation_type)
  - Scoort per metric via `score_current_trace` (relevance, completeness, accuracy, helpfulness, language_quality, intent_recognition)
  - Schrijft samenvatting (average score, best/worst) in trace output
- LLM‑as‑judge evaluatie:
  - Laadt templates uit `evaluation/configuration/general.toml` (fallback inline)
  - Roept Azure OpenAI aan met JSON‑structured output (score/comment)
- Veilige PEM‑afhandeling: temp‑file met strikte permissies, cleanup na afloop

### Flow (extern evaluate → traces)

Zie ook `docs/diagrams/external-eval.mmd`.

1) Start Langfuse trace met input `{question, conversation_id, evaluation_type}`
2) Open Nova WebSocket en verzamel events (intent, bronnen, streamed answer)
3) Op `final_result` of `response_complete`: bouw resultaatobject
4) Run evaluaties via Azure OpenAI per metric → voeg scores toe aan huidige trace
5) Schrijf output (answer snippet, intent, sources_count, success, avg score) → flush

### Welke metadata/scores staan in de trace
- Input: `question`, `conversation_id`, `evaluation_type="external_pattern"`
- Output: `answer` (ingekort), `intent`, `sources_count`, `success`, `evaluations_completed`, `average_evaluation_score`
- Scores: `relevance`, `completeness`, `accuracy`, `helpfulness`, `language_quality`, `intent_recognition` (+ evt. `evaluation_error` of `nova_basic_quality`)

### Aanbevolen screenshots in Langfuse (koppeling met dit script)
- `run_experiment_waterfall_clarify.png`: waterfall van een item met de externe span‑naam `nova_external_evaluation`
- `judge_prompt_v1.png`: system prompt uit `general.toml` + JSON‑output
- `scores_table_experiment.png`: tabel met gemiddelde scores per metric
- `metadata_filters.png`: filter op `evaluation_type=external_pattern`

### Veiligheid
- In deze publieksrepo geen secrets/tokens. De code ondersteunt Secrets Manager en tijdelijke PEM‑bestanden, maar die zijn hier niet nodig.

### Beperkingen
- Dit is een trace‑/observability‑showcase. Zonder interne infra/data zal deze code in deze repo niet draaien; dat is bewust.
