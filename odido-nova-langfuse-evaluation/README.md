## Odido NLU + Langfuse Evaluation (code showcase)

Doel: laten zien hoe ik Nova/Izzy-evaluaties heb ingericht met Langfuse datasets & runs, LLM-as-judge prompts en scoringslogica. Deze repo bevat code en documentatie, geen werkende runtime of secrets.

### Inhoud
- `evaluation/`: evaluatiecode (dataset creation, dataset runs, evaluators, Langfuse scoring hooks)
- `docs/`: uitleg, diagrammen en Langfuse-screenshots
  - `docs/traces-external-evaluation.md` beschrijft de externe trace‑aanpak en hoe dit in Langfuse zichtbaar is
- `METRICS.md`: definitie van alle scores
- `CHANGES.md`: benchmark (V0) vs experiment (V1) verschillen

### Belangrijke notities
- Alle sleutels/secrets verwijderd of geanonimiseerd
- Screenshots tonen functionaliteit; geen PII (geblurred)
- Code kan niet draaien zonder interne infra; dit is bewust
