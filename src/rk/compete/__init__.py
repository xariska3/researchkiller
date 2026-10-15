"""Phase 2 competitive evaluation (docs/phase2_spec.md): is the live tool better than what
researchers already use?

- rk.compete.systems: the systems compared (Claude and an OpenAI model with their web
  search tools, Perplexity Sonar, Semantic Scholar search), one prompt for all, every
  response cached and every real call charged to a per-system cost cap.
- rk.compete.resolve: whether a listed paper exists, through free routes only
  (OpenAlex singleton lookups, Crossref title search).
- rk.compete.evaluate: the stratified query sample, the live tool's own lists replayed
  from the live-eval caches, scoring and the report (python -m rk compete).
"""
