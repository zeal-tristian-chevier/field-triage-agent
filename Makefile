REPORT ?= Pump P-104 vibrating, bearing temp 92C, last serviced 14 months ago

.PHONY: run test eval smoke gate

run:
	uv run python -m field_triage.cli "$(REPORT)"

test:
	uv run pytest -q

# full 30-case run, logged to LangSmith when LANGSMITH_API_KEY is set
eval:
	uv run python -m evals.run_eval --subset full
	uv run python -m evals.gate results/latest.json

# 10-case subset, same as the PR check
smoke:
	uv run python -m evals.run_eval --subset smoke
	uv run python -m evals.gate results/latest.json
