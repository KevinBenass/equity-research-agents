.PHONY: setup data eval test lint diagram

UV ?= uv
RUN = $(UV) run --env-file .env

setup:    ## Install dependencies and create .env
	$(UV) sync
	@test -f .env || (cp .env.example .env && echo "Created .env: add ANTHROPIC_API_KEY and SEC_USER_AGENT")

data:     ## Download SEC XBRL company facts
	$(RUN) era data

eval:     ## Run the evaluation suite and update the README results
	$(RUN) era eval

diagram:  ## Export the LangGraph topology
	$(UV) run era graph-diagram

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
