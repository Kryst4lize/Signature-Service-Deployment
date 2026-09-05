.PHONY: help build up shell down test test-image lint format typecheck check lock test-lock-drift pipeline clean

COMPOSE    := docker compose
TEST_IMAGE := signature-training-test

help:                  ## show this help
	@grep -hE '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t18

build:                 ## build the GPU training image
	$(COMPOSE) build

up:                    ## start the training container
	$(COMPOSE) up -d

shell: up              ## open a shell inside it
	$(COMPOSE) exec training bash

down:                  ## stop it
	$(COMPOSE) down

lock:                  ## refresh uv.lock from pyproject.toml
	uv lock

lint:                  ## ruff check
	uv run ruff check .

format:                ## ruff format
	uv run ruff format .

typecheck:             ## ty check
	uv run ty check src

check: lint typecheck  ## lint + typecheck, no formatting changes
	uv run ruff format --check .

test-lock-drift:       ## fail if tests/Dockerfile pins drift from uv.lock
	@uv run python scripts/check_test_pins.py

test-image:            ## build the lightweight test runner
	docker build -q -t $(TEST_IMAGE) -f tests/Dockerfile .

test: test-image       ## run the test suite (no GPU, no TF/torch)
	docker run --rm -v "$(CURDIR):/w" $(TEST_IMAGE) python -m pytest tests/ -q

pipeline: up           ## run every stage end to end inside the container
	$(COMPOSE) exec training sigtrain all

clean:                 ## remove generated datasets and artifacts
	rm -rf data/processed artifacts
