.PHONY: help build up shell down test test-image pipeline clean

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

test-image:            ## build the lightweight test runner
	docker build -q -t $(TEST_IMAGE) -f tests/Dockerfile .

test: test-image       ## run the test suite (no GPU, no TF/torch)
	docker run --rm -v "$(CURDIR):/w" $(TEST_IMAGE) python -m pytest tests/ -q

pipeline: up           ## run every stage end to end inside the container
	$(COMPOSE) exec training sigtrain all

clean:                 ## remove generated datasets and artifacts
	rm -rf data/processed artifacts
