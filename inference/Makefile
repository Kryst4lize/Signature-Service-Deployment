.PHONY: help up dev down logs test test-integration test-image clean

COMPOSE    := docker compose
DEV        := -f docker-compose.yml -f docker-compose.dev.yml
TEST_IMAGE := sigverify-api-test
PGTEST     := sigverify_pgtest
PGNET      := sigverify_testnet

help:                      ## show this help
	@grep -hE '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t20

up:                        ## build and start the full stack
	$(COMPOSE) up -d --build

dev:                       ## start with hot reload
	$(COMPOSE) $(DEV) up

down:                      ## stop everything (volumes preserved)
	$(COMPOSE) down

logs:                      ## follow the api logs
	$(COMPOSE) logs -f api

test-image:                ## build the test runner image
	docker build -q -t $(TEST_IMAGE) -f tests/Dockerfile .

test: test-image           ## unit tests, no services required
	docker run --rm -v "$(CURDIR):/w" $(TEST_IMAGE) \
	  python -m pytest tests/test_tensors.py -q

test-integration: test-image  ## unit + endpoint tests against a throwaway pgvector
	@-docker network create $(PGNET) >/dev/null 2>&1
	@-docker rm -f $(PGTEST) >/dev/null 2>&1
	@docker run -d --name $(PGTEST) --network $(PGNET) \
	  -e POSTGRES_PASSWORD=test -e POSTGRES_DB=sig -e POSTGRES_USER=sig \
	  -v "$(CURDIR)/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql:ro" \
	  pgvector/pgvector:pg17 >/dev/null
	@echo "waiting for postgres..."
	@for i in $$(seq 1 30); do \
	    docker exec $(PGTEST) pg_isready -U sig -d sig >/dev/null 2>&1 && break; \
	    sleep 2; \
	 done
	@docker run --rm --network $(PGNET) -v "$(CURDIR):/w" \
	  -e TEST_DATABASE_URL=postgresql+asyncpg://sig:test@$(PGTEST):5432/sig \
	  $(TEST_IMAGE) python -m pytest tests/ -q; \
	  status=$$?; \
	  docker rm -f $(PGTEST) >/dev/null 2>&1; \
	  exit $$status

clean:                     ## remove test containers and networks
	-docker rm -f $(PGTEST) >/dev/null 2>&1
	-docker network rm $(PGNET) >/dev/null 2>&1
