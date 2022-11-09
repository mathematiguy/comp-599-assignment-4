REPO_NAME := $(shell basename `git rev-parse --show-toplevel` | tr '[:upper:]' '[:lower:]')
GIT_TAG ?= $(shell git log --oneline | head -n1 | awk '{print $$1}')
COMPUTE ?= gpu
DOCKER_REGISTRY := mathematiguy
IMAGE := $(DOCKER_REGISTRY)/$(REPO_NAME)-$(COMPUTE)
HAS_DOCKER ?= $(shell which docker)
UID ?= user
GID ?= user
DOCKER_ARGS ?=

ifeq ($(COMPUTE),gpu)
	GPU_FLAGS=--gpus all
else
	GPU_FLAGS=
endif

RUN ?= $(if $(HAS_DOCKER), docker run $(DOCKER_ARGS) $(GPU_FLAGS) --ipc host -it --rm -v $$(pwd):/code -w /code -u $(UID):$(GID) $(IMAGE))

.PHONY: docker docker-push docker-pull enter enter-root

run: code.py
	$(RUN) python $<

test:
	$(RUN) pytest

JUPYTER_PASSWORD ?= jupyter
JUPYTER_PORT ?= 8888
.PHONY: jupyter
jupyter: DOCKER_ARGS=-u $(UID):$(GID) --rm -it -p $(JUPYTER_PORT):$(JUPYTER_PORT) -e NB_USER=$$USER -e NB_UID=$(UID) -e NB_GID=$(GID)
jupyter:
	$(RUN) jupyter lab \
		--port $(JUPYTER_PORT) \
		--ip 0.0.0.0 \
		--NotebookApp.password="$(shell $(RUN) \
			python3 -c \
			"from notebook.auth import passwd; print(passwd('$(JUPYTER_PASSWORD)', 'sha1'))")"

docker:
	docker build $(DOCKER_ARGS) --tag $(IMAGE):$(GIT_TAG) . -f Dockerfile.$(COMPUTE)
	docker tag $(IMAGE):$(GIT_TAG) $(IMAGE):latest

docker-push:
	docker push $(IMAGE):$(GIT_TAG)
	docker push $(IMAGE):latest

docker-pull:
	docker pull $(IMAGE):$(GIT_TAG)
	docker tag $(IMAGE):$(GIT_TAG) $(IMAGE):latest

enter: DOCKER_ARGS=-it
enter:
	$(RUN) bash

enter-root: DOCKER_ARGS=-it
enter-root: UID=root
enter-root: GID=root
enter-root:
	$(RUN) bash
