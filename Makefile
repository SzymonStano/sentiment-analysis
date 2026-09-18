DOCKER_IMAGE_NAME=pdiow-lab3-dvc
DOCKER_CONTAINER_NAME=pdiow-container-dvc

build:
	docker build --build-arg UID=$(shell id -u) --build-arg GID=$(shell id -g) -t $(DOCKER_IMAGE_NAME) . 

format:
	docker run --rm -v $(PWD):/app $(DOCKER_IMAGE_NAME) ruff format .

run_docker_dvc:
	docker run --rm -it -v "$(PWD)":/app -p 8888:8888 -u $(shell id -u):$(shell id -g) pdiow-lab3-dvc bash

run_jupyter_dvc:
	docker run --rm -it \
		-v "$(PWD)":/app \
		-p 8888:8888 \
		-u $(shell id -u):$(shell id -g) \
		-e JUPYTER_RUNTIME_DIR=/tmp/jupyter_runtime \
		pdiow-lab3-dvc \
		jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
