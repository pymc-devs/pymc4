.PHONY: help venv conda docker docstyle format style black test lint check
.DEFAULT_GOAL = help

PYTHON = python3
PIP = pip3
CONDA = conda
SHELL = bash

help:
	@printf "Usage:\n\n"
	@grep -E '^[a-zA-Z_-]+:.*?# .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[1;34mmake %-10s\033[0m %s\n", $$1, $$2}'

conda:  # Set up a conda environment for development.
	@printf "Creating conda environment...\n"
	${CONDA} create --name env
	( \
	source env/bin/activate; \
	${CONDA} install -r requirements.txt; \
	${CONDA} install -r requirements-dev.txt; \
	deactivate; \
	)
	@printf "\n\nConda environment created! \033[1;34mRun \`source env/bin/activate\` to activate it.\033[0m\n\n\n"

venv:  # Set up a Python virtual environment for development.
	@printf "Creating Python virtual environment...\n"
	${PYTHON} -m venv venv
	( \
	source venv/bin/activate; \
	${PIP} install -U pip; \
	${PIP} install -r requirements.txt; \
	${PIP} install -r requirements-dev.txt; \
	deactivate; \
	)
	@printf "\n\nVirtual environment created! \033[1;34mRun \`source venv/bin/activate\` to activate it.\033[0m\n\n\n"

docker:  # Set up a Docker image for development.
	@printf "Creating Docker image...\n"
	${SHELL} ./scripts/container.sh --build
	@printf "\n\nDocker image created! \033[1;34mRun \`source venv/bin/activate\` to activate it.\033[0m\n\n\n"

docstyle:
	@printf "Checking documentation with pydocstyle...\n"
	pydocstyle pymc4/
	@printf "\033[1;34mPydocstyle passes!\033[0m\n\n"

format:
	@printf "Checking code style with black...\n"
	black --check pymc4/
	@printf "\033[1;34mBlack passes!\033[0m\n\n"

style:
	@printf "Checking code style with pylint...\n"
	pylint pymc4/
	@printf "\033[1;34mPylint passes!\033[0m\n"

black:  # Format code in-place using black.
	black pymc4/

test:  # Test code using pytest.
	pytest -v pymc4/tests/ --cov=pymc4/ --html=testing-report.html --self-contained-html

lint: docstyle format style  # Lint code using pydocstyle, black and pylint.

check: lint test  # Lint and test code.
