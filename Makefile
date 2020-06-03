.PHONY: help venv conda docker docstyle format style types black test lint check notebooks docstrings
.DEFAULT_GOAL = help

PYTHON = python
PIP = pip
CONDA = conda
SHELL = bash


help:
	@printf "Usage:\n"
	@grep -E '^[a-zA-Z_-]+:.*?# .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[1;34mmake %-10s\033[0m%s\n", $$1, $$2}'

conda:  # Set up a conda environment for development.
	@printf "Creating conda environment...\n"
	${CONDA} create --yes --name pymc4-env python=3.6
	( \
	${CONDA} activate pymc4-env; \
	${PIP} install -U pip; \
	${PIP} install -r requirements.txt; \
	${PIP} install -r requirements-dev.txt; \
	${CONDA} deactivate; \
	)
	@printf "\n\nConda environment created! \033[1;34mRun \`conda activate pymc4-env\` to activate it.\033[0m\n\n\n"

venv:  # Set up a Python virtual environment for development.
	@printf "Creating Python virtual environment...\n"
	rm -rf pymc4-venv
	${PYTHON} -m venv pymc4-venv
	( \
	source pymc4-venv/bin/activate; \
	${PIP} install -U pip; \
	${PIP} install -r requirements.txt; \
	${PIP} install -r requirements-dev.txt; \
	deactivate; \
	)
	@printf "\n\nVirtual environment created! \033[1;34mRun \`source pymc4-venv/bin/activate\` to activate it.\033[0m\n\n\n"

docker:  # Set up a Docker image for development.
	@printf "Creating Docker image...\n"
	${SHELL} ./scripts/container.sh --build

docstyle:
	@printf "Checking documentation with pydocstyle...\n"
	pydocstyle pymc4/
	@printf "\033[1;34mPydocstyle passes!\033[0m\n\n"

format:
	@printf "Checking code style with black...\n"
	black --check --diff pymc4 tests
	@printf "\033[1;34mBlack passes!\033[0m\n\n"

style:
	@printf "Checking code style with pylint...\n"
	pylint pymc4/
	@printf "\033[1;34mPylint passes!\033[0m\n\n"

types:
	@printf "Checking code type signatures with mypy...\n"
	python -m mypy --ignore-missing-imports pymc4/
	@printf "\033[1;34mMypy passes!\033[0m\n\n"

black:  # Format code in-place using black.
	black pymc4/ tests/

notebooks: notebooks/*
	jupyter nbconvert --config nbconfig.py --execute --ExecutePreprocessor.kernel_name="pymc4-dev" --ExecutePreprocessor.timeout=1200
	rm notebooks/*.html

test:  # Test code using pytest.
	pytest -v pymc4 tests --doctest-modules --html=testing-report.html --self-contained-html

lint: docstyle format style types  # Lint code using pydocstyle, black, pylint and mypy.

check: lint test  # Both lint and test code. Runs `make lint` followed by `make test`.

docstrings:  # Test that docstrings can generate an apidoc without warnings
	bash -c "trap 'trap - SIGINT SIGTERM ERR; rm -rf test_docs; exit 1' SIGINT SIGTERM ERR; $(MAKE) _docstrings"

_docstrings:
	sphinx-apidoc --ext-autodoc \
	--ext-intersphinx \
	--ext-mathjax \
	--extension sphinx.ext.napoleon \
	--extension matplotlib.sphinxext.plot_directive \
	--full --separate --module-first \
	-o test_docs pymc4 && \
	${PYTHON} ./scripts/apidoc_intersphinx_patch.py test_docs/conf.py
	sphinx-build -nWT test_docs/ test_docs/_build/