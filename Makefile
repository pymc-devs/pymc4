docstyle:
	echo "Checking documentation with pydocstyle..."
	python -m pydocstyle ${SRC_DIR}/pymc4/
	echo "Success!"


format:
	echo "Checking code style with black..."
	python -m black -l 100 --check pymc4/
	echo "Success!"


style:
	echo "Checking code style with pylint..."
	python -m pylint pymc4/
	echo "Success!"


test:
	pytest -v pymc4/tests/ --cov=pymc4/

lint: docstyle format style


check: lint test