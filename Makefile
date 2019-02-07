docstyle:
	@echo "Checking documentation with pydocstyle..."
	pydocstyle pymc4/
	@echo "Pydocstyle passes! \n"


format:
	@echo "Checking code style with black..."
	black --check pymc4/
	@echo "Black passes! \n"

style:
	@echo "Checking code style with pylint..."
	pylint pymc4/
	@echo "Pylint passes!\n"

black:
	black pymc4/

test:
	pytest -v pymc4/tests/ --cov=pymc4/ --html=testing-report.html --self-contained-html


lint: docstyle format style

check: lint black test