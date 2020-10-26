@ECHO OFF

REM Command file to centralize test tasks

SET ROOT_DIR="%~dp0"
CALL :joinpath %ROOT_DIR% pymc4
SET PACKAGE_DIR=%RESULT%
CALL :joinpath %ROOT_DIR% tests
SET TESTS_DIR=%RESULT%
SET PYTHON=python
SET PIP=pip
SET CONDA=conda

if "%1" == "" (
    call :help
    exit /B 0
)

call :%1
EXIT /B %ERRORLEVEL%


:help
echo.Usage:
echo. make help: display help
echo. make venv: create python virtual environment
echo. make conda: create conda environment
echo. make docker: create docker image from application
echo. make docstyle: check package documentation with pydocstyle
echo. make format: check package formating with black
echo. make style: check package style with pylint
echo. make types: check typehints with mypy
echo. make black: apply black formating to the entire package and tests
echo. make test: run tests with pytest
echo. make lint: run all docstyle, format, style and docscheck checks
echo. make check: run lint, test
echo. make notebooks: execute jupyter notebooks
EXIT /B 0


:joinpath
set Path1=%~1
set Path2=%~2
if {%Path1:~-1,1%}=={\} (set Result="%Path1%%Path2%") else (set Result="%Path1%\%Path2%")
EXIT /B %ERRORLEVEL%


:conda
echo.Creating conda environment...
%CONDA% create --yes --name pymc4-env python=3.6
%CONDA% activate pymc4-env
%PIP% install -U pip
%PIP% install -r requirements.txt
%PIP% install -r requirements-dev.txt
%CONDA% deactivate
if %ERRORLEVEL%==0 (
    echo.Conda environment created! Run conda activate pymc4-env to activate it.
) else (
    echo.Failed to create conda environment.
)
EXIT /B %ERRORLEVEL%


:venv
echo.Creating Python virtual environment...
rmdir /s /q pymc4-venv
%PYTHON% -m venv pymc4-venv
pymc4-venv\Scripts\activate
%PIP% install -U pip
%PIP% install -r requirements.txt
%PIP% install -r requirements-dev.txt
deactivate
if %ERRORLEVEL%==0 (
    echo.Virtual environment created! Run pymc4-venv\Scripts\activate to activate it."
) else (
    echo.Failed to create virtual environment.
)
EXIT /B %ERRORLEVEL%


:docker
echo.Creating Docker image...
scripts\container --build
if %ERRORLEVEL%==0 (
    echo.Successfully built docker image.
) else (
    echo.Failed to build docker image.
)
EXIT /B %ERRORLEVEL%


:docstyle
echo.Checking documentation with pydocstyle...
%PYTHON% -m pydocstyle %PACKAGE_DIR%
if %ERRORLEVEL%==0 (
    echo.Pydocstyle passes!
) else (
    echo.Pydocstyle failed!
)
EXIT /B %ERRORLEVEL%


:format
echo.Checking code format with black...
%PYTHON% -m black --check --diff %PACKAGE_DIR% %TESTS_DIR%
if %ERRORLEVEL%==0 (
    echo.Black passes!
) else (
    echo.Black failed!
)
EXIT /B %ERRORLEVEL%


:style
echo.Checking style with pylint...
%PYTHON% -m pylint %PACKAGE_DIR%
if %ERRORLEVEL%==0 (
    echo.Pylint passes!
) else (
    echo.Pylint failed!
)
EXIT /B %ERRORLEVEL%


:types
echo.Checking type hints with mypy...
%PYTHON% -m mypy --ignore-missing-imports %PACKAGE_DIR%
if %ERRORLEVEL%==0 (
    echo.Mypy passes!
) else (
    echo.Mypy failed!
)
EXIT /B %ERRORLEVEL%


:black
%PYTHON% -m black  %PACKAGE_DIR% %TESTS_DIR%
EXIT /B %ERRORLEVEL%


:notebooks
jupyter nbconvert --config nbconfig.py --execute --ExecutePreprocessor.kernel_name="pymc4-dev" --ExecutePreprocessor.timeout=1200 --to html
del %ROOT_DIR%notebooks\*.html
EXIT /B %ERRORLEVEL%


:test
%PYTHON% -m pytest -v %PACKAGE_DIR% %TESTS_DIR% --doctest-modules --html=testing-report.html --self-contained-html
EXIT /B %ERRORLEVEL%


:lint
call :docstyle && call :format && call :style && call :types
EXIT /B %ERRORLEVEL%


:check
call :lint && call :test
EXIT /B %ERRORLEVEL%
