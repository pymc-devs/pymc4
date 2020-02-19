@ECHO OFF

for %%a in ("%%~dp0:~0,-1") do set "SRC_DIR=%%~dpa"

for %%a in (%*) do (
    if "%%a" == "--build" (
        docker build -t pymc4 ^
            -f %SRC_DIR%\scripts\Dockerfile ^
            --build-arg SRC_DIR=. %SRC_DIR% ^
            --rm
    )
)

for %%a in (%*) do (
    if "%%a" == "--clear_cache" (
        for /R %cd% %%G IN (__pycache__) do rmdir /s /q %%G
    )
)

for %%a in (%*) do (
    if "%%a" == "--test" (
        docker run --mount type=bind,source=%cd%,target=/opt/pymc4/ pymc4:latest bash -c ^
                                   "pytest -v pymc4 tests --doctest-modules --cov=pymc4/"
    )
)

EXIT /B 0