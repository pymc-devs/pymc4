# Docker FAQ

* Image can be built locally using the command `make docker` or the command
 `./scripts/container.sh --build` from the root `pymc4` directory
 
* After image is built an interactive bash session can be run 
`docker run -it pymc4 bash`

* Command can be issued to the container such as linting and testing
 without interactive session
  * `docker run pymc4 bash -c "pytest pymc4/tests"`
  * `docker run pymc4 bash -c "./scripts/lint.sh"`
