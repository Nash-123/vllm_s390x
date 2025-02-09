#!/bin/bash

# This script build the CPU docker image and run the offline inference inside the container.
set -ex

# Setup cleanup
remove_docker_container() { docker rm -f cpu-test || true; docker system prune -f; }
trap remove_docker_container EXIT
remove_docker_container

# Try building the docker image
docker build -t cpu-test -f Dockerfile.s390x .
