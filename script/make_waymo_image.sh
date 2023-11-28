#!/bin/bash


# while [[ "$#" -gt 0 ]]; do
#     case $1 in
#         -s|--ssh) SSH="$2"; shift ;;
#         *) echo "Unknown parameter passed: $1"; exit 1 ;;
#     esac
#     shift
# done

# Get directory of this script
SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DOCKER_DIR=${SOURCE_DIR}/docker/waymo

#Get version from autolabel3d/__init__.py
# VERSION=0.1
IMAGE_NAME=waymo
# :${VERSION}

DOCKER_BUILDKIT=1 docker build --build-arg CACHEBUST=$(date +%s) -t ${IMAGE_NAME} ${DOCKER_DIR}

