#!/usr/bin/env bash

source set_env.sh

rm build_$IMAGE_NAME.out 2>/dev/null
nohup docker build -t $IMAGE_NAME .. > docker_build_$IMAGE_NAME.out &
