#!/usr/bin/env bash

source set_env.sh

docker run --privileged -v /home/ubuntu/greengrass-tutorial/docker/mb3/ml_inference:/ml_inference -v /home/ubuntu/greengrass-tutorial/docker/mb3/ml_image:/ml_image -v /home/ubuntu/greengrass-tutorial/docker/mb3/ml_model:/ml_model -ti $IMAGE_NAME
