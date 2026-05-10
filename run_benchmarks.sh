#!/usr/bin/env bash

set -e

# SCENES=(
#   garden_ce
#   bonsai_ce
#   room_ce
#   counter_ce
#   kitchen_ce
# )


# BASE_DIR="output/mipnerf360"

# SCENES=(
#   figurines_ce
#   ramen_ce
#   teatime_ce
# )

# BASE_DIR="output/lerf"

SCENES=(
  fern_ce
  flower_ce
  fortress_ce
  horns_ce
  leaves_ce
  orchids_ce
  room_ce
  trex_ce
)

BASE_DIR="output/llff"


GPU_ID=1

for SCENE in "${SCENES[@]}"; do
  echo "Running benchmark for ${SCENE}"

  CUDA_VISIBLE_DEVICES=${GPU_ID} \
  python3 benchmark.py \
    -c ${BASE_DIR}/${SCENE}/config.yaml

  echo "Finished ${SCENE}"
  echo
done
