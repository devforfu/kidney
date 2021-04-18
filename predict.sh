#!/bin/bash

export PYTHONPATH=`pwd`
export DATASET_ROOT=/mnt/fast/data/kidney/raw
export DEFAULT_OUTPUT_DIR=/mnt/fast/data/kidney/outputs
export CHECKPOINTS_DIR="${HOME}/experiments"
export SAMPLE_TYPE=All
export DEVICE="cuda:0"
export EXPERIMENT_ID=
export RUN_ID=
export OUTPUT=
export FACTORY=

while [[ "$#" -gt 0 ]]
do
  case "$1" in
    -e|--experiment-id)
    EXPERIMENT_ID=${2}
    shift
    ;;
    -r|--run-id)
    RUN_ID=${2}
    shift
    ;;
    -f|--factory)
    FACTORY=${2}
    shift
    ;;
    -o|--output)
    OUTPUT=${2}
    shift
    ;;
    -s|--sample-type)
    SAMPLE_TYPE="${2}"
    shift
    ;;
    -d|--device)
    DEVICE="${2}"
    shift
    ;;
  esac
shift
done

if [[ -z "${EXPERIMENT_ID}" ]]
then
  echo "Error: --experiment-id is not set"
  exit 1
fi

if [[ -z "${FACTORY}" ]]
then
  echo "Error: --factory is not set"
  exit 1
fi

if [[ -z "${RUN_ID}" ]]
then
  RUN_ID="predictions"
fi

if [[ -z "${OUTPUT}" ]]
then
  OUTPUT="${DEFAULT_OUTPUT_DIR}/${EXPERIMENT_ID}/${RUN_ID}.csv"
fi

echo "***"
echo "Running inference with parameters:"
echo "- checkpoints: ${CHECKPOINTS_DIR}/${EXPERIMENT_ID}/${RUN_ID}"
echo "- model factory: ${FACTORY}"
echo "- sample type: ${SAMPLE_TYPE}"
echo "- device: ${DEVICE}"
echo "- output: ${OUTPUT}"
echo "***"

python bin/inference.py \
  --checkpoints_dir="${CHECKPOINTS_DIR}/${EXPERIMENT_ID}/checkpoints" \
  --factory_class="${FACTORY}" \
  --output_file="${OUTPUT}" \
  --device="${DEVICE}" \
  --sample_type="${SAMPLE_TYPE}"
