#!/bin/bash

# -----------------
# Global parameters
# -----------------

export PYTHONPATH=`pwd`
export DATASET_ROOT=/mnt/fast/data/kidney/raw
export CHECKPOINTS_ROOT="${HOME}/experiments"
export CONFIGS_ROOT="$(pwd)/configurations_predict"
export CONFIG_FILE=
export EXPERIMENT_ID=


# -----------------------
# Parsing input arguments
# -----------------------

while [[ "$#" -gt 0 ]]
do
  case "$1" in
    -e|--experiment-id)
    EXPERIMENT_ID="${2}"
    shift
    ;;
    *)
    echo "Unknown parameter: $1"
    exit 1
    ;;
  esac
shift
done

if [[ -z "${EXPERIMENT_ID}" ]]
then
  echo "Error: --experiment-id is not provided"
  exit 1
fi

CONFIG_FILE="${CONFIGS_ROOT}/${EXPERIMENT_ID}.yaml"

if [[ ! -f "${CONFIG_FILE}" ]]
then
  echo "Error: config file is not found: ${CONFIG_FILE}"
  exit 1
fi


# ----------------
# Helper functions
# ----------------


function predict {
  RUN_ID=${1} python prototype/predict.py
  return_status=$?
  if [[ $return_status -ne 0 ]]
  then
    echo
    echo "!!! Warning: prediction for RUN_ID=${1} failed !!!"
  else
    echo "Prediction for RUN_ID=${1} is done"
  fi
}


# -----------------
# K-fold prediction
# -----------------

echo "***"
echo "Running folds experiment with ID: ${EXPERIMENT_ID}"
echo "Using config file: ${CONFIG_FILE}"
echo "***"

for directory in ${CHECKPOINTS_ROOT}/${EXPERIMENT_ID}/checkpoints/*
do
  echo "Reading checkpoint: ${directory}"
  predict $(basename ${directory})
done
