#!/bin/bash

# -----------------
# Global parameters
# -----------------

export PYTHONPATH=`pwd`
export DATASET_ROOT=/mnt/fast/data/kidney/raw
export CONFIG_FILE="$(pwd)/configurations/dev.yaml"
export VAL_FILE="$(pwd)/validation_json/dev.json"


# -----------------------
# Parsing input arguments
# -----------------------

while [[ "$#" -gt 0 ]]
do
  case "$1" in
    -c|--config)
    CONFIG_FILE="${2}"
    shift
    ;;
    -v|--validation)
    VAL_FILE="${2}"
    shift
    ;;
    *)
    echo "Unknown parameter: $1"
    exit 1
    ;;
  esac
shift
done

if [[ ! -f "${CONFIG_FILE}" ]]
then
  echo "Error: config file is not found: ${CONFIG_FILE}"
  exit 1
fi

if [[ ! -f "${VAL_FILE}" ]]
then
  echo "Error: validation file is not found: ${VAL_FILE}"
  exit 1
fi


# ----------------
# Helper functions
# ----------------

function random_string {
  size=${1:-10}
  cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w ${size} | head -n 1
}

function generate_timestamp {
  date +"%a_%d_%b__%H_%M_%S"
}

function execute_script_with_fold {
  VAL_FOLD=${1} python prototype/main.py
  return_status=$?
  if [[ $return_status -ne 0 ]]
  then
    echo
    echo "!!! Warning: training fold ${1} failed !!!"
  else
    echo "The fold ${1} is done"
  fi
}


# --------------------------
# K-fold validation training
# --------------------------

unique_id=`random_string`
n_folds=`jq ".n_folds" ${VAL_FILE}`

echo "***"
echo "Running folds experiment with unique ID: ${unique_id}"
echo "Using config file: ${CONFIG_FILE}"
echo "Using validation file: ${VAL_FILE}"
echo "The total number of folds: ${n_folds}"
echo "***"

for ((i=0;i<n_folds;i++))
do
  echo "Training with fold $((i+1)) of ${n_folds}"
   execute_script_with_fold "${i}"
   sleep 1
done
