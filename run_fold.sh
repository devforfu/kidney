#!/bin/bash
# A shell version of run_fold.py script.
#
# This version directly invokes Python interpreter to execute run.py script with
# a given execution parameters specified as a list of lines in .txt file. The
# given parameters are extended with additional information about validation
# keys and unique identifier to designate executed script.
#
# The key difference between Python-based implementation is that no reflection
# is used to dynamically load an experiment but a dedicated process is executed.
# It also helps to ensure that after a single fold training is done, the
# CUDA-allocated memory is freed completely. So it could help to deal with memory
# access errors.


# -----------------
# Global parameters
# -----------------

export PYTHONPATH=`pwd`
export DATASET_ROOT=/mnt/fast/data/kidney
export EXPERIMENT=smp
export VAL_NAME=simple_k_fold_4
export RUN_FILE=


# -----------------------
# Parsing input arguments
# -----------------------

while [ "$#" -gt 0 ]; do
  case $1 in
    --experiment) EXPERIMENT="${2}"; shift ;;
    --run-file) RUN_FILE="${2}"; shift ;;
    --val-name) VAL_NAME="${2}"; shift ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

if [ -z "${RUN_FILE}" ]
then
  echo "Error: cannot start validation if RUN_FILE parameter is unset."
  exit 1
fi

if [ -z "${VAL_NAME}" ]
then
  echo "Error: cannot start validation if VAL_NAME parameter is unset."
  exit 1
fi

export VALIDATION_SCHEME="`pwd`/validation/${VAL_NAME}.txt"

echo "Selected validation scheme: ${VALIDATION_SCHEME}"

# ----------------
# Helper functions
# ----------------

function get_valid_keys {
  train_keys=
  valid_keys=
  while IFS=" " read -ra SPLIT; do
    for i in "${SPLIT[@]}"; do
      if [ -z "$train_keys" ]; then
        train_keys=$i
      elif [ -z "$valid_keys" ]; then
        valid_keys=$i
      fi
    done
  done <<< "$1"
  echo $valid_keys
}

function random_string {
  size=${1:-10}
  cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w ${size} | head -n 1
}

function create_extended_run_args {
  unique_id=${1}
  fold=${2}
  tmp_dir=/tmp/${unique_id}
  filename=${tmp_dir}/fold_${fold}.txt
  mkdir -p ${tmp_dir}
  cp ${3} ${filename}
  echo ${filename}
}

function generate_timestamp {
  date +"%a_%d_%b__%H_%M_%S"
}

function extend_run_file {
  filename=${1}
  shift
  while [ "$#" -gt 0 ]; do
    printf "\n${1}" >> ${filename}
    shift
  done
}

function execute_python_script {
  python run.py ${1} ${2}
  return_status=$?
  if [ $return_status -ne 0 ]; then
    echo
    echo "!!! Warning: training fold ${3} failed !!!"
    echo
  else
    echo "The fold ${3} is done"
  fi
}


# --------------------------
# K-fold validation training
# --------------------------

if [ -f "${VALIDATION_SCHEME}" ]; then

  echo "Running K-fold training with validation scheme: ${VALIDATION_SCHEME}"
  unique_id=`random_string`
  fold=0

  while read line; do

    if [ -z "$line" ]; then
      continue
    fi

    echo "Executing fold: ${fold}"

    valid_keys=$(get_valid_keys "${line}")
    filename=$(create_extended_run_args ${unique_id} ${fold} ${RUN_FILE})
    timestamp=$(generate_timestamp)
    extra_arguments=(
      "--fold=${valid_keys}"
      "--tags=id:${unique_id},fold_no:${fold},valid_keys:${valid_keys},impl:${EXPERIMENT}"
      "--experiment_name=${EXPERIMENT}_${unique_id}"
      "--timestamp=${timestamp}"
      "--data_loader_multiprocessing_context=spawn"
    )

    extend_run_file "${filename}" "${extra_arguments[@]}"

    echo "Starting training with validation keys: ${valid_keys}"

    execute_python_script ${EXPERIMENT} ${filename} ${fold}

    fold=$((fold+1))

    sleep 1

  done < ${VALIDATION_SCHEME}

  echo "K-fold training '${EXPERIMENT}_${unique_id}' is done!"

else

  echo "Error: validation scheme is not found: ${VALIDATION_SCHEME}"
  exit 1

fi
