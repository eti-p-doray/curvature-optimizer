#!/bin/bash

module restore jax_modules

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$EBROOTCUDNN/lib64"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"

source .env/bin/activate