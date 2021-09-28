#!/bin/bash

# enable module support
source /etc/profile.d/modules.sh
module load gcc/7.2.0 || exit 1
module load cuda10.1/blas/10.1.105 || exit 1
module load cuda10.1/toolkit/10.1.105 || exit 1
module load cudnn/7.6.3_cuda10.1 || exit 1
module load intel/mkl/64/2019/5.281 || exit 1
module load nccl/2.4.7_cuda10.1 || exit 1
module load sox

. ${HOME}/nnet_pytorch/tools/activate_python.sh
