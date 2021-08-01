source env.sh # Your computing environment may require module loads for instance.
              # env.sh has all of these it also activates the nnet_pytorch conda
              # environment
              

export ROOT=${HOME}/nnet_pytorch/tools # nnet_pytorch installation
export KALDI_ROOT=${ROOT}/kaldi # Kaldi is linked or installed as part of nnet_pytorch
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/sph2pipe_v2.5:$KALDI_ROOT/tools/openfst/bin:${ROOT}/../nnet_pytorch:$PWD:$PATH:${ROOT}/../nnet_pytorch/utils/:/home/hltcoe/mwiesner/kaldi/tools/srilm/lm/bin/i686-m64
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

export OPENFST_PATH=${ROOT}/openfst #/PATH/TO/OPENFST
export LD_LIBRARY_ORIG=${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${OPENFST_PATH}/lib:${LD_LIBRARY_PATH}
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${OPENFST_PATH}/lib:/usr/local/cuda/lib64

export PYTHONPATH=${PYTHONPATH}:${ROOT}/../nnet_pytorch/:${ROOT}/../nnet_pytorch/utils/
export PYTHONUNBUFFERED=1
. ${ROOT}/activate_python.sh

export LC_ALL=C

