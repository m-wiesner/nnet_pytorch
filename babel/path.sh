source env.sh

export ROOT=../tools
export KALDI_ROOT=${ROOT}/kaldi
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/sph2pipe_v2.5:$KALDI_ROOT/tools/openfst/bin:${ROOT}/../nnet_pytorch:$PWD:$PATH:${ROOT}/../nnet_pytorch/utils/:${KALDI_ROOT}/tools/srilm/lm/bin/i686-m64/:${KALDI_ROOT}/tools/sctk-2.4.10/bin:${KALDI_ROOT}/tools/sctk-2.4.10/src/hubscr
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

