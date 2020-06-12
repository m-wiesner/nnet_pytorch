export KALDI_ROOT=`pwd`/../../..
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/sph2pipe_v2.5:$KALDI_ROOT/tools/openfst/bin:`pwd`/../nnet_pytorch:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

export OPENFST_PATH= #/PATH/TO/OPENFST
export LD_LIBRARY_ORIG=${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${OPENFST_PATH}/lib:/usr/local/cuda/lib64

export PYTHONPATH=${PYTHONPATH}:`pwd`/../nnet_pytorch/
export PYTHONUNBUFFERED=1
. `pwd`/../neurips_env/bin/activate

export LC_ALL=C

