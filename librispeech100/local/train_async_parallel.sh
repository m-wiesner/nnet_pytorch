#!/bin/bash
. ./path.sh
. ./cmd.sh

# Training (batch and gpu configs)
nj=2
gpu=false
delay_updates=1
num_epochs=20
validation_spks=30
batches_per_epoch=500
perturb_spk=false
resume=
init=
priors_only=false
num_pdfs=2328 # gmm -- 9512
optim="adam"
mean_var="(True, 'norm')"

# Debugging and data dumping
debug=false
skip_datadump=true

# Model parameters
subsample=3
model=ChainTDNN #TDNN, ChainTDNN, Resnet, ChainResnet, WideResnet, ChainWideResnet
hdim=625
num_layers=12
prefinal_dim=192
layers="[[625, 3, 1], [625, 1, 3], [625, 3, 1], [625, 3, 1]]"
width=10
depth=28
bottleneck=92
dropout=0.2
objective=LFMMI #CrossEntropy, SemisupLFMMI, EBM_LFMMI

# TS Comparison Loss
teachers=""
ts_comparison_weight=0.5
ts_margin=1.0
ts_num_negatives=4


# Optimizer parameters
lr=0.0005
weight_decay=1e-05
warmup=500
decay=1e-05
fixed=0

# Chain
denom_graph=exp/chain_4/den.fst
xent=0.2
l2=0.0001

# Semisup
sgld_steps=10
sgld_buffer=10000
sgld_reinit_p=0.05
sgld_stepsize=1.0
sgld_noise=1.0
sgld_warmup=0
sgld_decay=0.0
sgld_thresh=0.1
ebm_weight=1.0
lfmmi_weight=1.0
xent_weight=1.0
sgld_replay_correction=1.0
sgld_weight_decay=1e-05
sgld_optim=sgd
l2_energy=0.0

. ./utils/parse_options.sh
if [ $# -ne 2 ]; then
  echo "Usage: ./train_nnet_pytorch.sh <datasets> <odir>"
  echo " --gpu ${gpu} --debug ${debug} --skip-datadump ${skip_datadump} --priors-only ${priors_only}"
  echo " --batches-per-epoch ${batches_per_epoch} --num-epochs ${num_epochs} --delay-updates ${delay_updates}"
  echo " --validation-spks ${validation_spks} --perturb-spk ${perturb_spk}"
  echo " --model ${model} --objective ${objective} --num-pdfs ${num_pdfs} --subsample ${subsample}"
  echo " --optim ${optim} --lr ${lr} --warmup ${warmup} --decay ${decay} --fixed ${fixed} --weight-decay ${weight_decay}"
  echo " --hdim ${hdim} --num-layers ${num_layers} --prefinal-dim ${prefinal_dim} --dropout ${dropout}"
  echo " --layers ${layers} --bottleneck ${bottleneck}"
  echo " --width ${width} --depth ${depth}"
  # Print Ojective specific parameters
  if [[ $objective == "LFMMI" || $objective == "SemisupLFMMI" ]]; then
    echo " --xent ${xent} --l2 ${l2} --denom-graph ${denom_graph}"
  elif [[ $objective == "SemisupLFMMI" || $objective == "LFMMI_EBM" ]]; then
    echo " --sgld-steps ${sgld_steps} --sgld-buffer ${sgld_buffer} --sgld-reinit-p ${sgld_reinit_p} --sgld-stepsize ${sgld_stepsize} --sgld-noise ${sgld_noise} --ebm-weight ${ebm_weight} --lfmmi-weight ${lfmmi_weight} --sgld-optim ${sgld_optim} --sgld-replay-correction ${sgld_replay_correction} --xent ${xent} --l2 ${l2} --denom-graph ${denom_graph} --l2-energy ${l2_energy} --sgld-warmup ${sgld_warmup} --sgld-decay ${sgld_decay} --sgld-thresh ${sgld_thresh} --sgld-weight-decay ${sgld_weight_decay}"
  elif [[ $objective == "CrossEntropy_EBM" ]]; then
    echo " --sgld-steps ${sgld_steps} --sgld-buffer ${sgld_buffer} --sgld-reinit-p ${sgld_reinit_p} --sgld-stepsize ${sgld_stepsize} --sgld-noise ${sgld_noise} --ebm-weight ${ebm_weight} --xent-weight ${xent_weight} --sgld-optim ${sgld_optim} --sgld-replay-correction ${sgld_replay_correction} --xent ${xent} --l2 ${l2} --denom-graph ${denom_graph} --l2-energy ${l2_energy} --sgld-warmup ${sgld_warmup} --sgld-decay ${sgld_decay} --sgld-thresh ${sgld_thresh} --sgld-weight-decay ${sgld_weight_decay}"
  fi

  exit 1; 
fi

datasets=$1
odir=$2

[ -z $denom_graph ] && [ $objective = 'LFMMI' ] && exit 1; 
mkdir -p ${odir}

###############################################################################
# This whole section is just setting a bunch of training options
###############################################################################

if $debug; then
  gpu=false
  num_epoch=2
  validation_spks=1
  batches_per_epoch=3
fi

# GPU vs. CPU training command
if $gpu; then
  gpu_opts="--gpu"
  train_cmd="utils/queue.pl --mem 2G --gpu 1 --config conf/gpu.conf" 
fi

# Dumpy data (Memory map). We only need to do this once and can skip it after
# that
skip_datadump_opts=""
if $skip_datadump; then
  skip_datadump_opts="--skip-datadump"
fi

if [ ! -z $init ]; then
  init_opts="--init ${init}"
fi

# Get priors only
priors_only_opts=""
if $priors_only; then
  priors_only_opts="--priors-only"
fi

# Objective Function options
obj_fun_opts=""
if [[ $objective = "LFMMI" || $objective = "SemisupLFMMI" ]]; then
  obj_fun_opts="--denom-graph ${denom_graph} --xent-reg ${xent} --l2-reg ${l2}"
fi

if [[ $objective = "SemisupLFMMI" || $objective = "LFMMI_EBM" ]]; then
  obj_fun_opts="${obj_fun_opts} --sgld-steps ${sgld_steps} --sgld-buffer ${sgld_buffer} --sgld-reinit-p ${sgld_reinit_p} --sgld-stepsize ${sgld_stepsize} --sgld-noise ${sgld_noise} --ebm-weight ${ebm_weight} --lfmmi-weight ${lfmmi_weight} --denom-graph ${denom_graph} --l2-energy ${l2_energy} --sgld-warmup ${sgld_warmup} --sgld-decay ${sgld_decay} --sgld-thresh ${sgld_thresh} --sgld-replay-correction ${sgld_replay_correction} --sgld-optim ${sgld_optim} --sgld-weight-decay ${sgld_weight_decay}" 
fi

if [[ $objective = "CrossEntropy_EBM" ]]; then
  obj_fun_opts="${obj_fun_opts} --sgld-steps ${sgld_steps} --sgld-buffer ${sgld_buffer} --sgld-reinit-p ${sgld_reinit_p} --sgld-stepsize ${sgld_stepsize} --sgld-noise ${sgld_noise} --ebm-weight ${ebm_weight} --xent-weight ${xent_weight} --l2-energy ${l2_energy} --sgld-warmup ${sgld_warmup} --sgld-decay ${sgld_decay} --sgld-thresh ${sgld_thresh} --sgld-replay-correction ${sgld_replay_correction} --sgld-optim ${sgld_optim} --sgld-weight-decay ${sgld_weight_decay}" 
fi

# Model options
mdl_opts=()
if [[ $model = "TDNN" || $model = "ChainTDNN" ]]; then
  mdl_opts=('--tdnn-hdim' "${hdim}" '--tdnn-num-layers' "${num_layers}" '--tdnn-dropout' "${dropout}" '--tdnn-prefinal-dim' "${prefinal_dim}")
elif [[ $model = "Resnet" || $model = "ChainResnet" ]]; then
  #mdl_opts="${mdl_opts} --resnet-bottleneck ${bottleneck} --resnet-layers [[625, 3, 1], [625, 1, 3], [625, 3, 1], [625, 3, 1]] --resnet-hdim ${hdim}"
  mdl_opts=('--resnet-bottleneck' "${bottleneck}" '--resnet-hdim' "${hdim}" '--resnet-layers' "${layers}")
elif [[ $model = "WideResnet" || $model = "ChainWideResnet" ]]; then
  mdl_opts=('--width' "${width}" '--depth' "${depth}")
fi

###############################################################################
# Training (IMPORTANT PART)
###############################################################################
echo""
echo "--------------- `date` -------------"
echo ""
# Train

resume_opts=
start_epoch=1
if [ ! -z $resume ]; then
  resume_opts="--resume ${resume}.mdl"
  start_epoch=$(( ${resume} + 1))
fi

[ -f ${odir}/.error ] && rm ${odir}/.error

for e in `seq ${start_epoch} ${num_epochs}`; do
  epoch_seed=`echo $nj $e | awk '{print $1*($2-1) + 1}'`
  (
    for j in `seq 1 ${nj}`; do
      job_seed=$(($epoch_seed + $j))
      ${train_cmd} ${odir}/train.${e}.${j}.log \
        train.py ${gpu_opts} ${resume_opts} ${skip_datadump_opts} \
          ${obj_fun_opts} \
          "${mdl_opts[@]}" \
          --subsample ${subsample} \
          --model ${model} \
          --objective ${objective} \
          --num-targets ${num_pdfs} \
          --expdir ${odir} \
          --datasets "${datasets}" \
          --mean-var "${mean_var}" \
          --batches-per-epoch ${batches_per_epoch} \
          --delay-updates ${delay_updates} \
          --num-epochs 1 \
          --validation-spks ${validation_spks} \
          --optim ${optim} \
          --weight-decay ${weight_decay} \
          --lr ${lr} \
          --warmup ${warmup} \
          --decay ${decay} \
          --fixed ${fixed} \
          --job ${j} \
          --seed ${job_seed} || touch ${odir}/.error &
      sleep 5
    done
    wait
  )
  
  [ -f ${odir}/.error ] && echo "$0: error on iteration ${e} of training" && exit 1;
  
  # Model averaging
  combine_models=""
  for j in `seq 1 $nj`; do
    combine_models="${combine_models} ${odir}/${e}.${j}.mdl"
  done
  combine_models.py ${odir}/${e}.mdl 80 ${odir}/conf.1.json --models ${combine_models}
  
  resume_opts="--resume ${e}.mdl"
done


