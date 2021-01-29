#!/bin/bash
. ./path.sh
. ./cmd.sh

# Training (batch and gpu configs)
nj_init=2
nj_final=8
gpu=false
delay_updates=1
num_epochs=20
validation_spks=30
batches_per_epoch=500
perturb_spk=false
seed=0 # Useful for restarting with new seed
resume=
init=
priors_only=false
num_pdfs=2328 # gmm -- 9512
optim="adam"

# Debugging and data dumping
debug=false

# Dataset parameters
datasetname=HybridASR

# Model parameters
idim=80
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
objective=LFMMI #CrossEntropy, SemisupLFMMI, LFMMI_EBM, SemisupMCE, LFMMI_MCE

# Ivector params
ivector_dim=
ivector_layers=1 # By default we use i-vectors to scale the output of the first layer

# TS Comparison Loss
teachers=""
ts_comparison_weight=0.5
ts_margin=1.0
ts_num_negatives=4


# Optimizer parameters
lr=0.0005
min_lr=0.000001
weight_decay=1e-05
warmup=500
decay=1e-05
fixed=0

# Chain
denom_graph=exp/chain_4/den.fst
xent=0.2
l2=0.0001
leaky_hmm=0.1

# Semisup
sgld_steps=10
sgld_max_steps=150
sgld_buffer=10000
sgld_reinit_p=0.05
sgld_stepsize=1.0
sgld_noise=1.0
sgld_warmup=0
sgld_decay=0.0
sgld_real_decay=0.0
sgld_thresh=0.1
sgld_init_val=1.5
sgld_epsilon=1e-04
ebm_type="uncond"
ebm_joint=false
ebm_tgt=
ebm_weight=1.0
mce_weight=1.0
lfmmi_weight=1.0
xent_weight=1.0
sgld_replay_correction=1.0
sgld_weight_decay=1e-05
sgld_optim=sgd
sgld_clip=1.0
l2_energy=0.0


. ./utils/parse_options.sh

echo "Num args: $#"

if [ $# -ne 2 ] && [ $# -ne 3 ]; then
  echo "Usage: ./train_async_parallel.sh <datasets> [<validation-datasets>] <odir>"
  echo " --gpu ${gpu} --debug ${debug} --priors-only ${priors_only}"
  echo " --batches-per-epoch ${batches_per_epoch} --num-epochs ${num_epochs} --delay-updates ${delay_updates}"
  echo " --validation-datasets ${validation_datasets} --perturb-spk ${perturb_spk}"
  echo " --model ${model} --objective ${objective} --num-pdfs ${num_pdfs} --subsample ${subsample}"
  echo " --optim ${optim} --lr ${lr} --warmup ${warmup} --decay ${decay} --fixed ${fixed} --min-lr ${min_lr} --weight-decay ${weight_decay}"
  echo " --hdim ${hdim} --num-layers ${num_layers} --prefinal-dim ${prefinal_dim} --dropout ${dropout}"
  echo " --layers ${layers} --bottleneck ${bottleneck}"
  echo " --width ${width} --depth ${depth}"
  # Print Ojective specific parameters
  if [[ $objective == "LFMMI" || $objective == "SemisupLFMMI" || $objective == "SemisupMCE" ]]; then
    echo " --xent ${xent} --l2 ${l2} --denom-graph ${denom_graph} --leaky-hmm ${leaky_hmm} --lfmmi-weight ${lfmmi_weight} --mce-weight ${mce_weight}"
  elif [[ $objective == "SemisupLFMMI" || $objective == "LFMMI_EBM" ]]; then
    echo " --sgld-epsilon ${sgld_epsilon} --sgld-init-val ${sgld_init_val} --sgld-steps ${sgld_steps} --sgld-max-steps ${sgld_max_steps} --sgld-buffer ${sgld_buffer} --sgld-reinit-p ${sgld_reinit_p} --sgld-stepsize ${sgld_stepsize} --sgld-noise ${sgld_noise} --ebm-weight ${ebm_weight} --ebm-type ${ebm_type} --ebm-joint ${ebm_joint} --lfmmi-weight ${lfmmi_weight} --sgld-optim ${sgld_optim} --sgld-replay-correction ${sgld_replay_correction} --xent ${xent} --l2 ${l2} --denom-graph ${denom_graph} --leaky-hmm ${leaky_hmm} --l2-energy ${l2_energy} --sgld-warmup ${sgld_warmup} --sgld-decay ${sgld_decay} --sgld-init-real-decay ${sgld_real_decay} --sgld-thresh ${sgld_thresh} --sgld-weight-decay ${sgld_weight_decay} --sgld-clip ${sgld_clip}"
  elif [[ $objective == "CrossEntropy_EBM" ]]; then
    echo " --sgld-steps ${sgld_steps} --sgld-max-steps ${sgld_max_steps} --sgld-buffer ${sgld_buffer} --sgld-reinit-p ${sgld_reinit_p} --sgld-stepsize ${sgld_stepsize} --sgld-noise ${sgld_noise} --ebm-weight ${ebm_weight} --ebm-type ${emb_type} --ebm-joint ${ebm_joint} --xent-weight ${xent_weight} --sgld-optim ${sgld_optim} --sgld-replay-correction ${sgld_replay_correction} --xent ${xent} --l2 ${l2} --denom-graph ${denom_graph} --leaky-hmm ${leaky_hmm} --l2-energy ${l2_energy} --sgld-warmup ${sgld_warmup} --sgld-decay ${sgld_decay} --sgld-init-real-decay ${sgld_real_decay} --sgld-thresh ${sgld_thresh} --sgld-weight-decay ${sgld_weight_decay} --sgld-clip ${sgld_clip}"
  elif [[ $objective == "LFMMI_MCE" ]]; then
    echo " --denom-graph ${denom_graph} --leaky-hmm ${leaky_hmm} --lfmmi-weight ${lfmmi_weight} --mce-weight ${mce_weight}"
  fi

  exit 1; 
fi

datasets=$1
if [ $# -eq 2 ]; then
  odir=$2
else
  validation_datasets=$2
  odir=$3
fi

[ -z $denom_graph ] && [ $objective = 'LFMMI' ] && exit 1; 
mkdir -p ${odir}

###############################################################################
# This whole section is just setting a bunch of training options
###############################################################################

if $debug; then
  gpu=false
  num_epoch=2
  batches_per_epoch=3
fi

# GPU vs. CPU training command
if $gpu; then
  gpu_opts="--gpu"
  train_cmd="utils/retry.pl utils/queue.pl --mem 4G --gpu 1 --config conf/gpu.conf"
fi

if [ ! -z $init ]; then
  init_opts="--init ${init}"
fi

# Get priors only
priors_only_opts=""
if $priors_only; then
  priors_only_opts="--priors-only"
fi

# Validation datasets
validation_opts=""
if [ ! -z $validation_datasets ]; then
  validation_opts="--validation-datasets $validation_datasets"
fi

# Objective Function options
obj_fun_opts=""
if [[ $objective = "LFMMI" || $objective = "SemisupLFMMI" ]]; then
  obj_fun_opts="--denom-graph ${denom_graph} --xent-reg ${xent} --l2-reg ${l2} --leaky-hmm ${leaky_hmm}"
fi

if [[ $objective = "LFMMI_MCE" ]]; then
  obj_fun_opts="--denom-graph ${denom_graph} --lfmmi-weight ${lfmmi_weight} --mce-weight ${mce_weight}"
  if [[ $ebm_type = "cond" ]]; then
    obj_fun_opts="${obj_fun_opts} --ebm-tgt ${ebm_tgt}"
  fi
fi

if [[ $objective = "SemisupMCE" ]]; then
  obj_fun_opts="--denom-graph ${denom_graph} --xent-reg ${xent} --l2-reg ${l2} --mce-weight ${mce_weight} --lfmmi-weight ${lfmmi_weight}"
fi

if [[ $objective = "SemisupLFMMI" || $objective = "LFMMI_EBM" ]]; then
  obj_fun_opts="${obj_fun_opts} --sgld-epsilon ${sgld_epsilon} --sgld-init-val ${sgld_init_val} --sgld-steps ${sgld_steps} --sgld-max-steps ${sgld_max_steps} --sgld-buffer ${sgld_buffer} --sgld-reinit-p ${sgld_reinit_p} --sgld-stepsize ${sgld_stepsize} --sgld-noise ${sgld_noise} --ebm-weight ${ebm_weight} --ebm-type ${ebm_type} --lfmmi-weight ${lfmmi_weight} --denom-graph ${denom_graph} --leaky-hmm ${leaky_hmm} --l2-energy ${l2_energy} --sgld-warmup ${sgld_warmup} --sgld-decay ${sgld_decay} --sgld-init-real-decay ${sgld_real_decay} --sgld-thresh ${sgld_thresh} --sgld-replay-correction ${sgld_replay_correction} --sgld-optim ${sgld_optim} --sgld-weight-decay ${sgld_weight_decay} --sgld-clip ${sgld_clip}" 
  if [[ $ebm_type = "cond" ]]; then
    obj_fun_opts="${obj_fun_opts} --ebm-tgt ${ebm_tgt}"
  fi
  if $ebm_joint; then
    obj_fun_opts="${obj_fun_opts} --ebm-joint"
  fi
fi

if [[ $objective = "CrossEntropy_EBM" ]]; then
  obj_fun_opts="${obj_fun_opts} --sgld-steps ${sgld_steps} --sgld-max-steps ${sgld_max_steps} --sgld-buffer ${sgld_buffer} --sgld-reinit-p ${sgld_reinit_p} --sgld-stepsize ${sgld_stepsize} --sgld-noise ${sgld_noise} --ebm-weight ${ebm_weight} --ebm-type ${ebm_type} --xent-weight ${xent_weight} --l2-energy ${l2_energy} --sgld-warmup ${sgld_warmup} --sgld-decay ${sgld_decay} --sgld-thresh ${sgld_thresh} --sgld-replay-correction ${sgld_replay_correction} --sgld-optim ${sgld_optim} --sgld-weight-decay ${sgld_weight_decay} --sgld-clip ${sgld_clip}" 
fi

# Model options
mdl_opts=()
if [[ $model = "TDNN" || $model = "ChainTDNN" ]]; then
  mdl_opts=('--tdnn-hdim' "${hdim}" '--tdnn-num-layers' "${num_layers}" '--tdnn-dropout' "${dropout}" '--tdnn-prefinal-dim' "${prefinal_dim}")
elif [[ $model = "BLSTM" || $model = "ChainBLSTM" || $model = "BLSTMWithIvector" || $model = "ChainBLSTMWithIvector" ]]; then
  mdl_opts=('--blstm-hdim' "${hdim}" '--blstm-num-layers' "${num_layers}" '--blstm-dropout' "${dropout}" '--blstm-prefinal-dim' "${prefinal_dim}")
elif [[ $model = "Resnet" || $model = "ChainResnet" ]]; then
  #mdl_opts="${mdl_opts} --resnet-bottleneck ${bottleneck} --resnet-layers [[625, 3, 1], [625, 1, 3], [625, 3, 1], [625, 3, 1]] --resnet-hdim ${hdim}"
  mdl_opts=('--resnet-bottleneck' "${bottleneck}" '--resnet-hdim' "${hdim}" '--resnet-layers' "${layers}")
elif [[ $model = "WideResnet" || $model = "ChainWideResnet" ]]; then
  mdl_opts=('--width' "${width}" '--depth' "${depth}")
fi

# Ivector options
ivector_opts=
if [[ $model = "BLSTMWithIvector" || $model = "ChainBLSTMWithIvector" ]]; then
  mdl_opts+=('--ivector-layers' "${ivector_layers}")
  ivector_opts="--ivector-dim $ivector_dim"
fi

###############################################################################
# Training 
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
  init_opts=""
fi

[ -f ${odir}/.error ] && rm ${odir}/.error

#train_cmd="qsub -v PATH -S /bin/bash -b y -q gpu.q -cwd -j y -N train.${e}.${j}.log -l gpu=1,num_proc=10,mem_free=64G,hostname='!r3n*&!r5n*&!r6n02&!r8n*&!r2n07',h_rt=600:00:00 -o ${odir}/train.${e}.${j}.log"
for e in `seq ${start_epoch} ${num_epochs}`; do
  nj=`echo ${num_epochs} ${nj_final} ${nj_init} ${e} | awk '{print int($4*($2-$3)/$1) + $3}'`
  epoch_seed=`echo $nj $e $seed | awk '{print ($3+1)*$1*($2-1) + 1}'`
  (
    for j in `seq 1 ${nj}`; do
      job_seed=$(($epoch_seed + $j))
      ${train_cmd} ${odir}/train.${e}.${j}.log \
        train.py ${gpu_opts} ${resume_opts} ${init_opts} \
          ${obj_fun_opts} ${validation_opts} \
          "${mdl_opts[@]}" ${ivector_opts}\
          --subsample ${subsample} \
          --datasetname "${datasetname}" \
          --model ${model} \
          --objective ${objective} \
          --num-targets ${num_pdfs} \
          --expdir ${odir} \
          --datasets "${datasets}" \
          --idim "${idim}" \
          --batches-per-epoch ${batches_per_epoch} \
          --delay-updates ${delay_updates} \
          --num-epochs 1 \
          --optim ${optim} \
          --weight-decay ${weight_decay} \
          --lr ${lr} \
          --min-lr ${min_lr} \
          --warmup ${warmup} \
          --decay ${decay} \
          --fixed ${fixed} \
          --job ${j} \
          --seed ${job_seed} || touch ${odir}/.error &
      sleep 10
    done
    wait
  )
  
  [ -f ${odir}/.error ] && echo "$0: error on iteration ${e} of training" && exit 1;
  
  # Model averaging
  combine_models=""
  for j in `seq 1 $nj`; do
    combine_models="${combine_models} ${odir}/${e}.${j}.mdl"
  done
  
  combine_models.py ${odir}/${e}.mdl ${odir}/conf.1.json --models ${combine_models} > ${odir}/combine.${e}.log
  resume_opts="--resume ${e}.mdl"
  init_opts=""
done


