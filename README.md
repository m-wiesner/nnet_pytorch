                   
                              nnet_pytorch    
                
-------------------------------------------------------------------------------
This code is intended to replace some of the Kaldi NNET frame work.
For this reason  We're referring to it as nnet_pytorch. The intention was to 
replicate Kaldi's nnet framework and training style with the following
modifications:

1. **Training examples are all created on the fly**: 
  Instead of dumping egs as in Kaldi, we dump memory mapped versions of
  features. It therefore very unlikely to ever see the same chunk of speech
  during training. We have not yet implemented on-the-fly numerator lattice
  creation. For this reason, training at this time uses only the single best
  alignment for the numerator lattice. This speeds up the gradient computation
  in LFMMI since we only have to do forward-backward on the denominator lattice.

2. **Pytorch native optimizers (Adam) instead of Natural Gradient**:
  All provided examples use pytorch-native optimizers. We generally train with
  1/3-1/2 of the training train used for warmup steps, and then decay the
  learning rate exponentially until the end of training. Other learning rate
  schedules could be added, as well as other optimizers.
 
3. **Model Averaging at the end of training**
  We find that averaging the last 20+ models of training almost always gives
  about 10% relative improvement. Unlike Kaldi, we do not use performance on a
  dev set for weighting when averaging models.


TODO
-------------------------------------------------------------------------------
**Online decoding**:
  Most models can be implemented in pytorch, but we have not yet supported
  any sort of online decoding, though it should be feasible.

**Training with Ivector or other speaker representations**:
  We do not yet support training with i-vectors though this should also be
  easy to implement.

**Online numerator lattice creation**
  There should be a way of easily supporting some sort of notion of alternative 
  paths in training without dumping a numerator lattice. Even just allowing
  alternative alignments should be easily doable though it is not yet
  implemented.

**On-the-fly Data perturbations**
  We have some rudimentary noising of audio examples implemented though we
  haven't really tried them out. We should also be able to support
  on-the-fly feature computation as an alternative to dumping the memory mapped
  features, even if it increases the training time somewhat.

**More compact representation of targets**
  We currently are writing the target sequences as raw text files. These should
  be gzipped, or maybe we should support loading them as needed on the fly. Also
  the way we currently deal with unlabeled data is somehwat ugly and should be fixed.
  Currently, a when creating minibatches we expect targets to exist. For unlabeled data
  or data we want to decode, we supply dummy targets consisting of -1 for all outputs.
  This is obviously a little wasteful, but on academic datasets, which are fairly small,
  this doesn't cause any problems.
  
**Create base class for models, objectives and optimizers with add_state_dict()**
  We need to define how to average state dicts from different models and
  objectives. For now this is handled in the script combine_models.py 


Example Recipes
-------------------------------------------------------------------------------
For now all of the examples are based on librispeech, though any existing 
kaldi recipe can be easily modified to use nnet_pytorch instead of nnet3. See
librispeech100 for a full example. The recipe in librilight probably needs to
be updated. We have also trained good performing ASR models for the BABEL data
sets with this code, but have not yet committed the recipe.

The procedure for making new recipes:

1. Use normal kaldi recipe to produce good HMM-GMM models.
2. Create the chain directory with the appropriate subsampling factor (model specific).
3. Create features for nnet_pytorch training (80-dim fbank features normally)
4. run local/split_memmap_data.sh to create memmapped versions of the features. These are readable in numpy.
5. run either ali-to-pdf to create training targets or ./local/prepare_unlabeled_tgt.sh to create the
   targets for labeled or unlabeled data. 

This repository has 2 main dependencies (also see requirements.txt):

1. Kaldi (For decoding and HMM-GMM training to get targets)
2. PyChain (For LFMMI in python)


For Pychain, I added a patch to make it more compatible with existing code, but
this may be unstable and could require changing if pychain continues to change.

Installation:
-------------------------------------------------------------------------------
1. Go to tools and write

make all

This will install kaldi, pychain, openfst, and set up a conda virtual environment

To run experiments, users will have to modify a few files.

librilight/cmd.sh -- contains commands for training and decoding. These commands
may need to be modified for new computing clusters.

librilight/conf/gpu.conf -- gpu configurations that may also need to be changed.

The CUDA_VISIBLE_DEVICES environment variable is set internally in the code.
Users should modify this line in the following files. It is indicated by 
many commented lines before and after with a note:

1. train.py
2. decode.py,
3. generate_conditional_from_buffer.py

Furthermore, some data will need to be downloaded for these examples. The
download takes long enough, and is large enough that it is not included in
the script. See the README in librilight for more information about using
the unlabeled data.


CORE ALGORITHMS:
------------------------------------------------------------------------------
The core algorithms can be found in nnet_pytorch. The model Archictecture is
defined in: 

  nnet_pytorch/models/WideResnet.py

For generative modeling the SGLD sampling is implemented in:

  nnet_pytorch/objectives/SGLDSampler.py 
  nnet_pytorch/objectives/LFMMI_EBM.py 

The Semi-supervised learning is mostly found in:

  nnet_pytorch/objectives/SemisupLFMMI.py


We describe how to run the librilight example:
-------------------------------------------------------------------------------
This is the example we used to produce the Libripseech results in our paper.


First go to path.sh and set all paths accordingly.
Then go to cmd.sh and possibly change the default training and decode submission
commands. Go to conf/gpu.conf and change according to computing environment.

All of the relevant steps for running an example are included in run.sh.
Simply executing ./run.sh should prepare the targets, and training data and
stop execution just before neural network training.

Data preparation, which takes a few hours (1-3h I think) is accomplished by
running:

./run.sh

After running this, additional tasks can be executed.

Train Supervised WideResNet Baseline: ./run-wrn.sh
Train Semisupervised WideResNet:      ./run-wrn-semisup.sh
Decode:                               ./decode.sh
Generate:                             ./generate.sh

Relevant flags can be set to modify training, decoding, and generation
parameters.
