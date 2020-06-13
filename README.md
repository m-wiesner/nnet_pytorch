                   
                   
               Matching Implicit Marginals of Conditional Models
                        in Semi-supervised Training for
                             Structured Prediction

-------------------------------------------------------------------------------
This is the code used to run the experiments in NeurIPS 2020 submission with
the above title. We will utimately provide 2 ASR examples:

1. LibriLight (10h subset of Librispeech)
2. BABEL Georgian

For now we have the LibriLight example. The Librispeech data is freely
available, while the Georgian data is available for purchase from the LDC.
The LibriLight example should (I hope) be functional. 

This repository has 2 main dependencies (also see requirements.txt):

1. Kaldi (For decoding and HMM-GMM training to get targets)
2. PyChain (For LFMMI in python)

Installation:
-------------------------------------------------------------------------------
1. Go to tools and write

make all

This will install kaldi, pychain, openfst, set up a virtual environment

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

The SGLD sampling is implemented in:

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

Neural Network Training
-------------------------------------------------------------------------------

Train Supervised Baseline: ./run.sh --stage 13

Train Semisupervised:      ./run.sh --stage 14

Decode:                    ./run.sh --stage 15

Generate:                  ./run.sh --stage 16
