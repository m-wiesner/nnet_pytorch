                   
                   
                   
                   Matching Implicit Marginals of Conditional Models
                          in Semi-supervised Training for
                              Structured Prediction


DISCLAIMER: It may be challenging to get this code to run as of right now. We
are actively working to make installation of dependencies easier and intend to
fully release our code with the camera ready paper should our submission be
accepted. The PyChain dependency is very new, somewhat unstable, and has changed
since we integrated it with out code. However, all of the steps to recreate our
Librispeech results are shown in librispeech/run.sh. The steps to produce the
BABEL results follow the kaldi/egs/babel/s5d recipe. For neural network
training, the librispeech/run.sh steps are still those that are used. Some
parameters were modificatied as discussed in Appendix B.


This is the code used to run the experiments in NeurIPS 2020 submission with
the above title. We will utimately provide 2 ASR examples:

1. LibriLight (10h subset of Librispeech)
2. BABEL Georgian

For now we have the LibriLight example. The Librispeech data is freely
available, while the Georgian data is available for purchase from the LDC.
The LibriLight example should (I hope) be functional although it requires
multiple dependencies that are tricky to get working together. 

This repository has 2 main dependencies (also see requirements.txt):

1. Kaldi (For decoding and HMM-GMM training to get targets)
2. PyChain (For LFMMI in python)

kaldi_io is a python package to convert kaldi features into numpy objects.
It can be installed with 

pip install kaldi_io

See INSTALL for installation instructions for kaldi and PyChain Installation




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

After this training can be run with: 

Supervised Baseline: ./run.sh --stage 13
Semisupervised:      ./run.sh --stage 14


Generating Conditional Samples:
You can generate conditional samples by running the following command for instance. 

./utils/queue.pl --mem 2G --gpu 1 --config conf/gpu.conf exp/model/cond_generate_160.mdl/log \
  generate_conditional_from_buffer.py --gpu \
    --target 701 38 343 789 321 767 767 767 229 1084 1084 1084 560 693 473 1071 1071 1071 \
    --gpu --idim 80 --chunk-width 72 --left-context 4 --right-context 4 \
    --modeldir exp/model --modelname 160.mdl \
    --dumpdir exp/model/cond_generate_160.mdl \
    --batchsize 32
 
