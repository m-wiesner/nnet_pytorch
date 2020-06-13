To start this example commands in the following files will have to be defined
according to the computing environment on which the code is running. 

librilight/cmd.sh -- contains commands for training and decoding. These commands
may need to be modified for new computing clusters.

librilight/conf/gpu.conf -- gpu configurations that may also need to be changed.

The CUDA_VISIBLE_DEVICES environment variable is set internally in the code.
Users should modify this line in the following files. It is indicated by 
many commented lines before and after with a note:

1. train.py
2. decode.py,
3. generate_conditional_from_buffer.py


Furthermore, the Librispeech Corpus (unlabled data) will need to be downloaded.
Place the path to the Librispeech data in the variable unlabeled_data found the
run.sh script (first line of code).
