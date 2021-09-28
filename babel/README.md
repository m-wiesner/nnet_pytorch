
                            Running the example



1. Set the trainin language by writing the language code to conf/train.list


2. Prepare HMM-GMM system for alignments with

       ./run.sh --FLP true
       ./run.sh --FLP false

to train using the FLP or LLP datasets respectively


3. Run the wav2vec2 or wrn systems with


   ./local/prepare_wav2vec2.sh
   ./run-wav2vec2.sh

  ./local/prepare_wrn.sh
  ./run-wrn.sh


4. Create the test data with

   ./local/create_raw_dev10h.sh
   ./local/create_fbank_dev10h.sh

5. Create the decoding graph and language model with

  If true, the training data is also created and the LM is trained from the
  training transcripts. If false, it assumes that data/multi_train already
  exists


  ./local/prepare_decoding.sh --prepare-training-data {true,false}


  For example you can run the script like this ...

  ./local/prepare_decoding.sh --prepare-training-data true --lang-orig ../babel_201/data/lang_multip/tri5_ali --dict-orig ../babel_201/exp/multi/dictp/tri5_ali --chaindir chain_wrn_3500 --graphname graph_FLP


6. Decode with


# WRN
   ./decode.sh --chaindir ${chaindir} \
               --model-dirname ${modelname} \
               --checkpoint ${checkpoint}.mdl \
               --subsampling 4 \
               --acwt 1.0 \
               --cw 220 \
               --feat-affix "_fbank_64" \
               --decode-nj 400 \
               --graph ${graphname}

# Wav2Vec2
   ./decode.sh --chaindir ${chaindir} \
               --model-dirname ${modelname} \
               --checkpoint ${checkpoint}.mdl \
               --subsampling 640 \
               --acwt 1.0 \
