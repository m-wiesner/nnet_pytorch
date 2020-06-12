kaldi: kaldi pychain
	git clone https://github.com/kaldi-asr/kaldi.git
	cd kaldi_github/tools; $(MAKE) all
	cd kaldi_github/src; ./configure --shared; $(MAKE) depend; $(MAKE) all

venv: requirements.txt
	test -d NeurIPS2020 || python3 -m venv NeurIPS2020
	. ./NeurIPS2020/bin/activate; pip install -r requirements.txt

openfst:
	wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.7.7.tar.gz
	tar -xzvf openfst-1.7.7.tar.gz
	cd openfst-1.7.7; ./configure --prefix=${PWD}/openfst-1.7.7 --enable-static --enable-shared --enable-ngram-fsts CXX="g++" LIBS="-ldl" CPPFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"; make -j 8; make -j 8 install

pychain: venv
	. ./NeurIPS2020/bin/activate; export OPENFST_PATH=${PWD}/openfst-1.7.7; export LD_LIBRARY_PATH=${PWD}/openfst-1.7.7/lib:${LD_LIBRARY_PATH}; cd ../nnet_pytorch/objectives; git clone --single-branch --branch pack https://github.com/YiwenShaoStephen/pychain.git; cd pychain/openfst_binding; python setup.py install; cd ../pytorch_binding; python setup.py install
