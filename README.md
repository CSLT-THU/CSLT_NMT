# ViVi_NMT: An Open Source Toolkit for Neural Machine Translation
## Contents
* [Version](#version)
* [Introduction](#introduction)
* [User Manual](#user-manual)
* [License](#license)
* [Development Team](#development-Team)
* [Contact](#contact)

## Version
1.0

## Introduction

ViVi_NMT(v1.0) is a neural machine translation toolkit developed by [Center for Speech and Language Technologies](http://cslt.riit.tsinghua.edu.cn/). 

On top of Tensorflow, this toolkit mainly reproduced the RNNsearch model proposed by Bahdanau et al.[1]

Note that, this code is modified on the ViVi_NMT(v0.1) code（upgrade code to make it support tensorflow 1.0 and later version）, and the 'multi_bleu.perl' script is downloaded from Moses.


## User Manual

### Installation

#### System Requirements

* Linux or MacOS
* Python 2.7

We recommand to use GPUs:

* NVIDIA GPUs 
* cuda 8.0

#### Installing Prerequisites

##### CUDA 8.0 environment
Assume CUDA 8.0 has been installed in "/usr/local/cuda-8.0/", then environment variables need to be set:

```
export PATH=/usr/local/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH 
```

##### Tensorflow 1.0
To have tensorflow 1.0 installed, serval methods can be applied. Here, we only introduce the installation through virtualenv. And we install the tensorflow-gpu, if you choose to use CPU, please install tensorflow of cpu.

```
pip install virtualenv --user
virtualenv --system-site-packages tf1.0  
source tf1.0/bin/activate
export TF_BINARY_URL=https://mirrors.tuna.tsinghua.edu.cn/tensorflow/linux/gpu/tensorflow_gpu-1.0.0-cp27-none-linux_x86_64.whl
pip install --upgrade $TF_BINARY_URL
```

##### Test installation
Get into python console, and import tensorflow. If no error is encountered, the installation is successful.


### Train
To train the model, run "translate.py" directly with default settings.

```
python translate.py
```

Model parameters and training settings can be set by command-line arguments, as follows:

```
--learning_rate: The initial learning rate of optimizer, default is 0.0005.
--learning_rate_decay_factor: Learning rate decays by this value, default is 0.99
--max_gradient_norm: Clip gradients to this norm, default is 1.0.
--batch_size: Batch size to use during training, default is 80.
--hidden_units: Size of hidden units for each layer, default is 1000.
--hidden_edim: Dimension of word embedding, default is 620.
--num_layers: Number of layers of RNN, default is 1.
--keep_prob: The keep probability used for dropout, default is 0.8.
--src_vocab_size: Vocabulary size of source language, default is 30000.
--trg_vocab_size: Vocabulary size of target language, default is 30000.
--data_dir: Data directory, default is './data'. 
--train_dir: Training directory, default is './train/.
--max_train_data_size: Limit on the size of training data (0: no limit), default is 0.
--steps_per_checkpoint: How many training steps to do per checkpoint, default is 1000.
```

Note that, we provide a sampled Chinese-English dataset in './data', with 10000 sentences in training set, 
400 sentences in development set, and another 400 sentences in testing set. We sample the training data from 
LDA corpora, and sample development and testing from NIST2005 and NIST 2003 respectively.

### Test
To test a trained model, for example, the 10000th checkpoint, run the command below.

```
python ./translate.py --model translate.ckpt-10000 --decode --beam_size 10 < data/test.src > test.trans
perl ./multi-bleu.perl data/test.trg < test.trans
```
Note that,if there are multiple target files such as test.trg0, test.trg1, test.trg2, and test.trg3, users need to set the value to the shared prefix test.trg.

Model parameters should be the same as settings when training, and other parameters for decoding are as follows.

```
--decode: True or False. Set to True for interactive decoding, default is False.
--model: The checkpoint model to load.
--beam_size: The size of beam search, default is 1, which represents a greedy search.
```


##

## License
Open source licensing is under the Apache License 2.0, which allows free use for research purposes. For commercial licensing, please email byryuer@gmail.com.

## Development Team

Project leaders: Dong Wang, Feng Yang

Project members: Shiyue Zhang, Shipan Ren, Jiyuan Zhang, Andi Zhang, Aodong Li

## Contact

If you have questions, suggestions and bug reports, please email [byryuer@gmail.com](mailto:byryuer@gmail.com).

## Reference

[1] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014.

