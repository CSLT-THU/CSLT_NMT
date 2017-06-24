# CSLT_NMT: An Open Source Toolkit for Neural Machine Translation
## Contents
* [Introduction](#introduction)
* [User Manual](#user-manual)
* [License](#license)
* [Development Team](#development-Team)
* [Contact](#contact)

## Introduction

CSLT_NMT is a neural machine translation toolkit developed by [Center of Speech and Language Technology](http://cslt.riit.tsinghua.edu.cn/). 

On top of [Tensorflow]( https://www.tensorflow.org/), this toolkit mainly reproduced the RNNsearch model proposed by Bahdanau et al.[1] 

Note that, this code is modified on the original seq2seq model code from [Tensorflow0.10](https://github.com/tensorflow/tensorflow/tree/r0.10), and the 'multi_bleu.perl' script is downloaded from [Moses](https://github.com/moses-smt/mosesdecoder/tree/master/scripts/generic).


## User Manual

### Installation

#### System Requirements

* Linux or MacOS
* Python 2.7

We recommand to use GPUs:

* NVIDIA GPUs 
* cuda 7.5

#### Installing Prerequisites

##### CUDA 7.5 environment
Assume CUDA 7.5 has been installed in "/usr/local/cuda-7.5/", then environment variables need to be set:

```
export PATH=/usr/local/cuda-7.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH 
```
##### Tensorflow 0.10
To have tensorflow 0.10 installed, serval methods can be applied. Here, we only introduce the installation through virtualenv. And we install the tensorflow-gpu, if you choose to use CPU, please install tensorflow of cpu.

```
pip install virtualenv --user
virtualenv --system-site-packages tf0.10  
source tf0.10/bin/activate
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl
pip install --upgrade $TF_BINARY_URL
```
##### Test installation
Get into python console, and import tensorflow. If no error is encountered, the installation is successful.

```
Python 2.7.5 (default, Nov  6 2016, 00:28:07) 
[GCC 4.8.5 20150623 (Red Hat 4.8.5-11)] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow 
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcurand.so locally
>>> 
```

### Train
To train the model, run "translate.py" directly with default settings.

```
python translate.py
```

Model parameters and training settings can be set by command-line arguments, as follows:

```
--learning_rate: The initial learning rate of optimizer, default is 0.0001.
--learning_rate_decay_factor: Learning rate decays by this value, default is 0.99
--max_gradient_norm: Clip gradients to this norm, default is 1.0.
--batch_size: Batch size to use during training, default is 80.
--hidden_units: Size of hidden units for each layer, default is 1000.
--hidden_edim: Dimension of word embedding, default is 620.
--num_layers: Number of layers of RNN, default is 1.
--keep_prob: The keep probability used for dropout, default is 1.0.
--src_vocab_size: Vocabulary size of source language, default is 30000.
--trg_vocab_size: Vocabulary size of target language, default is 30000.
--data_dir: Data directory, default is './data'.
--train_dir: Training directory, default is './train/.
--max_train_data_size: Limit on the size of training data (0: no limit), default is 0.
--steps_per_checkpoint: How many training steps to do per checkpoint, default is 1000.
```

### Test
To test a trained model, for example, to test the 10000th checkpoint, run the command below.

```
python ./translate.py --model translate.ckpt- --decode --beam_size 10 < data/test.src > test.trans
perl ./multi-bleu.perl data/test.trg < test.trans
```

Model parameters should be the same settings when training, and other parameters for decoding are as follows.

```
--decode: True or False. Set to True for interactive decoding, default is False.
--model: The checkpoint model to load.
--beam_size: The size of beam search, default is 1, which represents a greedy search.
```

## License
Open source licensing is under the Apache License 2.0, which allows free use for research purposes. For commercial licensing, please email byryuer@gmail.com.

## Development Team

Project leaders: Dong Wang, Feng Yang

Project members: Shiyue Zhang, Jiyuan Zhang, Andi Zhang, Aodong Li, Shipan Ren

## Contact

If you have questions, suggestions and bug reports, please email [byryuer@gmail.com](mailto:byryuer@gmail.com).

## Reference

[1] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014.

