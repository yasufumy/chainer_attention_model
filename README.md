# A Neural Attention Model for Machine Translation

## Setup

```sh
$ git clone https://github.com/yasfmy/chainer_attention_model.git
$ cd chainer_attention_model
$ git submodule init
$ git submodule update
$ cd tools
$ python setup.py install
```

## Usage

```sh
$ python train_attmt.py --epoch [num of epoch] --embed [embed size] --hidden [hidden size] \
    --batch [batch size] --unk [threshold of unknown words] --gpu [gpu id] \
    --train_src [path to train data for source language] --train_trg [path to train data for target language] \
    --test_src [path to test data for source language] --test_trg [path to test data for target language] \
    --model [filename of model parameter] --output [filename of output of test data]
```
