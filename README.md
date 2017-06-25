# MNIST recognition

implement the LeNet for digit recognition using frameworks as follows:

1. Tensorflow
2. Keras
3. Caffe
4. PyTorch

## objective

My goal in this project is to get familiar with these frameworks.

## Why MNIST

MNIST seems to be a "hello world" in DL. And also a very good choice to get hands on CNN.

## Usage

### prequiste

you will need many things, but fortunately all is easy to install and cofigue.

- python
- tensorflow
- tensorboard
- tqdm: this was used for logged imfo
- pytorch
- keras

### run

run like this

```
python main.py  --train --train_num 1000 --load --framework tensorflow
```

- --train : indicate that you want to train the model. If this flag is set on, a --train_num must followed
- --load : use while you want to load a local model
- --framework: choose your framework: tensorflow, pytorch or keras ...

> there are other flags, like the --log_dir indicating the directory to store the log info

## Experiments

Experiments were made. See `meta/doc/report.md` for detailed result and analysis.

I made some modifications on the lenet architecture in order to find what matters and what not
