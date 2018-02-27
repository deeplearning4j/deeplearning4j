#!/usr/bin/env bash

pip3 install keras==1.2.2

KERAS_BACKEND=tensorflow python3 dense.py
KERAS_BACKEND=theano python3 dense.py
KERAS_BACKEND=tensorflow python3 conv2d.py
KERAS_BACKEND=theano python3 conv2d.py
KERAS_BACKEND=tensorflow python3 lstm.py
KERAS_BACKEND=theano python3 lstm.py
KERAS_BACKEND=tensorflow python3 embedding_lstm.py
KERAS_BACKEND=theano python3 embedding_lstm.py
KERAS_BACKEND=tensorflow python3 embedding_conv1d.py
KERAS_BACKEND=theano python3 embedding_conv1d.py
KERAS_BACKEND=tensorflow python3 simple_rnn.py
KERAS_BACKEND=theano python3 simple_rnn.py
KERAS_BACKEND=tensorflow python3 bidirectional_lstm.py
KERAS_BACKEND=theano python3 bidirectional_lstm.py
KERAS_BACKEND=tensorflow python3 batchnorm_conv2d.py
KERAS_BACKEND=theano python3 batchnorm_conv2d.py
KERAS_BACKEND=tensorflow python3 space_to_depth_simple.py
KERAS_BACKEND=tensorflow python3 space_to_depth_graph.py

pip3 install keras==2.1.3

KERAS_BACKEND=tensorflow python3 dense.py
KERAS_BACKEND=theano python3 dense.py
KERAS_BACKEND=tensorflow python3 conv2d.py
KERAS_BACKEND=theano python3 conv2d.py
KERAS_BACKEND=tensorflow python3 lstm.py
KERAS_BACKEND=theano python3 lstm.py
KERAS_BACKEND=tensorflow python3 embedding_lstm.py
KERAS_BACKEND=theano python3 embedding_lstm.py
KERAS_BACKEND=tensorflow python3 embedding_conv1d.py
KERAS_BACKEND=theano python3 embedding_conv1d.py
KERAS_BACKEND=tensorflow python3 simple_rnn.py
KERAS_BACKEND=theano python3 simple_rnn.py
KERAS_BACKEND=tensorflow python3 bidirectional_lstm.py
KERAS_BACKEND=theano python3 bidirectional_lstm.py
KERAS_BACKEND=tensorflow python3 batchnorm_conv2d.py
KERAS_BACKEND=theano python3 batchnorm_conv2d.py
KERAS_BACKEND=tensorflow python3 space_to_depth_simple.py
KERAS_BACKEND=tensorflow python3 space_to_depth_graph.py
