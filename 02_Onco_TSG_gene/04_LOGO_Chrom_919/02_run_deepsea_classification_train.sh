#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python 02_deep_sea_train_classification_tfrecord.py \
  --save ./data/model_6_gram \
  --weight-path ./data/genealbert_6_gram_2_layer_8_heads_256_dim_weights_48-0.888665.hdf5 \
  --train-data ./data/Cosmic/train_6_gram_classification_tfrecord_1000 \
  --test-data ./data/Cosmic/test_6_gram_classification_tfrecord_1000 \
  --valid-data ./data/Cosmic/valid_6_gram_classification_tfrecord_1000 \
  --seq-len 1000 \
  --we-size 128 \
  --model-dim 256 \
  --transformer-depth 2 \
  --num-heads 8 \
  --batch-size 512 \
  --ngram 6 \
  --stride 1 \
  --num-classes 2 \
  --model-name cosmic_6_gram_2_layer_8_heads_256_dim_1000 \
  --use-conv \
  --use-position \
  --verbose 1 \
  --task train


CUDA_VISIBLE_DEVICES=2 python 02_deep_sea_train_classification_tfrecord.py \
  --save ./data/model \
  --weight-path ./data/model_6_gram/cosmic_6_gram_2_layer_8_heads_256_dim_1000_weights_149-0.971831-0.766891.hdf5 \
  --train-data ./data/Cosmic/train_6_gram_classification_tfrecord_1000 \
  --test-data ./data/Cosmic/test_6_gram \
  --valid-data ./data/Cosmic/valid_6_gram \
  --seq-len 1000 \
  --we-size 128 \
  --model-dim 256 \
  --transformer-depth 2 \
  --num-heads 8 \
  --batch-size 512 \
  --ngram 6 \
  --stride 1 \
  --num-classes 2 \
  --model-name cosmic_6_gram_2_layer_8_heads_256_dim_1000 \
  --use-conv \
  --use-position \
  --verbose 1 \
  --task test





