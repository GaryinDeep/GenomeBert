#!/bin/bash

python 01_deepsea_tfrecord_utils.py \
  --data ./data/Cosmic/train_6_gram \
  --output ./data/Cosmic/train_6_gram_classification_tfrecord_1000 \
  --ngram 6 \
  --stride 1 \
  --slice 200000 \
  --pool-size 24 \
  --task classification


python 01_deepsea_tfrecord_utils.py \
  --data ./data/Cosmic/test_6_gram \
  --output ./data/Cosmic/test_6_gram_classification_tfrecord_1000 \
  --ngram 6 \
  --stride 1 \
  --slice 200000 \
  --pool-size 24 \
  --task classification


python 01_deepsea_tfrecord_utils.py \
  --data ./data/Cosmic/valid_6_gram \
  --output ./data/Cosmic/valid_6_gram_classification_tfrecord_1000 \
  --ngram 6 \
  --stride 1 \
  --slice 200000 \
  --pool-size 24 \
  --task classification



