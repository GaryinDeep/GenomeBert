#!/bin/bash

# 3-gram
python 01_generate_DNA_refseq_tfrecord.py \
  --data ./data/hg19/train_3_gram \
  --output ./data/hg19/train_3_gram_tfrecord \
  --chunk-size 10000 \
  --seq-size 1000 \
  --seq-stride 100 \
  --ngram 3 \
  --stride 3 \
  --slice-size 100000 \
  --hg-name hg19 \
  --pool-size 32

# 4-gram
python 01_generate_DNA_refseq_tfrecord.py \
  --data ./data/hg19/train_4_gram \
  --output ./data/hg19/train_4_gram_tfrecord \
  --chunk-size 10000 \
  --seq-size 1000 \
  --seq-stride 100 \
  --ngram 4 \
  --stride 4 \
  --slice-size 100000 \
  --hg-name hg19 \
  --pool-size 32

# 5-gram
python 01_generate_DNA_refseq_tfrecord.py \
  --data ./data/hg19/train_5_gram \
  --output ./data/hg19/train_5_gram_tfrecord \
  --chunk-size 10000 \
  --seq-size 1000 \
  --seq-stride 100 \
  --ngram 5 \
  --stride 5 \
  --slice-size 100000 \
  --hg-name hg19 \
  --pool-size 32

# 6-gram
python 01_generate_DNA_refseq_tfrecord.py \
  --data ./data/hg19/train_6_gram \
  --output ./data/hg19/train_6_gram_tfrecord \
  --chunk-size 10000 \
  --seq-size 1000 \
  --seq-stride 100 \
  --ngram 6 \
  --stride 6 \
  --slice-size 100000 \
  --hg-name hg19 \
  --pool-size 32

