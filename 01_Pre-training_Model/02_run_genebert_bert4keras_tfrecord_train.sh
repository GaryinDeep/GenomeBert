#!/bin/bash

# 3_gram
CUDA_VISIBLE_DEVICES=0,1,2,3 python 02_train_gene_transformer_lm_hg_bert4keras_tfrecord.py \
  --save ./data/model/3_gram \
  --train-data ./data/hg19/train_3_gram_tfrecord \
  --seq-len 1000 \
  --model-dim 256 \
  --transformer-depth 2 \
  --num-heads 8 \
  --batch-size 256 \
  --ngram 3 \
  --stride 3 \
  --model-name genealbert_3_gram_2_layer_8_heads_256_dim \
  --steps-per-epoch 4000 \
  --shuffle-size 4000

# 4_gram
CUDA_VISIBLE_DEVICES=0,1,2,3 python 02_train_gene_transformer_lm_hg_bert4keras_tfrecord.py \
  --save ./data/model/4_gram \
  --train-data ./data/hg19/train_4_gram_tfrecord \
  --seq-len 1000 \
  --model-dim 256 \
  --transformer-depth 2 \
  --num-heads 8 \
  --batch-size 256 \
  --ngram 4 \
  --stride 4 \
  --model-name genealbert_4_gram_2_layer_8_heads_256_dim \
  --steps-per-epoch 4000 \
  --shuffle-size 4000

# 5_gram
CUDA_VISIBLE_DEVICES=0,1,2,3 python 02_train_gene_transformer_lm_hg_bert4keras_tfrecord.py \
  --save ./data/model/5_gram \
  --train-data ./data/hg19/train_5_gram_tfrecord \
  --seq-len 1000 \
  --model-dim 256 \
  --transformer-depth 2 \
  --num-heads 8 \
  --batch-size 256 \
  --ngram 5 \
  --stride 5 \
  --model-name genealbert_5_gram_2_layer_8_heads_256_dim \
  --steps-per-epoch 4000 \
  --shuffle-size 4000 

# 6_gram
CUDA_VISIBLE_DEVICES=0,1,2,3 python 02_train_gene_transformer_lm_hg_bert4keras_tfrecord.py \
  --save ./data/model/6_gram \
  --train-data ./data/hg19/train_6_gram_tfrecord \
  --seq-len 1000 \
  --model-dim 256 \
  --transformer-depth 2 \
  --num-heads 8 \
  --batch-size 256 \
  --ngram 6 \
  --stride 6 \
  --model-name genealbert_6_gram_2_layer_8_heads_256_dim \
  --steps-per-epoch 4000 \
  --shuffle-size 4000

# load previous weight
CUDA_VISIBLE_DEVICES=0,1,2,3 python 02_train_gene_transformer_lm_hg_bert4keras_tfrecord.py \
  --save ./data/model/5_gram/1 \
  --train-data ./data/hg19/train_5_gram_tfrecord \
  --seq-len 1000 \
  --model-dim 256 \
  --transformer-depth 2 \
  --num-heads 8 \
  --batch-size 256 \
  --ngram 5 \
  --stride 5 \
  --model-name genealbert_5_gram_2_layer_8_heads_256_dim \
  --steps-per-epoch 4000 \
  --shuffle-size 4000 \
  --weight-path ./data/model/5_gram/genealbert_5_gram_2_layer_8_heads_256_dim_weights_39-0.884426.hdf5
