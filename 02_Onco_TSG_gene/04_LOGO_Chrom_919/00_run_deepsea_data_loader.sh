#!/bin/bash

python 00_deepsea_data_loader.py \
  --data ./data/Cosmic/ \
  --output ./data/Cosmic/ \
  --ngram 6 \
  --stride 1 \
  --slice 200000 \
  --pool-size 24
