# GenomeBert

This repository contains code and pre-trained weights for GenomeBert.


## Usage

### Requirement

```
## System
Ubuntu 18.04
gcc 7.5.0

## Conda environment
cudatoolkit               10.0.130                      0    defaults
cudnn                     7.6.5                cuda10.0_0    defaults
...
keras                     2.3.1                         0    defaults
keras-applications        1.0.8                      py_1    defaults
keras-base                2.3.1                    py36_0    defaults
keras-preprocessing       1.1.2              pyhd3eb1b0_0    defaults
pandas                    1.1.5            py36ha9443f7_0    defaults
python                    3.6.9                h265db76_0    defaults
...
tensorflow                2.0.0           gpu_py36h6b29c10_0    defaults
tensorflow-base           2.0.0           gpu_py36h0ec5d1f_0    defaults
tensorflow-estimator      2.0.0              pyh2649769_0    defaults
tensorflow-gpu            2.0.0                h0d30ee6_0    defaults

```



### Installation

As a prerequisite, you must have Tensorfolw-gpu 2.0.0 installed to use this repository.

You can use this three-liner for installation:

```shell
conda create --name logo python==3.6.9 tensorflow-gpu==2.0 keras==2.3.1 numpy pandas tqdm scipy scikit-learn matplotlib jupyter notebook nb_conda
source activate logo
pip install biopython==1.68
```


## Pre-training model

Check out the file “01_Pre-training_Model/README.txt”

### 1. Data preprocessing
From https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.25_GRCh37.p13/
Download GCF_000001405.25_GRCh37.p13_genomic.fna.gz, And unzip, for example, unzip to /data/hg19/GCF_000001405.25_GRCh37.p13_genomic.fna

### 2. To generate ref sequence, , about 100G of space is required:
```shell  
bash 00_generate_refseq_sequence.sh
```

### 3. To generate tfrecord, about 170G of space is required (different kmer requires slightly different storage space, kmer=3, 4, 5, 6)
```shell  
bash 01_generate_DNA_refseq_tfrecord.sh
```

### 4. Perform DNA sequence pre-training, respectively (kmer=3,4,5,6, perform training)
```shell 
./02_run_genebert_bert4keras_tfrecord_train.sh
```

## 02_Onco_TSG_gene

Check out the file “2_Onco_TSG_gene/04_LOGO_Chrom_919/README.txt”

### 1. Data preparation
From https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.25_GRCh37.p13/
Download GCF_000001405.25_GRCh37.p13_genomic.fna.gz,And unzip, for example, unzip to /data/GCF_000001405.25_GRCh37.p13_genomic.fna

```shell 
python ./CGC_preprocessing.py
```

### 2. Convert to kmer sequence file, kmer is 3, 4, 5, 6 respectively  

```shell 
bash 00_run_deepsea_data_loader.sh
```

### 3. Generate tfrecord file 

```shell 
bash 01_run_deepsea_tfrecord_utils.sh
```

### 4. Carry out LOGO_Chrom_919 training and testing  

```shell 
bash 02_run_deepsea_classification_train.sh
```

## 03_Cancer_Driver_identification

Check out the file “03_Cancer_Driver/05_LOGO_C2P/README.txt”

### 1. Predicting cancerous effects using finetune genomebert (02_Onco_TSG_gene/04_LOGO_Chrom_predict/run_GeneBert_919.sh)

```shell 
bash run_GeneBert_919.sh
```

### 2. train xgboost model   

```shell 
python B_ML_train_test.py
```

