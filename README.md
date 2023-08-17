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


## 02_Onco_TSG_gene

Check out the file “2_Onco_TSG_gene/04_LOGO_Chrom_919/README.txt”


## 03_Cancer_Driver_identification

Check out the file “03_Cancer_Driver/05_LOGO_C2P/README.txt”




