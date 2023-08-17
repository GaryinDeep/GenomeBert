0. From https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.25_GRCh37.p13/ Download GCF_000001405.25_GRCh37.p13_genomic.fna.gz,
And unzip, for example, unzip to /data/GCF_000001405.25_GRCh37.p13_genomic.fna

1. Data preparation   python ./CGC_preprocessing.py

2. Convert to kmer sequence file, kmer is 3, 4, 5, 6 respectively  ./00_run_deepsea_data_loader.sh

3. Generate tfrecord file ./01_run_deepsea_tfrecord_utils.sh

4. Carry out LOGO_Chrom_919 training and testing  ./02_run_deepsea_classification_train.sh


