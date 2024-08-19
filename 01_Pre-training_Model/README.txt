
1. From https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.25_GRCh37.p13/ Download GCF_000001405.25_GRCh37.p13_genomic.fna.gz,
And unzip, for example, unzip to /data/hg19/GCF_000001405.25_GRCh37.p13_genomic.fna

2. To generate ref sequence, , about 100G of space is required: ./00_generate_refseq_sequence.sh  

3. To generate tfrecord, about 170G of space is required (different kmer requires slightly different storage space, kmer=3, 4, 5, 6): ./01_generate_DNA_refseq_tfrecord.sh

4. Perform DNA sequence pre-training, respectively (kmer=3,4,5,6, perform training): ./02_run_genebert_bert4keras_tfrecord_train.sh
