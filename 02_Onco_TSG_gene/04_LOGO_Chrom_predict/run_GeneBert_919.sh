# LChuang: USAGE 
# (tf20_hhp) LChuang@bio-SYS-4028GR-TR:/alldata/LChuang_data/myP/GeneBert/BGI-Gene_new/examples/vcf-predict$
# USAGE: bash run_GeneBert_919_2002_3357.sh demo.vcf 2
# input:
#   demo.vcf
#   GPU=2
# Output:
#   .out.ref.csv
#   .out.alt.csv
#   .out.logfoldchange.csv
#   .out.evalue.csv
#   .out.evalue_gmean.csv   # main output

CUDA_VISIBLE_DEVICES=0 python GeneBert_predict_vcf_slice_e8-512pos_new.py \
--inputfile ../../03_Cancer_Driver/00_Data_Preprocessing/data/Output/vcf/multi_data_novivo.vcf \
--outputpath ../../03_Cancer_Driver/05_LOGO_C2P/data/output/multi_novivo/  \
--reffasta ./data/male.hg19.fasta \
--weight-path ./data/cosmic_6_gram_2_layer_8_heads_256_dim_1000_weights_149-0.971831-0.766891.hdf5 \
--maxshift 0 \
--seq-len 1000 \
--we-size 128 \
--model-dim 256 \
--transformer-depth 2 \
--num-heads 8 \
--batch-size 512 \
--ngram 6 \
--stride 1 \
--num-classes 2 \
--pool-size 36 \
--use-conv \
--use-position \
--task test

CUDA_VISIBLE_DEVICES=0 python GeneBert_predict_vcf_slice_e8-512pos_new.py \
--inputfile ../../03_Cancer_Driver/00_Data_Preprocessing/data/Output/vcf/in_vivo_data.vcf \
--outputpath ../../03_Cancer_Driver/05_LOGO_C2P/data/output/vivo/  \
--reffasta ./data/male.hg19.fasta \
--weight-path ./data/cosmic_6_gram_2_layer_8_heads_256_dim_1000_weights_149-0.971831-0.766891.hdf5 \
--maxshift 0 \
--seq-len 1000 \
--we-size 128 \
--model-dim 256 \
--transformer-depth 2 \
--num-heads 8 \
--batch-size 512 \
--ngram 6 \
--stride 1 \
--num-classes 2 \
--pool-size 36 \
--use-conv \
--use-position \
--task test












