import pandas as pd
from sklearn.preprocessing import LabelEncoder

CGC_file = "./data/Cosmic/Cancer Gene Census.tsv"
CGC_tsv = pd.read_csv(CGC_file, sep = '\t')

location_str = "Genome Location"
annotation = "Role in Cancer"

chr_str = "chr"
start_pos_str = "start_pos"
end_pos_str = "end_pos"
annotation_onco_str = "annotation_onco"
annotation_TSG_str = "annotation_TSG"
annotation_str= "annotation"

# CGC_tsv = CGC_tsv[CGC_tsv.Somatic == "yes" ] # 体细胞
chr = []
start_pos = []
end_pos = []
annotation_onco = []
annotation_TSG = []
annotation_both = []
for  Pos, anno in zip(CGC_tsv[location_str],  CGC_tsv[annotation]): 
    chr.append(Pos.split(":")[0])
    start_pos.append(Pos.split(":")[-1].split("-")[0])
    end_pos.append(Pos.split(":")[-1].split("-")[-1])

    if  type(anno) == float:
        annotation_onco.append("nan")
        annotation_TSG.append("nan")
        annotation_both.append("nan_nan")
    else:
        if "oncogene" in anno:
            annotation_onco.append("oncogene")
            anno_both = "oncogene_"
        else:
            annotation_onco.append("nan")
            anno_both = "nan_"

        if "TSG" in anno:
            annotation_TSG.append("TSG")
            anno_both = anno_both+"TSG"
        else:
            annotation_TSG.append("nan")
            anno_both = anno_both+"nan"
        
        annotation_both.append(anno_both)

encoder = LabelEncoder()
annotation_both_encode= encoder.fit_transform(annotation_both)      
data = pd.DataFrame({chr_str:chr,
                                              start_pos_str:start_pos,
                                              end_pos_str:end_pos,
                                              annotation_onco_str:annotation_onco, 
                                              annotation_TSG_str:annotation_TSG,
                                              annotation_str:annotation_both_encode})
data.to_csv("./data/Cosmic/data.csv", sep='\t', index=False)





import numpy as np
import os
import sys
import gzip
import re
from pyfaidx import Fasta
from sklearn.model_selection import train_test_split

fasta = './data/GCF_000001405.25_GRCh37.p13_genomic.fna'
genome = Fasta(fasta)

# 染色体名称字典
chr_dict = {"NC_000001.10": 1,
            "NC_000002.11": 2,
            "NC_000003.11": 3,
            "NC_000004.11": 4,
            "NC_000005.9": 5,
            "NC_000006.11": 6,
            "NC_000007.13": 7,
            "NC_000008.10": 8,
            "NC_000009.11": 9,
            "NC_000010.10": 10,
            "NC_000011.9": 11,
            "NC_000012.11": 12,
            "NC_000013.10": 13,
            "NC_000014.8": 14,
            "NC_000015.9": 15,
            "NC_000016.9": 16,
            "NC_000017.10": 17,
            "NC_000018.9": 18,
            "NC_000019.9": 19,
            "NC_000020.10": 20,
            "NC_000021.8": 21,
            "NC_000022.10": 22,
            "NC_000023.10": 'X',
            "NC_000024.9": 'Y'}
rev_chr_dict = {}
for k, v in chr_dict.items():
    rev_chr_dict[str(v)] = k

set_atcg = set(list('NATCG'))
atcg_dict = {'N': 0, 'A': 1, 'G': 2, 'C': 3, 'T': 4}

# 产生序列与标签
seq_len = 1000 # 长度为1000bp
interval = 200    # 以200bp进行分割
features = []
labels = []
for index, row in data.iterrows(): 
    print("index:", index)
    chr = row[chr_str]
    strat_pos = row[start_pos_str]
    end_pos = row[end_pos_str]
    annotation_onco = row[annotation_onco_str]
    annotation_TSG = row[annotation_TSG_str]
    
    # 检查是否有误
    if chr=="" or strat_pos=="" or end_pos =="" or annotation_onco==""  or  annotation_onco=="":
        continue
    # 规范化
    chr = str(chr)
    strat_pos = int(strat_pos)
    end_pos = int(end_pos)
    annotation_onco = str(annotation_onco)
    annotation_TSG = str(annotation_TSG)
    
    # 标签
    if annotation_onco == "oncogene":
        annotation_1 = 1
    elif annotation_onco == "nan":
        annotation_1 = 0
    else:
        print("label error")

    if annotation_TSG == "TSG":
        annotation_2 = 1
    elif annotation_TSG == "nan":
        annotation_2 = 0
    else:
        print("label error")
    label= [annotation_1, annotation_2]

    # 序列
    ref_chr = rev_chr_dict.get(chr, '')  # 将染色体1, 2, ...X 转化为NC_000001.10, ... 
    if len(ref_chr) == 0:  # 出现错误的染色体
        print(chr)

    strat_pos_slice_list = []
    end_pos_slice_list = []
    for  i in range(strat_pos, end_pos, interval): # 200bp分割位置
        if (i+ seq_len)<= end_pos:
            strat_pos_slice_list.append(i)
            end_pos_slice_list.append(i+ seq_len)  # seq=1000bp

    for strat_pos_slice, end_pos_slice in zip(strat_pos_slice_list, end_pos_slice_list):
        # 产生序列
        ref_start = strat_pos_slice
        ref_end = end_pos_slice
        seq = str(genome[ref_chr][(ref_start-1):(ref_end-1)]) # vcf文件中pos从1开始, 所以减1   
        
        # 检查序列中是否存在不是 'NATCG'的字符
        is_atcg = True
        for atcg in set(list(seq)):
            if atcg.upper() not in set_atcg:
                is_atcg = False
        if is_atcg is False:
            print("Not ATCG\n",seq)
            continue       
        
        # AGCT is converted to 1, 2, 3, 4 
        seq_num = np.array([atcg_dict[ACTG.upper() ]for ACTG in seq]) #  (1000)  

        features.append(seq_num)
        labels.append(label)

features= np.array(features)  # (n, 1000)  AGCT is converted to 1, 2, 3, 4 
labels= np.array(labels)    # (n, 2)

# 分离训练集与验证集
indices = range(labels.shape[0])
feature_train, feature_valid_test, label_train,  label_valid_test, indices_train,  indices_valid_test= train_test_split(features, labels, indices, test_size=0.2,random_state=0, shuffle = True)
feature_valid, feature_test, label_valid,  label_test, indices_valid,  indices_test= train_test_split(feature_valid_test, label_valid_test, indices_valid_test, test_size=0.5,random_state=0, shuffle = True)

print(indices_test)
np.save("./data/Cosmic/train.npy", {"feature": feature_train, "label": label_train})  # (330471, 1000)  (330471, 2)
np.save("./data/Cosmic/valid.npy", {"feature": feature_valid, "label": label_valid})  # (41309, 1000)  (41309, 2)
np.save("./data/Cosmic/test.npy", {"feature": feature_test, "label": label_test})         # (41309, 1000)  (41309, 2)
# np.save("./data/Cosmic/data.npy", {"feature": features, "label": labels})
# np.load("./data/Cosmic/data.npy", allow_pickle=True).item()