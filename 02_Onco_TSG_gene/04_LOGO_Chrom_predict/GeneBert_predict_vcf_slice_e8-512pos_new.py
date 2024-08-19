import joblib
import argparse
import json
import math
import os
import random
import sys
import numpy as np
import pyfasta
import pandas as pd

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score
from tensorflow.keras.layers import Lambda, Dense
from multiprocessing import Pool

sys.path.append("../../")
from bgi.bert4keras.models import build_transformer_model
from bgi.common.callbacks import LRSchedulerPerStep
from bgi.common.refseq_utils import get_word_dict_for_n_gram_number
from bgi.bert4keras.backend import K

if tf.__version__.startswith('1.'):  # tensorflow 1
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
else:  # tensorflow 2
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


"""
功能: Fetches sequences from the genome. 从基因组提取序列
return; 
string: ref sequence, 
string: alt sequence, 
Bool: whether ref allele matches with reference genome
"""
def fetchSeqs(chr, pos, ref, alt, shift=0, inputsize=1000):
    windowsize = inputsize + 100  # 扩大窗，为了检索容纳插入缺失标记
    mutpos = int(windowsize / 2 - 1 - shift)  # mutpos等于249~1849
    seq = genome.sequence({'chr': chr, 'start': pos + shift - int(windowsize / 2 - 1), 'stop': pos + shift + int(windowsize / 2)}) # hg19基因序列从1开始, 序列包含stop位置的基因
    return seq[:mutpos] + ref + seq[(mutpos + len(ref)):], seq[:mutpos] + alt + seq[(mutpos + len(ref)):], seq[mutpos:(mutpos + len(ref))].upper() == ref.upper()  # 原版


"""
功能: Convert sequences to 0-1 encoding and truncate to the input size.The output concatenates the forward and reverse complement sequence encodings. 
           将AGCT序列转为onehot，并补充互补链，用于后续ngram化
Args:
seqs: list of sequences (e.g. produced by fetchSeqs)   [n,1100]
inputsize: the number of basepairs to encode in the output
Returns:
numpy array of dimension: (2 x number of sequence) x 4 x inputsize
2 x number of sequence because of the concatenation of forward and reversecomplement sequences.
"""
def encodeSeqs(seqs, inputsize=1000):
    seqsnp = np.zeros((len(seqs), 4, inputsize), np.bool_) # [slice_size, 4, 1100]
    mydict = {'A': np.asarray([1, 0, 0, 0]), 'T': np.asarray([0, 0, 0, 1]),
                        'G': np.asarray([0, 1, 0, 0]), 'C': np.asarray([0, 0, 1, 0]), 
                        'N': np.asarray([0, 0, 0, 0]), 'H': np.asarray([0, 0, 0, 0]),
                        'a': np.asarray([1, 0, 0, 0]),  't': np.asarray([0, 0, 0, 1]),
                        'g': np.asarray([0, 1, 0, 0]),  'c': np.asarray([0, 0, 1, 0]),
                        'n': np.asarray([0, 0, 0, 0]), '-': np.asarray([0, 0, 0, 0])}
    for n,line in enumerate(seqs):
        cline = line[int(math.floor(((len(line) - inputsize) / 2.0))):int(math.floor(len(line) - (len(line) - inputsize) / 2.0))] # [1000] 取中间，去除window_size多余的
        for i, c in enumerate(cline):
            if c  not in mydict:
                print("error: not in gene dict:  ", c) # 检查是否包含字典中没有的基因
            seqsnp[n, :, i] = mydict[c]

    dataflip = seqsnp[:, ::-1, ::-1]  # 获取互补链 [n, 4, 1000]
    seqsnp = np.concatenate([seqsnp, dataflip], axis=0)  # 补了互补链，变成（2n，4，1000）
    return seqsnp


"""
功能: 将encode后的onthot进行ngram化
args: 
data : (2*slice_size, 4, 1000)
n_gram_value : base on ngram, example: ngram=3, n_gram_value=[100,10,1]
return:
np.array(x_data) : [2*slice_size, 1000] 
"""
def onehot_to_ngram(data=None,n_gram=3,step=1,num_word_dict=None,actg_value=np.array([1, 2, 3, 4]), n_gram_value=None):
    x_data = []
    for index, ii in enumerate(range(data.shape[0])): 
        actg = np.matmul(actg_value, data[ii, :, :]) # [1000]
        gene = []
        for kk in range(0, len(actg), step):
            actg_temp_value = 0
            if kk + n_gram <= len(actg):
                actg_temp_value = np.dot(actg[kk:kk + n_gram], n_gram_value)
                actg_temp_value = int(actg_temp_value)
            else:
                for gg in range(kk, len(actg)):
                    actg_temp_value += actg[gg] * (10 ** (n_gram - gg % n_gram - 1))
                actg_temp_value = actg_temp_value * (10 ** (kk % n_gram))
            gene.append(num_word_dict.get(actg_temp_value, 0)) #[1000] 转换为ngram

        x_data.append(np.array(gene)) # [2*slice_size, 1000] 
        # if index % 10000 == 0 and index > 0:  # 计数
        #     print("Index : {}, Gene len : {}".format(index, len(gene)))
    return np.array(x_data)   # [2*slice_size, 1000] 


"""
功能:  将原始数据裁剪后补反链并转换成ngram格式
args:  seqslist: [slice_size, 1100)]
"""
def preccess_data(seqslist, inputsize, ngram, step, word_dict, actg_value, n_gram_value):
    tem_encoded = encodeSeqs(seqslist, inputsize=inputsize).astype(np.float32)  # tem_encoded: (2*slice_size, 4, 1000),将原始数据onehot后裁剪并补反链
    tem_ngram_input = onehot_to_ngram(data=tem_encoded, n_gram=ngram, step=step,
                                                                                    num_word_dict=word_dict,
                                                                                    actg_value=actg_value,
                                                                                    n_gram_value=n_gram_value)  # step==1： [2*slice_size, 1000]
    print('seqslist length : {},  tem_encoded shape : {}, tem_ngram_input_shape:{}'.format(len(seqslist), tem_encoded.shape, tem_ngram_input.shape), end= " ")
    tem_ngram_input_lenth = tem_ngram_input.shape[0]
    if tem_ngram_input_lenth / 2 == len(seqslist): # 正反链
        pos_ngram_input = tem_ngram_input[:int(tem_ngram_input_lenth / 2), :]  # pos  正链 [slice_size, 1000]
        neg_ngram_input = tem_ngram_input[int(tem_ngram_input_lenth / 2):, :]  # neg  补链 [slice_size, 1000]
        output = (pos_ngram_input, neg_ngram_input) # 数组([slice_size, 1000], [slice_size, 1000]) (正链，反链)
        print("slice process successful")
    else:
        output = tem_ngram_input
        print("warming")
    return output # 数组([slice_size, 1000], [slice_size, 1000]) (正链，反链)


"""
功能: 将输入的ngram数组序列化, 用于后续tf模型预测  来自load_npz_record,给原方法补上parse_function
args:  x_data_:[2n, 1000]
"""
def npz2record(x_data_, batch_size=32, ngram=5, only_one_slice=True, slice_index=None, shuffle=False, seq_len=200, num_classes=919, 
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE,):

    def data_generator():   # 数据生成器
        x_data = x_data_  # 一定要保证是(2n, 1000)，ngram=3，stride=1之后也是1000
        y_data = np.zeros((x_data.shape[0], num_classes), np.bool_).astype(np.int)  # 仿真的标签，预测时不需要
        print("x_data_shape:", x_data.shape)
        # print(y_data.shape)

        x_data_all = []
        y_data_all = []
        for ii in range(ngram):
            if slice_index is not None and ii != slice_index:   # 取符合当前切片的ii
                continue
            if only_one_slice is True:    # 按stride =ngram对序列进行切片  
                slice_indexes = []
                max_slice_seq_len = x_data.shape[1] // ngram * ngram # 1000
                for gg in range(ii, max_slice_seq_len, ngram):   # 从ii开始，stride=ngram
                    slice_indexes.append(gg)  # [200]
                x_data_slice = x_data[:, slice_indexes] # [2n, 200]
                x_data_all.append(x_data_slice)  # [1, 2n, 200]
                y_data_all.append(y_data)
            else:
                x_data_all.append(x_data)  # [ngram, 2n, 200]
                y_data_all.append(y_data)
        x_data_all = np.concatenate(x_data_all) # only_one_slice==true: [2n, 200]  false: [ngram*2n, 200]
        y_data_all = np.concatenate(y_data_all) 
        for x, y in zip(x_data_all, y_data_all):
            yield x, y
        
    def parse_function(x, y):    # dataset map函数
        masked_sequence = x
        segment_id = K.zeros_like(masked_sequence, dtype='int64')
        sequence = y
        y = K.cast(sequence, K.floatx())
        x = { 'Input-Token': masked_sequence,
                 'Input-Segment': segment_id,}
        y = {  'CLS-Activation': y  }
        # print("x: ", masked_sequence)
        # print("y: ", y)
        return x, y

    dataset = tf.data.Dataset.from_generator(data_generator, output_types=(tf.float32, tf.int16),
                                             output_shapes=(tf.TensorShape([seq_len]), tf.TensorShape([num_classes])))
    dataset = dataset.map(map_func=parse_function, num_parallel_calls=num_parallel_calls)
    return dataset


"""
功能: 因为stride=ngram, 因此ngram次预测结果取平均会更加准确
ngram_input : ngram numpy array [2n, 1000]
TEM_BATCH_SIZE : predict batch size, default=1
return : y_pred, numpy array
"""
def predict_avg(ngram_input=None, TEM_BATCH_SIZE=32, ngram=5, seq_len=2000, num_classes=2002):
    y_preds = []
    for ii in range(ngram):
        dataset = npz2record(ngram_input, batch_size=TEM_BATCH_SIZE, ngram=ngram, only_one_slice=True,
                                                    slice_index=ii, shuffle=False, seq_len=seq_len, num_classes=num_classes) # 改处理包含了stride
        dataset = dataset.batch(TEM_BATCH_SIZE)  
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        y_pred = albert.predict(dataset, steps=math.ceil(ngram_input.shape[0] / (TEM_BATCH_SIZE)), verbose=1) #  [2n, 919]
        print("\nPredict epoch:{}, y_pred_shape : {}".format(ii+1, y_pred.shape))
        y_preds.append(y_pred)   # [ngram, 2n, 919]

    all_y_pred = None
    for jj in range(len(y_preds)):
        y_pred = y_preds[jj]  # [2n, 919]
        if all_y_pred is None:
            all_y_pred = y_pred
        else:
            all_y_pred += y_pred # [2n, 919]
    y_pred = all_y_pred / len(y_preds) # [2n, 919]
    y_pred_val = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1]))
    # y_pred_val = y_pred_val[:, 0:num_classes]

    return y_pred_val



# =================================主函数
if __name__ == '__main__':

    _argparser = argparse.ArgumentParser(description='A simple example of the Transformer language model in Genomics',
                                                                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _argparser.add_argument('--inputfile', type=str, metavar='PATH',  default=None, help='Path to vcf file')
    _argparser.add_argument('--outputpath', type=str, metavar='PATH',  default=None, help='Path to output file')
    _argparser.add_argument('--backgroundfile', type=str, metavar='PATH', default=None, help='Path to background file')
    _argparser.add_argument('--maxshift', type=int, default=0, metavar='INTEGER', help='The number of shift seq')
    _argparser.add_argument( '--reffasta', type=str, metavar='PATH', default='/data/male.hg19.fasta', help='Path to a file of reference')
    _argparser.add_argument( '--weight-path', type=str, metavar='PATH', default=None, help='Path to a pretain weight')
    _argparser.add_argument('--epochs', type=int, default=150, metavar='INTEGER', help='The number of epochs to train')
    _argparser.add_argument('--lr', type=float, default=2e-4, metavar='FLOAT', help='Learning rate')
    _argparser.add_argument('--batch-size', type=int, default=32, metavar='INTEGER', help='Training batch size')
    _argparser.add_argument('--seq-len', type=int, default=256, metavar='INTEGER', help='Max sequence length')
    _argparser.add_argument('--we-size', type=int, default=128, metavar='INTEGER', help='Word embedding size')
    _argparser.add_argument('--model', type=str, default='universal', metavar='NAME', choices=['universal', 'vanilla'],
                                                            help='The type of the model to train: "vanilla" or "universal"')
    _argparser.add_argument('--CUDA-VISIBLE-DEVICES', type=int, default=0, metavar='INTEGER', help='CUDA_VISIBLE_DEVICES')
    _argparser.add_argument('--num-classes', type=int, default=10, metavar='INTEGER',help='Number of total classes')
    _argparser.add_argument('--vocab-size', type=int, default=20000, metavar='INTEGER',help='Number of vocab')
    _argparser.add_argument('--slice-size', type=int, default=10000, metavar='INTEGER', help='Slice size')
    _argparser.add_argument('--ngram', type=int, default=6, metavar='INTEGER', help='length of char ngram')
    _argparser.add_argument('--stride', type=int, default=2, metavar='INTEGER', help='stride size')
    _argparser.add_argument('--has-segment', action='store_true',help='Include segment ID')
    _argparser.add_argument('--num-heads', type=int, default=4, metavar='INTEGER', help='Heads of self attention')
    _argparser.add_argument('--model-dim', type=int, default=128, metavar='INTEGER', help='Heads of self attention')
    _argparser.add_argument('--transformer-depth', type=int, default=2, metavar='INTEGER', help='Heads of self attention')
    _argparser.add_argument('--num-gpu', type=int, default=1, metavar='INTEGER', help='Number of GPUs')
    _argparser.add_argument('--task', type=str, default='train', metavar='NAME',choices=['train', 'valid', 'test'],
                                                            help='The type of the task')
    _argparser.add_argument('--verbose', type=int, default=2, metavar='INTEGER', help='Verbose')
    _argparser.add_argument('--steps-per-epoch', type=int, default=10000, metavar='INTEGER', help='steps per epoch')
    _argparser.add_argument('--shuffle-size', type=int, default=1000, metavar='INTEGER', help='Buffer shuffle size')
    _argparser.add_argument('--num-parallel-calls', type=int, default=16, metavar='INTEGER', help='Num parallel calls')
    _argparser.add_argument('--prefetch-buffer-size', type=int, default=4, metavar='INTEGER',help='Prefetch buffer size')
    _argparser.add_argument('--pool-size', type=int, default=16, metavar='INTEGER',help='Pool size of multi-thread')
    
    _argparser.add_argument('--use-position', action='store_true', help='Using position ids')  # 触发则为True 否则为False
    _argparser.add_argument('--use-segment', action='store_true', help='Using segment ids')
    _argparser.add_argument('--use-conv', action='store_true', help='Using Conv1D layer')
    # _argparser.add_argument('--save', type=str, required=True, metavar='PATH', help='A path where the best model should be saved / restored from')
    # _argparser.add_argument('--slice', type=list, default=[6], help='Slice')
    # _argparser.add_argument('--word-prediction', action='store_true', help='Word prediction')
    # _argparser.add_argument('--class-prediction', action='store_true', help='class prediction')    
    _args = _argparser.parse_args()

    batch_size = _args.batch_size
    epochs = _args.epochs
    num_gpu = _args.num_gpu

    max_seq_len = _args.seq_len
    initial_epoch = 0

    ngram = _args.ngram
    stride = _args.stride

    word_from_index = 10
    word_dict = get_word_dict_for_n_gram_number(n_gram=ngram, word_index_from=word_from_index)

    num_classes = _args.num_classes
    only_one_slice = True
    # vocab_size = len(word_dict) + word_from_index
    vocab_size = len(word_dict) + word_from_index + 3

    slice_size = _args.slice_size
    pool_size = _args.pool_size

    max_depth = _args.transformer_depth
    model_dim = _args.model_dim
    embedding_size = _args.we_size
    num_heads = _args.num_heads

    use_position = _args.use_position
    use_segment = _args.use_segment
    use_conv = _args.use_conv

    shuffle_size = _args.shuffle_size
    num_parallel_calls = _args.num_parallel_calls
    prefetch_buffer_size = _args.prefetch_buffer_size

    word_seq_len = max_seq_len // ngram * int(ngram / ngram) # n-mer后长度
    print("max_seq_len: ", max_seq_len, " word_seq_len: ", word_seq_len)

    pretrain_weight_path = _args.weight_path
    steps_per_epoch = _args.steps_per_epoch

    inputfile = _args.inputfile
    outputpath = _args.outputpath
    maxshift = _args.maxshift  # maxshift = 0
    inputsize = _args.seq_len  # inputsize = 1000

    genome = pyfasta.Fasta(_args.reffasta)
    CHRS = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9',
                    'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
                    'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY']

    # save_path = _args.save
    # model_name = _args.model_name
    # config_save_path = os.path.join(save_path, "{}_config.json".format(model_name))


    # =================================导入数据 .vcf文件
    # inputfile = "data.vcf"
    vcf = pd.read_csv(inputfile, sep='\t', header=None, comment='#')  
    vcf.columns = ['chr', 'pos', 'Label', 'ref', 'alt'] + list(vcf.columns[5:])  # 修改了对vcf的修改，支持info的输出
    vcf.iloc[:, 0] = 'chr' + vcf.iloc[:, 0].map(str).str.replace('chr', '')    # 数据染色质名称标准化, 把vcf文件中，染色体名字为数字的，改成chr+数字的格式
    vcf = vcf[vcf.iloc[:, 0].isin(CHRS)]   # 判断输入的VCF文件中的突变所在染色体是否规范
    vcf.pos = vcf.pos.astype(int)
    print('VCF file shape is : ', vcf.shape)

    # ================================= 导入模型并行预测
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    if strategy.num_replicas_in_sync >= 1:
        num_gpu = strategy.num_replicas_in_sync
    with strategy.scope():  
        config = {"attention_probs_dropout_prob": 0,
                            "hidden_act": "gelu",
                            "hidden_dropout_prob": 0,
                            "embedding_size": embedding_size,
                            "hidden_size": model_dim,
                            "initializer_range": 0.02,
                            "intermediate_size": model_dim * 4,
                            "max_position_embeddings": 512,     # 必须修改为512
                            "num_attention_heads": num_heads,
                            "num_hidden_layers": max_depth,
                            "num_hidden_groups": 1,
                            "net_structure_type": 0,
                            "gap_size": 0,
                            "num_memory_blocks": 0,
                            "inner_group_num": 1,
                            "down_scale_factor": 1,
                            "type_vocab_size": 0,
                            "vocab_size": vocab_size,
                            "custom_masked_sequence": False, 
                            "use_position_ids": use_position,
                            "custom_conv_layer": use_conv,
                            "use_segment_ids": use_segment}     # 模型配置

        bert = build_transformer_model( configs=config,
                                                                            # checkpoint_path=checkpoint_path,
                                                                            model='bert',
                                                                            return_keras_model=False, ) # 导入模型
        output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output) # 取特征
        # output = Dense(name='CLS-Activation', units=num_classes, activation='sigmoid', kernel_initializer=bert.initializer)(output)   # 屏蔽则为取特征
        albert = tf.keras.models.Model(bert.model.input, output)
        albert.compile(loss=[tf.keras.losses.BinaryCrossentropy()],
                                        optimizer=tf.keras.optimizers.Adam(0.0001), 
                                        metrics=['accuracy', tf.keras.metrics.AUC()])
        albert.summary()
        
        if pretrain_weight_path is not None and len(pretrain_weight_path) > 0: # 加载参数
            albert.load_weights(pretrain_weight_path, by_name=True) # 自动匹配层名，若找不到则跳过
            print("Load weights: ", pretrain_weight_path)
            # albert = tf.keras.models.Model(inputs=bert.model.input, outputs = albert.get_layer('CLS-token').output)
   
    # =================================导入数据
    actg_value = np.array([1, 2, 3, 4])
    print("actg_value", actg_value)
    n_gram_value = np.ones(ngram)
    for ii in range(ngram):
        n_gram_value[ii] = int(n_gram_value[ii] * (10 ** (ngram - ii - 1)))
    print("n_gram_value: ", n_gram_value)

    # 提取参考基因序列以及等位基因序列
    for shift in [0, ] + list(range(-200, -maxshift - 1, -200)) + list(range(200, maxshift + 1, 200)): # 当maxshift==0时： [0]
        refseqs = []
        altseqs = []
        ref_matched_bools = []
        print("shift is ", shift)
        print("__Fetching Seqs...__")
        for i in range(vcf.shape[0]):
            refseq, altseq, ref_matched_bool = fetchSeqs(vcf.iloc[i, 0], vcf.iloc[i, 1], vcf.iloc[i, 3], vcf.iloc[i, 4],
                                                                                                         shift=shift, inputsize=inputsize) # [n, 1100]
            refseqs.append(refseq)  # vcf信息对应的refseq，长度为1100
            altseqs.append(altseq)  # vcf信息对应的altseq，长度为1100
            ref_matched_bools.append(ref_matched_bool)  # ref等位基因是否与参考基因组匹配上
        print("catch refseq length:", len(refseq))
        print("catch altseq length:", len(altseq))

        if shift == 0:
            # only need to be checked once
            print("Number of variants with reference allele matched with reference genome:", np.sum(ref_matched_bools))
            print("Number of input variants:", len(ref_matched_bools))
            print("Computing 1st file..")
        
        # =================================数据预处理
        print('__Processing REF Seqs and ALT Seqs__')
        pool = Pool(processes=pool_size)# 使用多进程并行处理
        data_all_list = []  
        for tem_seqs in [refseqs, altseqs]:  
            #  按slice_size分批处理
            num_row = len(tem_seqs)
            results = []
            for ii in range(math.ceil(num_row / slice_size)):  
                # print('{}:{}'.format(slice_size*ii, slice_size*(ii+1)))
                slice_seqslist = tem_seqs[slice_size * ii: slice_size * (ii + 1)]  # [slice_size, 1100] 溢出也没关系
                # print(len(slice_seqslist))
                result = pool.apply_async(preccess_data,args=(slice_seqslist,
                                                                                                                    inputsize,
                                                                                                                    ngram,
                                                                                                                    stride,
                                                                                                                    word_dict,
                                                                                                                    actg_value,
                                                                                                                    n_gram_value  ))  # 数组([slice_size, 1000], [slice_size, 1000]) (正链，反链)
                results.append(result) # [([slice_size, 1000], [slice_size, 1000]) , ...]
            # 汇总结果
            pos_data_all = []
            neg_data_all = []
            for result in results:   # result:数组([slice_size, 1000], [slice_size, 1000]) (正链，反链)
                pos_data, neg_data = result.get()# pos_data: [slice_size,1000]     neg_data: [slice_size,1000]
                if len(pos_data) > 0 and len(neg_data) > 0 and len(pos_data) == len(neg_data):
                    pos_data_all.extend(pos_data)  # [n,1000]
                    neg_data_all.extend(neg_data)  # [n,1000]
            pos_data_all = np.array(pos_data_all)
            neg_data_all = np.array(neg_data_all)
            data_all = np.vstack([pos_data_all, neg_data_all])  # [2n, 1000]
            print("data_all: ", data_all.shape)
            data_all_list.append(data_all) # [2, 2n, 1000]: [参考与等位，正链与补链，链长]
            print("come to next part")
        pool.close()
        pool.join()

        # =================================开始预测
        print('__Predicting__') 
        y_pred_ref_ori = predict_avg(ngram_input=data_all_list[0],
                                                                    TEM_BATCH_SIZE=int(batch_size),
                                                                    ngram=ngram,
                                                                    seq_len=word_seq_len,
                                                                    num_classes=num_classes) # [2n, 919]
        y_pred_ref = np.where(y_pred_ref_ori > 0.0000001, y_pred_ref_ori, 0.0000001) # 当条件成立时where方法返回x，当条件不成立时where返回y
        print("y_pred_ref shape", y_pred_ref.shape) # [14376, 919]
        y_pred_alt_ori = predict_avg(ngram_input=data_all_list[1],
                                                                    TEM_BATCH_SIZE=int(batch_size),
                                                                    ngram=ngram,
                                                                    seq_len=word_seq_len,
                                                                    num_classes=num_classes)# [2n, 919]
        y_pred_alt = np.where(y_pred_alt_ori > 0.0000001, y_pred_alt_ori, 0.0000001) # 代替0值
        print("y_pred_alt shape", y_pred_alt.shape)# [14376, 919]

        # deepsea版本.
        # ref_encoded = encodeSeqs(refseqs, inputsize=inputsize).astype(np.float32)    #shape是(20, 4, 2000),因为包含了正和反两条链
        # print('ref_encoded shape', ref_encoded.shape)
        # # 输入序列，得到转换后可被tf-deepsea接受的格式，2100bp变成中间的2000bp变成片段变成词index
        # ref_ngram_input = onehot_to_ngram(data=ref_encoded, n_gram = ngram, step = stride,
        # num_word_dict = word_dict,
        # actg_value = actg_value,
        # n_gram_value = n_gram_value)
        # alt_encoded = encodeSeqs(altseqs, inputsize=inputsize).astype(np.float32)    #shape是(20, 4, 2000),因为包含了正和反两条链
        # print('alt_encoded shape', alt_encoded.shape)
        # alt_ngram_input = onehot_to_ngram(data=alt_encoded, n_gram = ngram, step = stride,
        # num_word_dict = word_dict,
        # actg_value = actg_value,
        # n_gram_value = n_gram_value)

        # print('__Predicting__')
        # y_pred_ref_ori = predict_avg(ngram_input=ref_ngram_input,
        # TEM_BATCH_SIZE=GLOBAL_BATCH_SIZE,
        # ngram=ngram,
        # seq_len=word_seq_len,
        # num_classes=num_classes)
        # y_pred_ref = np.where(y_pred_ref_ori>0.0000001, y_pred_ref_ori, 0.0000001)
        # print("y_pred_ref shape",y_pred_ref.shape)
        # y_pred_alt_ori = predict_avg(ngram_input=alt_ngram_input,
        # TEM_BATCH_SIZE=GLOBAL_BATCH_SIZE,
        # ngram=ngram,
        # seq_len=word_seq_len,
        # num_classes=num_classes)
        # y_pred_alt = np.where(y_pred_alt_ori>0.0000001, y_pred_alt_ori, 0.0000001)
        # print("y_pred_alt shape",y_pred_alt.shape)

        # =================================保存结果
        print("Writing to csv")
        # header = np.loadtxt('/alldata/LChuang_data/myP/DeepSEA/DeepSEA-v0.94/resources/predictor.names',dtype=np.str)
        wfile1 = "{}data_{}bs_{}gram_{}feature.out.ref.csv".format(outputpath, batch_size, ngram, num_classes)
        wfile2 = "{}data_{}bs_{}gram_{}feature.out.alt.csv".format(outputpath, batch_size, ngram, num_classes)
        wfile3 = "{}data_{}bs_{}gram_{}feature.out.logfoldchange.csv".format(outputpath, batch_size, ngram, num_classes)
        wfile4 = "{}data_{}bs_{}gram_{}feature.out.diff.csv".format(outputpath, batch_size, ngram, num_classes)
        wfile6 = "{}data_{}bs_{}gram_{}feature.out.evalue.csv".format(outputpath, batch_size, ngram, num_classes)
        wfile7 = wfile6.replace('evalue.csv', 'evalue_gmean.csv')
        wfile8 = wfile6.replace('evalue.csv', 'funsig.csv')
    
        # 将ref与alt相减，data中分别包含了logfolddiff相对差异数组与diff相对差异数组（取256隐藏特征时np.log2可能产生空值,原因未明）
        diff_data = np.hstack([np.log2(y_pred_alt / (1 - y_pred_alt + 1e-12)) - np.log2(y_pred_ref / (1 - y_pred_ref + 1e-12)), y_pred_alt - y_pred_ref])  #  [14376, 1838]
        diff_data = diff_data[:int((diff_data.shape[0] / 2)), :] / 2.0 + diff_data[int((diff_data.shape[0] / 2)):, :] / 2.0  # 正负链相加各自取平均
        print("logfoldchange and diff array shape :", diff_data.shape) # [7188, 1838]
        # ref与alt结果
        y_pred_ref = y_pred_ref[:int((y_pred_ref.shape[0] / 2)), :] / 2.0 + y_pred_ref[int((y_pred_ref.shape[0] / 2)):, :] / 2.0  # 正链结果/2  + 负链结果/2
        y_pred_alt = y_pred_alt[:int((y_pred_alt.shape[0] / 2)), :] / 2.0 + y_pred_alt[int((y_pred_alt.shape[0] / 2)):, :] / 2.0
        print("y_pred_ref array shape :", y_pred_ref.shape) # [7188, 919]
        print("y_pred_alt array shape :", y_pred_alt.shape) # [7188, 919]        

        header = list(range(y_pred_ref.shape[-1]))

        # ref写入
        temp = pd.DataFrame(y_pred_ref)     
        temp.columns = header
        if vcf.shape[0] == temp.shape[0]:
            temp = pd.concat([vcf, temp], axis=1)
            temp.to_csv(wfile1, sep='\t', float_format='%.8f', header=True, index=False)
            print("Saved ", wfile1)
        else:
            print("vcf.shape[0] is not equal to temp.shape[0]")
        
        # alt写入
        temp = pd.DataFrame(y_pred_alt) 
        temp.columns = header
        if vcf.shape[0] == temp.shape[0]:
            temp = pd.concat([vcf, temp], axis=1)
            temp.to_csv(wfile2, sep='\t', float_format='%.8f', header=True, index=False)
            print("Saved ", wfile2)
        else:
            print("vcf.shape[0] is not equal to temp.shape[0]")

        # logfoldchange写入
        temp = pd.DataFrame(diff_data[:, :y_pred_ref.shape[-1]]) # [7188, 919]  前num_classes列
        temp.columns = header
        if vcf.shape[0] == temp.shape[0]:
            temp = pd.concat([vcf, temp], axis=1)
            temp.to_csv(wfile3, sep='\t', float_format='%.8f', header=True, index=False)
            print("Saved ", wfile3)
        else:
            print("vcf.shape[0] is not equal to temp.shape[0]")

        # diff写入
        temp = pd.DataFrame(diff_data[:, y_pred_ref.shape[-1]:]) # [7188, 919]  后num_classes列
        temp.columns = header
        if vcf.shape[0] == temp.shape[0]:
            temp = pd.concat([vcf, temp], axis=1)
            temp.to_csv(wfile4, sep='\t', float_format='%.8f', header=True, index=False)
            print("Saved ", wfile4)
        else:
            print("vcf.shape[0] is not equal to temp.shape[0]")


        # #compute E-values for chromatin effects（版本1）
        # vcf_row = vcf.shape[0]
        # backgroundfile = _args.backgroundfile
        # ecdfs=joblib.load(backgroundfile)
        # datae=np.ones((data.shape[0],num_classes))
        # for i in range(num_classes):
        # datae[:,i]=1-ecdfs[i](np.abs(data[:,i+num_classes]*data[:,i]))
        # #将0值替换
        # datae[datae==0]=1e-6

        # compute E-values for chromatin effects（版本2）
        # print("compute E-values for chromatin effects V2")
        # datae = np.ones((data.shape[0], num_classes))
        # # json_pkl_path = '/alldata/Nzhang_data/project/T2D/2.background/1.2002mark_5gram/'
        # json_pkl_path = _args.backgroundfile
        # tem_file = os.listdir(json_pkl_path)
        # pkl_filelist = []
        # for item in tem_file:
        #     if item.endswith('.json'):
        #         json_file_name = os.path.join(json_pkl_path, item)
        #         print(json_file_name)
        #         pkl_dict = json.load(open(json_file_name))  # load dict
        #     elif item.endswith('.pkl'):
        #         pkl_filelist.append(item)
        # print("pkl_filelist length is :", len(pkl_filelist))
        # # 对每一列计算evalue
        # for i in range(num_classes):
        #     tem_background_pkl = pkl_dict[str(i)]  # get pkl file
        #     tem_background_pkl = os.path.join(json_pkl_path, tem_background_pkl)
        #     ecdfs = joblib.load(tem_background_pkl)
        #     datae[:, i] = 1 - ecdfs(np.abs(data[:, i + num_classes] * data[:, i]))
        #     if i % 100 == 0:
        #         print("Pkl finnished :", i)
        #         # print("Finished:", tem_background_pkl)
        # # 将0值替换
        # datae[datae == 0] = 1e-6
        # print("Finished all, writing E-value output file...")

        # # write E-values for chromatin effects
        # temp_evalue = pd.DataFrame(datae[:, :num_classes])
        # temp_evalue.columns = header
        # if vcf_row == temp_evalue.shape[0]:
        #     temp = pd.concat([vcf, temp_evalue], axis=1)
        #     temp.to_csv(wfile6, float_format='%.8f', header=True, index=False)
        #     print("Saving ", wfile6)
        # else:
        #     print("vcf.shape[0] is not equal to temp.shape[0]")

        # del temp

        # # write gmean of E-values for chromatin effects
        # print("Writing gmean of E-value output file...")
        # import scipy
        # from scipy import stats

        # gmean_row_value = scipy.stats.gmean(temp_evalue, axis=1)
        # gmean_df = pd.DataFrame(list(gmean_row_value))
        # gmean_df.columns = ['gmean']
        # print(gmean_df.shape)

        # new_df = pd.concat([vcf, gmean_df], axis=1)
        # new_df.to_csv(wfile7, sep=',', index=None)
        # print(new_df.shape)
        # print("Saving ", wfile7)


        # write compute E-values for Functional Significance scores
        #print("Writing gmean of Functional-Significance output file...")
        #datadeepsea = np.exp(np.mean(np.log(datae), axis=1))
        #datadeepsea_df = pd.DataFrame(list(datadeepsea))
        #datadeepsea_df.columns = ['Functional significance score']
        #print(datadeepsea_df.shape)

        #new_df = pd.concat([vcf, datadeepsea_df], axis=1)
        #new_df.to_csv(wfile8, sep=',', index=None)
        #print(new_df.shape)
        #print("Saving ", wfile8)

        # 立即跳出程序
        os._exit(0)



"""
# 
"""

'''
# 并行版本(2002)
cd /data/BGI-Gene_new/examples/vcf-predict
CUDA_VISIBLE_DEVICES=2,3 python GeneBert_predict_vcf_slice.py \
--inputfile /data/BGI-Gene_new/examples/vcf-predict/1million_background_SNPs_1000G_converted.vcf \
--reffasta /data/male.hg19.fasta \
--maxshift 0 \
--weight-path /data/BGI-Gene_new/data_exp_mat/genebert_3_gram_2_layer_8_heads_256_dim_expecto_[mat]_weights_67-0.9601-0.9674.hdf5 \
--seq-len 2000 \
--model-dim 256 \
--transformer-depth 2 \
--num-heads 8 \
--batch-size 256 \
--num-classes 2002 \
--shuffle-size 4000 \
--pool-size 64 \
--slice-size 10000 \
--ngram 3 \
--stride 1
# 并行版本(3357)
cd /data/BGI-Gene_new/examples/vcf-predict
CUDA_VISIBLE_DEVICES=4,5,6,7 python GeneBert_predict_vcf_slice.py \
--inputfile /data/BGI-Gene_new/examples/vcf-predict/1million_background_SNPs_1000G_converted.vcf \
--reffasta /data/male.hg19.fasta \
--maxshift 0 \
--weight-path /data/BGI-Gene_new/data_wanrenexp_3357_selene/genebert_3_gram_2_layer_8_heads_256_dim_wanrenexp_[baseonEpoch10]_weights_106-0.9746-0.9768.hdf5 \
--seq-len 2000 \
--model-dim 256 \
--transformer-depth 2 \
--num-heads 8 \
--batch-size 256 \
--num-classes 3357 \
--shuffle-size 4000 \
--pool-size 64 \
--slice-size 10000 \
--ngram 3 \
--stride 1

'''

'''
# 非并行版本

# cd /data/BGI-Gene_new/examples/vcf-predict
# CUDA_VISIBLE_DEVICES=0 python GeneBert_predict_vcf.py \
# --inputfile /data/BGI-Gene_new/examples/vcf-predict/1million_background_SNPs_1000G_converted.vcf \
# --reffasta /data/male.hg19.fasta \
# --maxshift 0 \
# --weight-path /data/BGI-Gene_new/data_exp_mat/genebert_3_gram_2_layer_8_heads_256_dim_expecto_[mat]_weights_109-0.9613-0.9678.hdf5 \
# --seq-len 2000 \
# --model-dim 256 \
# --transformer-depth 2 \
# --num-heads 8 \
# --batch-size 256 \
# --num-classes 2002 \
# --shuffle-size 4000 \
# --ngram 3 \
# --stride 1

# wanrenexp3357 5gram ori
# cd /data/BGI-Gene_new/examples/vcf-predict
# CUDA_VISIBLE_DEVICES=1 python GeneBert_predict_vcf.py \
# --inputfile /data/BGI-Gene_new/examples/vcf-predict/1million_background_SNPs_1000G_converted.vcf \
# --reffasta /data/male.hg19.fasta \
# --maxshift 0 \
# --weight-path /data/BGI-Gene_new/data_wanrenexp_3357_selene/genebert_3_gram_2_layer_8_heads_256_dim_wanrenexp_[baseonEpoch10]_weights_108-0.9746-0.9764.hdf5 \
# --seq-len 2000 \
# --model-dim 256 \
# --transformer-depth 2 \
# --num-heads 8 \
# --batch-size 256 \
# --num-classes 3357 \
# --shuffle-size 4000 \
# --ngram 3 \
# --stride 1

# --weight-path /data/BGI-Gene_new/data_wanrenexp_3357_selene/genebert_3_gram_2_layer_8_heads_256_dim_wanrenexp_[baseonEpoch10]_weights_108-0.9746-0.9764.hdf5
# --weight-path /data/BGI-Gene_new/data_exp_mat/genebert_3_gram_2_layer_8_heads_256_dim_expecto_[mat]_weights_109-0.9613-0.9678.hdf5
'''
