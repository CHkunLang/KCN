# -*- coding: utf-8 -*-

import numpy as numpy
import os

instance = []
probability = []

def merge(result_path):
    with open (result_path,'r') as f:
        for line in f:
            line = line.strip().split('\t')
            if line not in instance:
                instance.append(line)
                probability.append('0.5')

def out_result(merge_path):
    out = open (merge_path,'w')
    for i in range(len(instance)):
        ins = '\t'.join(instance[i])
        out.write(ins+'\t'+probability[i]+'\n')
    out.close()

def merge_intra_inter(intra_path,inter_path,save_path):
    if os.path.isdir(save_path):
        pass
    else:
        os.makedirs(save_path)
    merge_path = save_path + 'final_result.txt'
    merge(intra_path)
    merge(inter_path)
    out_result(merge_path)
