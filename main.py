# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from merge_result import merge_intra_inter
import torch.nn.functional as F

#torch.manual_seed(1337)
#torch.cuda.manual_seed(1337)
#mySeed = np.random.RandomState(1234)

def get_property(max_len,wordEmbed,entityEmbed,relationEmbed,sentence,ent_feat,ctd):
    sent_id = ent_feat[0]
    e1_id = ent_feat[1]
    e1_start_pos = ent_feat[2]
    e1_end_pos   = ent_feat[3]+1
    e2_id = ent_feat[4]
    e2_start_pos = ent_feat[5]
    e2_end_pos   = ent_feat[6]+1
    len_e1 = e1_end_pos - e1_start_pos
    len_e2 = e2_end_pos - e2_start_pos
    ######################generate the sequence##################################
    e1=sentence[e1_start_pos:e1_end_pos]
    e2=sentence[e2_start_pos:e2_end_pos]
    pos = sorted([(e1_start_pos,-1),(e1_end_pos,-1),(e2_start_pos,-2),(e2_end_pos,-2)],key=lambda p : p[0])
    sentence = sentence[:pos[0][0]]+[pos[0][1]]+sentence[pos[1][0]:pos[2][0]]+[pos[2][1]]+sentence[pos[3][0]:]
    n        = len(sentence)

    e1vectors = torch.cat([wordEmbed(Variable(torch.LongTensor([int(e)]).cuda())).view(100,1) for e in e1],1)
    e2vectors = torch.cat([wordEmbed(Variable(torch.LongTensor([int(e)]).cuda())).view(100,1) for e in e2],1)

    e1vector      = torch.sum(e1vectors,1)/len(e1)
    e2vector      = torch.sum(e2vectors,1)/len(e2)
    wordVectorLength = len(e2vector)
    #wordsembeddings:
    
    if sentence[0] == -1:
        words = e1vector.view(wordVectorLength,1)
    elif sentence[0] == -2:
        words = e2vector.view(wordVectorLength,1)
    else:
        words = wordEmbed(Variable(torch.LongTensor([sentence[0]]).cuda())).view(wordVectorLength,1)
    for word in sentence[1:]:
        if word == -1:
            words = torch.cat([words, e1vector.view(wordVectorLength,1)],1)
        elif word == -2:
            words = torch.cat([words, e2vector.view(wordVectorLength,1)],1)
        else:
            words = torch.cat([words, wordEmbed(Variable(torch.LongTensor([word]).cuda())).view(wordVectorLength,1)],1)
    wordswithPos = words
    ##########################generate the kb##############################################
    E1 = entityEmbed(Variable(torch.LongTensor([ctd[0]]).cuda()))
    E2 = entityEmbed(Variable(torch.LongTensor([ctd[1]]).cuda()))
    relation = relationEmbed(Variable(torch.LongTensor([ctd[2]]).cuda()))
    return sent_id, wordswithPos, e1_id,e2_id,E1,E2,e1vectors,e2vectors,len(e1),len(e2),relation, n


def train_and_test(data_path,ctd_path,save_path,is_intra,func):
    #####################load the data###########################
    
    train_Sentence,_,train_Label,train_Entities,\
    dev_Sentence,_,dev_Label,dev_Entities,\
    test_Sentence,_,test_Label,test_Entities,\
    word2vector0 = pickle.load(open(data_path,'rb'),encoding = 'iso-8859-1') #the second variable is useless in this model
    ctd_train,_,ctd_dev,_, ctd_test,_, entity2vector0, relation2vector0 = pickle.load(open(ctd_path, 'rb'),encoding = 'iso-8859-1') #the second variable is useless in this model
    #####################get he feature###########################sentence length
    max_len = max(max([len(sentence) for sentence in train_Sentence]),\
                  max([len(sentence) for sentence in dev_Sentence]),\
                  max([len(sentence) for sentence in test_Sentence]))
    max_features = len(word2vector0)  #the number of the words
    embedding_dim = len(word2vector0[0])
    embedding_dim_kg = len(entity2vector0[0])
    print ("max sentence length: {}".format(max_len))
    print ("unique word number: {}".format(max_features))
    print ("word embedding dim: {}".format(embedding_dim))
    print ("knowledge embedding dim: {}".format(embedding_dim_kg))
    #################prepare the embedding layer#################
    pubid = []
    with open ('./data_clean/PubID.txt') as f:
        for line in f:
            pubid.append(line.strip())
    id_num = len(pubid)
    mySeed.shuffle(pubid)
    train_Label0 = train_Label+dev_Label

    word2vector = nn.Parameter(torch.FloatTensor(word2vector0).cuda())
    wordEmbed = nn.Embedding(max_features,embedding_dim)
    wordEmbed.weight = word2vector

    entitynum = len(entity2vector0)
    entity2vector = nn.Parameter(torch.FloatTensor(entity2vector0).cuda())
    entityEmbed = nn.Embedding(entitynum,embedding_dim_kg)
    entityEmbed.weight = entity2vector

    relationnum = len(relation2vector0)
    relation2vector = nn.Parameter(torch.FloatTensor(relation2vector0).cuda())
    relationEmbed = nn.Embedding(relationnum,embedding_dim_kg)
    relationEmbed.weight = relation2vector

    ######################prepare the data#######################
    print ('generate the train data...')
    train_set = []
    for i in range(len(train_Sentence)):
        sampleTuple = get_property(max_len,wordEmbed,entityEmbed,relationEmbed,train_Sentence[i],train_Entities[i],ctd_train[i])
        train_set.append(sampleTuple)
    print ('generate the development data...')
    dev_set = []
    for i in range(len(dev_Sentence)):
        sampleTuple = get_property(max_len,wordEmbed,entityEmbed,relationEmbed,dev_Sentence[i], dev_Entities[i], ctd_dev[i])
        dev_set.append(sampleTuple)
    #combine the test and development set
    train_set0 = train_set+dev_set
    train_set = []
    train_Label = []
    dev_set = []
    dev_Label = []
    dev_id = pubid[int(id_num*0.8):]
    for i in range(len(train_set0)):
        if train_set0[i][0] in dev_id:
            dev_set.append(train_set0[i])
            dev_Label.append(train_Label0[i])
        else:
            train_set.append(train_set0[i])
            train_Label.append(train_Label0[i])
    print ('generate the test data...')
    test_set = []
    for i in range(len(test_Sentence)):
        sampleTuple = get_property(max_len,wordEmbed,entityEmbed,relationEmbed,test_Sentence[i],test_Entities[i], ctd_test[i])
        test_set.append(sampleTuple)
    #########################Model###############################    
    model = func(wordEmbed,entityEmbed,relationEmbed, embedding_dim,embedding_dim_kg)
    print("model fitting - {}".format(model.name))
    return model.train(train_set,train_Label,dev_set,dev_Label,test_set,test_Label,save_path,dev_id,is_intra)

if __name__ == '__main__':
    from KCN_model import KCN
    ##############intra sentence level########################
    data_path ='./data_clean/CDR_intra_data_clean/data.pkl'
    ctd_path = './data_clean/CTD_intra_data_clean/data.pkl'
    save_path = './results/KCN/'
    if os.path.isdir(save_path):
        pass
    else:
        os.makedirs(save_path)
    #####train and predict########
    intra_path = train_and_test(data_path,ctd_path,save_path,True,KCN)

    ##############inter sentence level########################
    data_path ='./data_clean/CDR_inter_data_clean/data.pkl'
    ctd_path = './data_clean/CTD_inter_data_clean/data.pkl'
    if os.path.isdir(save_path):
        pass
    else:
        os.makedirs(save_path)
    #####train and predict########
    inter_path = train_and_test(data_path,ctd_path,save_path,False,KCN)
    merge_intra_inter(intra_path,inter_path,save_path)

