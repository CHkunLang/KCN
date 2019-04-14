# -*- coding: utf-8 -*-

import numpy as np
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from collections import defaultdict
import time
import argparse

from doc_level_evaluation import evaluate_score

#mySeed = np.random.RandomState(1234)
class KCN():
    def __init__(self, wordEmbed,entityEmbed,relationEmbed, embedding_dim,embedding_dim_kg, classNumber=2, numEpoches=30):
        self.name = 'KCN_model'
        self.wordEmbed = wordEmbed
        self.entityEmbed = entityEmbed
        self.relationEmbed = relationEmbed

        self.batchSize = 20
        self.wordVectorLength = embedding_dim

        self.vectorLength = embedding_dim
        self.entityLength = embedding_dim_kg
        self.classNumber = classNumber
        self.numEpoches = numEpoches
        self.convdim = 100
        self.dropout_train = nn.Dropout(p=0.5)
        self.dropout_test = nn.Dropout(p=0)

        self.kernel_size = [1,2,3,4,5]

        self.convstc = nn.ModuleList([nn.Conv1d(self.vectorLength, self.convdim, K, padding=int((K-1)/2), bias = True) for K in self.kernel_size]).cuda()
        self.convsc = nn.ModuleList([nn.Conv1d(self.vectorLength, self.convdim, K, padding=int((K-1)/2), bias = True) for K in self.kernel_size]).cuda()
        self.convstd = nn.ModuleList([nn.Conv1d(self.vectorLength, self.convdim, K, padding=int((K-1)/2), bias = True) for K in self.kernel_size]).cuda()
        self.convsd = nn.ModuleList([nn.Conv1d(self.vectorLength, self.convdim, K, padding=int((K-1)/2), bias = True) for K in self.kernel_size]).cuda()
        
        self.chemical_W = Variable(torch.FloatTensor(mySeed.uniform(-0.01, 0.01, (self.convdim, self.entityLength ))).cuda (), requires_grad = True)
        self.chemical_b = Variable(torch.FloatTensor(mySeed.uniform(-0.01, 0.01, (self.convdim, 1 ))).cuda (), requires_grad = True)
        self.disease_W  = Variable(torch.FloatTensor(mySeed.uniform(-0.01, 0.01, (self.convdim, self.entityLength ))).cuda (), requires_grad = True)
        self.disease_b  = Variable(torch.FloatTensor(mySeed.uniform(-0.01, 0.01, (self.convdim, 1 ))).cuda (), requires_grad = True)

        self.LinearLayer_W = Variable(torch.FloatTensor(mySeed.uniform(-0.01, 0.01, (self.convdim, self.convdim * 2 * len(self.kernel_size) ))).cuda (), requires_grad = True)
        self.LinearLayer_b = Variable(torch.FloatTensor(mySeed.uniform(-0.01, 0.01, (self.convdim, 1 ))).cuda (), requires_grad = True)

        self.attention_W = Variable(torch.FloatTensor(mySeed.uniform(-0.01, 0.01, (self.entityLength, self.convdim ))).cuda (), requires_grad = True)
        self.attention_b = Variable(torch.FloatTensor(mySeed.uniform(-0.01, 0.01, (self.entityLength, 1))).cuda (), requires_grad = True)
        
        self.softmaxLayer_W = Variable(torch.FloatTensor(mySeed.uniform(-0.01, 0.01, ( self.classNumber, self.convdim ))).cuda(), requires_grad=True)
        self.softmaxLayer_b = Variable(torch.FloatTensor(mySeed.uniform(-0.01, 0.01, ( self.classNumber,1 ))).cuda(), requires_grad=True)
        self.softmax = torch.nn.Softmax(dim = 1)
        self.loss_function = torch.nn.NLLLoss()  
  
    def forward(self, contxtWords, e1,e2, e1vs,e2vs,e1v,e2v, relation, senlength,is_train):
        softmaxLayer_W = self.softmaxLayer_W
        softmaxLayer_b = self.softmaxLayer_b
        vectorLength = self.vectorLength
        if is_train:
            dropout = self.dropout_train
        else:
            dropout = self.dropout_test

        #generate entity expression

        E1 = torch.mm(self.chemical_W, e1.view(self.wordVectorLength,1)) + self.chemical_b
        E2 = torch.mm(self.disease_W, e2.view(self.wordVectorLength,1)) + self.disease_b
        
        contxt_chem = []
        contxt_dis = []
        gate_chem = []
        gate_dis = []
        for i,conv in enumerate(self.convstc):
            if i%2:
                contxt_chem.append(torch.tanh(conv(torch.cat([contxtWords.view(1,vectorLength,senlength),Variable(torch.zeros(1,vectorLength,1).cuda())],2))))
            else:
                contxt_chem.append(torch.tanh(conv(contxtWords.view(1,vectorLength,senlength))))

        for i,conv in enumerate(self.convstd):
            if i%2:
                contxt_dis.append(torch.tanh(conv(torch.cat([contxtWords.view(1,vectorLength,senlength),Variable(torch.zeros(1,vectorLength,1).cuda())],2))))
            else:
                contxt_dis.append(torch.tanh(conv(contxtWords.view(1,vectorLength,senlength))))

        for i,conv in enumerate(self.convsc):
            if i%2:
                gate_chem.append(torch.relu(conv(torch.cat([contxtWords.view(1,vectorLength,senlength),Variable(torch.zeros(1,vectorLength,1).cuda())],2))+ E1.view(1,self.convdim,1)))# 
            else:
                gate_chem.append(torch.relu(conv(contxtWords.view(1,vectorLength,senlength)) + E1.view(1,self.convdim,1)))#

        for i,conv in enumerate(self.convsd):
            if i%2:
                gate_dis.append(torch.relu(conv(torch.cat([contxtWords.view(1,vectorLength,senlength),Variable(torch.zeros(1,vectorLength,1).cuda())],2))+ E2.view(1,self.convdim,1)))# 
            else:
                gate_dis.append(torch.relu(conv(contxtWords.view(1,vectorLength,senlength)) + E2.view(1,self.convdim,1)))#
    
        contxtWords_chem = [(i*j).squeeze(0) for i, j in zip(contxt_chem, gate_chem)]
        contxtWords_dis = [(i*j).squeeze(0) for i, j in zip(contxt_dis, gate_dis)]

        contxtWords0_chem = []
        contxtWords0_dis = []
        for contxt_chem,contxt_dis in zip(contxtWords_chem,contxtWords_dis):
            att = self.softmax( torch.mm(relation.view(1,self.entityLength), torch.tanh(torch.mm(self.attention_W,contxt_chem) + self.attention_b)) )
            contxtWords0_chem.append(torch.mm(att,contxt_chem.transpose(0,1)).view(self.convdim,1))

            att = self.softmax( torch.mm(relation.view(1,self.entityLength), torch.tanh(torch.mm(self.attention_W,contxt_dis) + self.attention_b)) )
            contxtWords0_dis.append(torch.mm(att,contxt_dis.transpose(0,1)).view(self.convdim,1))
        contxtWords0_chem = torch.cat(contxtWords0_chem,0)
        contxtWords0_dis = torch.cat(contxtWords0_dis,0)
        contxtWords0 = torch.cat([contxtWords0_chem,contxtWords0_dis],0)

        linearLayerOut = torch.relu(torch.mm(self.LinearLayer_W,dropout(contxtWords0)) + self.LinearLayer_b)
        finallinearLayerOut = torch.mm(softmaxLayer_W,linearLayerOut) + softmaxLayer_b
        return finallinearLayerOut

    def train(self, trainset,trainLabel,valset,valLabel,testset,testLabel,resultOutput,dev_id,is_intra):
        F1 = 0
        indicates=list(range(len(trainset)))
        trainsetSize = len(trainset)
        parameters =[
            self.wordEmbed.weight,
            self.entityEmbed.weight,
            self.relationEmbed.weight,
            self.chemical_W,
            self.chemical_b,
            self.disease_W,
            self.disease_b,
            self.attention_W,
            self.attention_b,
            self.softmaxLayer_W,
            self.softmaxLayer_b,
            self.LinearLayer_W,
            self.LinearLayer_b
        ]
        for conv in self.convstc:
            parameters = parameters + list (conv.parameters())
        for conv in self.convstd:
            parameters = parameters + list (conv.parameters())
        for conv in self.convsc:
            parameters = parameters + list (conv.parameters())
        for conv in self.convsd:
            parameters = parameters + list (conv.parameters())
        learn_rate = 0.0001 if is_intra else 0.0002
        optimizer = optim.Adam(parameters, lr = learn_rate)

        for epoch_idx in range (self.numEpoches):
            mySeed.shuffle(indicates)
            total_loss = Variable(torch.FloatTensor([0]).cuda(), requires_grad=True)
            sum_loss= 0.0
            print("=====================================================================")
            print("epoch " + str(epoch_idx) + ", trainSize: " + str(trainsetSize))

            count = 0
            correct = 0
            time0 = time.time()
            for i in range(len(indicates)):
                sentid, sentwords,e1id,e2id,e1,e2,e1vs,e2vs,e1v,e2v,relation, senlength= trainset[indicates[i]]
                finallinearLayerOut =  self.forward(
                    sentwords,
                    e1,
                    e2,
                    e1vs,
                    e2vs,
                    e1v,
                    e2v,
                    relation,
                    senlength,
                    True
                )
                log_prob = F.log_softmax(finallinearLayerOut.view(1, self.classNumber),dim = 1)
                loss = self.loss_function(log_prob, Variable(torch.LongTensor([trainLabel[indicates[i]]]).cuda()))
                classification = self.softmax(finallinearLayerOut.view(1, self.classNumber))

                total_loss = torch.add(total_loss, loss)

                predict = np.argmax(classification.cpu().data.numpy())
                if predict == trainLabel[indicates[i]]:
                    correct += 1.0
                count += 1
####################Update#######################
                if count % self.batchSize == 0:
                    total_loss = total_loss/self.batchSize
                    total_loss.backward(retain_graph=True)
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss = Variable(torch.FloatTensor([0]).cuda(),requires_grad = True)
            optimizer.step()
            optimizer.zero_grad()
    
            resultStream = open(resultOutput + "vresult_" + str(epoch_idx) + ".txt", 'w')
            probPath   = resultOutput + "vprob_" + str(epoch_idx) + ".txt"
            VP,VR,VF = self.test(valset,valLabel,dev_id,resultStream, probPath)

            resultStream = open(resultOutput + "tresult_" + str(epoch_idx) + ".txt", 'w')
            probPath   = resultOutput + "tprob_" + str(epoch_idx) + ".txt"
            TP,TR,TF = self.test(testset,testLabel,[], resultStream, probPath)

            resultStream.close()
            evaluate_score(self.name,resultOutput + "tresult_" + str(epoch_idx) + ".txt" , is_intra)

            if VF >= F1:
                F1 = VF
                file_path = resultOutput + "tresult_" + str(epoch_idx) + ".txt"
####################Update#######################
            
            time1 = time.time()
            print("val  P: ", VP, " R: ", VR , " F1: ", VF)
            print("test P: ", TP, " R: ", TR , " F1: ", TF)
            print("Iteration", epoch_idx, "Loss", total_loss.cpu().data.numpy()[0] / self.batchSize, "train Acc: ", float(correct / count) , "time: ", str(time1 - time0))
        return file_path

    def test(self, testset, testLabel,dev_id,resultStream, probPath):
        time0 = time.time()
        probs = []
        predict_dic = []
        test_dic = []
        correct = 0
        count = 0
        gold_correct = 0
        for i in range(len(testset)):
            sentid, sentwords, e1id,e2id,e1,e2,e1vs,e2vs,e1v,e2v,relation,senlength = testset[i]
            finallinearLayerOut =  self.forward(
                sentwords,
                e1,
                e2,
                e1vs,
                e2vs,
                e1v,
                e2v,
                relation,
                senlength,
                False
            )
            classification = self.softmax(finallinearLayerOut.view(1, self.classNumber))
            prob = classification.cpu().data.numpy().reshape(self.classNumber)
            predict = np.argmax(prob)
            probs.append(prob)
            prediction = [sentid,e1id,e2id,predict]
            if predict and (prediction not in predict_dic):
                resultStream.write("\t".join([sentid, "CID", e1id, e2id]) + "\n")
                predict_dic.append(prediction)
                if predict and testLabel[i]:
                    correct += 1
                count += 1
        
        if dev_id != []:
            gold_correct = 0
            with open ('./data_clean/PubGold.txt') as f:
                for line in f:
                    line = line.strip().split('\t')
                    if line[0] in dev_id:
                        gold_correct += 1
        else:
            gold_correct = 1066
        
        P = correct/max(count,1)
        R = correct/gold_correct
        F = 2*P*R/max(0.0001,(P+R))
        if probPath:
            np.savetxt(probPath, probs, '%.5f',delimiter=' ')

        time1 = time.time()
        print("test time : ", str(time1 - time0))
        return P,R,F
