# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 10:16:02 2021

@author: Xianghe Zhu
"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import csv
import random
from propy.CTD import CalculateCTD

#open the ProtVec and mapping embeddings
with open('protVec_100d_3grams.csv', mode='r') as f:
    reader = csv.DictReader(f,delimiter = '\t')
    name_100d3 = list(reader.fieldnames)
    name_100d3 = name_100d3[1:101]
    
gram_100d3 = pd.read_csv('protVec_100d_3grams.csv',delimiter = '\t')
seq_name = list(gram_100d3['words'])
seq_value = gram_100d3[name_100d3].values
print(seq_value.shape)

def getKmers(sequence, size):
    return [sequence[x:x+size] for x in range(len(sequence) - size + 1)]
#QR decomposition
def intlabel(aaname):  
    Btworandom = 'DN'
    Jtworandom = 'IL'
    Ztworandom = 'EQ'
    Xallrandom = 'ACDEFGHIKLMNPQRSTVWY'
    sequencea = aaname
    data_integera = []
    seqa=getKmers(sequencea, 3)
    l_max = len(aaname)
    for i in range(l_max-2):
        tempa = seqa[i]
        for j in range(len(seqa[i])):
          lis = tempa[j]
          klis = lis.replace('B',random.choice(Btworandom))
          klis = klis.replace('J',random.choice(Jtworandom))
          klis = klis.replace('Z',random.choice(Ztworandom))
          klis = klis.replace('X',random.choice(Xallrandom))
          tempa = tempa.replace(lis,klis)
        data_integera.append(seq_name.index(tempa))
    train_dataa = np.zeros((l_max-2,100))
    for j in range(l_max-2):
        train_dataa[j][:] = seq_value[data_integera[j]]
    q,r = np.linalg.qr(train_dataa.T,mode = 'complete')
    return q#np.array([sum(x) for x in zip(*train_dataa)])

def aac(aaname):
    N = len(aaname)
    Btworandom = 'DN'
    Jtworandom = 'IL'
    Ztworandom = 'EQ'
    Xallrandom = 'ACDEFGHIKLMNPQRSTVWY'

    #l_max = len(aaname)
    aac = np.zeros((20,))
    lis = aaname
    klis = lis.replace('B',random.choice(Btworandom))
    klis = klis.replace('J',random.choice(Jtworandom))
    klis = klis.replace('Z',random.choice(Ztworandom))
    klis = klis.replace('X',random.choice(Xallrandom))
    aaname = aaname.replace(lis,klis)
    for j in range(N):
          for k in range(20):

            if aaname[j] == Xallrandom[k]:
              aac[k] = aac[k]+1
    aac = aac/N
    return aac

def dpc(aaname):
    N = len(aaname)
    Btworandom = 'DN'
    Jtworandom = 'IL'
    Ztworandom = 'EQ'
    Xallrandom = 'ACDEFGHIKLMNPQRSTVWY'
    sequencea = aaname
    #data_integera = []
    seqa=getKmers(sequencea, 2)
    #l_max = len(aaname)
    aac = np.zeros((20,20))
    for i in range(N-1):
        tempa = seqa[i]
        for j in range(len(seqa[i])):
          lis = tempa[j]
          klis = lis.replace('B',random.choice(Btworandom))
          klis = klis.replace('J',random.choice(Jtworandom))
          klis = klis.replace('Z',random.choice(Ztworandom))
          klis = klis.replace('X',random.choice(Xallrandom))
          tempa = tempa.replace(lis,klis)
        for j in range(20):
          for k in range(20):

            if tempa[0] == Xallrandom[j] and tempa[1] == Xallrandom[k]:
              aac[j][k] = aac[j][k]+1
    aac = aac/(N-1)
    return aac

aap_matrix = np.loadtxt(open("aap.csv","rb"),delimiter=",")
def aap(aaname):
    N = len(aaname)
    Btworandom = 'DN'
    Jtworandom = 'IL'
    Ztworandom = 'EQ'
    Xallrandom = 'ACDEFGHIKLMNPQRSTVWY'
    sequencea = aaname
    #data_integera = []
    seqa=getKmers(sequencea, 2)
    #l_max = len(aaname)
    aac = np.zeros((N-1,))
    for i in range(N-1):
        tempa = seqa[i]
        for j in range(len(seqa[i])):
          lis = tempa[j]
          klis = lis.replace('B',random.choice(Btworandom))
          klis = klis.replace('J',random.choice(Jtworandom))
          klis = klis.replace('Z',random.choice(Ztworandom))
          klis = klis.replace('X',random.choice(Xallrandom))
          tempa = tempa.replace(lis,klis)
        for j in range(20):
          for k in range(20):

            if tempa[0] == Xallrandom[j] and tempa[1] == Xallrandom[k]:
              aac[i] = aap_matrix[j][k]
    
    return aac


def ctd(input_file):
    global feature_file, feature
    feature_file = open("feature.csv", 'w')
    input_file = pd.read_csv(input_file)
    input_file = input_file['Alignment']
    count = 0
    for row in range(0, len(input_file)):
        seq = str(input_file.loc[row])
        feature = CalculateCTD(seq)
        if row == 0:
            write_header() 
        for key in sorted(feature.keys()):
            feature_value = feature[key]
            count = count + 1
            if count == 147:
                count = 0
                write_to_csv = str(feature_value) + '\n'
            else:
                write_to_csv = str(feature_value) + ','   
            #write_to_csv = str(feature_value) + ','     
            feature_file.write(write_to_csv)
            
    feature_file.close()                
    #return feature_value
        


def write_header():
#    feature_file = open(os.path.expanduser("C:/Users/Rayin/Google Drive/Tier_2_MOE2014/2_Conference/GIW/Data/feature/feature.csv"), 'w')
    column = 0
    for key in sorted(feature.keys()):
        feature_header = key
        column = column + 1
        if column == 147:
            column = 0
            write_header = str(feature_header) + '\n'
        else:
            write_header = str(feature_header) + ','
        feature_file.write(write_header)
        
ctd('virus_realddd.csv')