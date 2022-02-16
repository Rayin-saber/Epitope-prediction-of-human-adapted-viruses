# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 10:31:55 2021import warnings


@author: Xianghe Zhu
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import random
from torch.autograd import Variable
from sklearn import neighbors
from sklearn import svm
from sklearn import ensemble
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import math
import torch
import torch.nn as nn
    

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def reshape_to_linear(x):
    output = np.reshape(x, (x.shape[0], -1))

    return output


# split data into training and testing
def train_test_split_data(feature, label, split_ratio, shuffled_flag):
    #setup_seed(m)
    train_x, test_x, train_y, test_y = [], [], [], []
    feature_new, label_new = [], []
    num_of_training = int(math.floor(len(feature) * (1 - split_ratio)))
    if shuffled_flag == True:
        shuffled_index = np.arange(len(feature))
        random.shuffle(shuffled_index)
    else:
        shuffled_index = np.arange(len(feature))
    for i in range(0, len(feature)):
        feature_new.append(feature[shuffled_index[i]])
        label_new.append(label[shuffled_index[i]])

    train_x = feature_new[:num_of_training]
    train_y = label_new[:num_of_training]
    test_x = feature_new[num_of_training:]
    test_y = label_new[num_of_training:]

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    return train_x, test_x, train_y, test_y
    

from xgboost import XGBClassifier
def cross_val(x_train, y_train,x_test,y_test):
  a = np.zeros((7,5))
  a6 = np.zeros((7,5))
  print(y_train.shape)
  #y_train = y_train.reshape((y_train.shape[0],))
  for i in range(7):
    if i==0:
      clf = neighbors.KNeighborsClassifier(algorithm='auto', leaf_size=30,
                                     metric='minkowski',
                                     metric_params=None, n_jobs=None,
                                     n_neighbors=8, p=1,
                                     weights='distance')
      print("=====================knn-cross====================")
      b = np.zeros(5)
      accuracy = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy') 
      #print(accuracy)
      b[0] = accuracy.mean()
      #calculate the precision
      precision = cross_val_score(clf, x_train, y_train, cv=5, scoring='precision_macro')
      b[1] = precision.mean()
      #calculate the recall score
      recall = cross_val_score(clf, x_train, y_train, cv=5, scoring='recall_macro')
      b[2] = recall.mean()
      #calculate the f_measure
      f_measure = cross_val_score(clf, x_train, y_train, cv=5, scoring='f1_macro')
      b[3] = f_measure.mean()
      auc = cross_val_score(clf, x_train, y_train, cv=5, scoring='roc_auc')
      b[4] = auc.mean()
      print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_AUC %.3f'
            % (accuracy.mean(), precision.mean(), recall.mean(), f_measure.mean(), auc.mean()))
      a[i] = b
      print("=====================KNN-test====================")

      clf = clf.fit(x_train,y_train)
      predicted = clf.predict(x_test)
      predicted = np.array(predicted)
      train_auc = roc_auc_score(y_test, predicted)
      train_acc = accuracy_score(y_test, predicted)
      train_pre = precision_score(y_test, predicted)
      train_rec = recall_score(y_test, predicted)
      train_fscore = f1_score(y_test, predicted)
      #train_mcc = matthews_corrcoef(y_test, predicted)
      outcome = [train_acc, train_pre, train_rec, train_fscore,train_auc]
      a6[i] = outcome
      print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_AUC %.3f'
            % (train_acc, train_pre, train_rec, train_fscore,train_auc))
    elif i ==1:
      clf = svm.SVC()
      print("=====================svm-cross====================")
      b = np.zeros(5)
      accuracy = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy') 
      b[0] = accuracy.mean()
      #calculate the precision
      precision = cross_val_score(clf, x_train, y_train, cv=5, scoring='precision_macro')
      b[1] = precision.mean()
      #calculate the recall score
      recall = cross_val_score(clf, x_train, y_train, cv=5, scoring='recall_macro')
      b[2] = recall.mean()
      #calculate the f_measure
      f_measure = cross_val_score(clf, x_train, y_train, cv=5, scoring='f1_macro')
      b[3] = f_measure.mean()
      auc = cross_val_score(clf, x_train, y_train, cv=5, scoring='roc_auc')
      b[4] = auc.mean()
      print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_AUC %.3f'
            % (accuracy.mean(), precision.mean(), recall.mean(), f_measure.mean(), auc.mean()))
      a[i] = b
      print("=====================svm-test====================")

      clf = clf.fit(x_train,y_train)
      predicted = clf.predict(x_test)
      predicted = np.array(predicted)
      train_auc = roc_auc_score(y_test, predicted)
      train_acc = accuracy_score(y_test, predicted)
      train_pre = precision_score(y_test, predicted)
      train_rec = recall_score(y_test, predicted)
      train_fscore = f1_score(y_test, predicted)
      #train_mcc = matthews_corrcoef(y_test, predicted)
      outcome = [train_acc, train_pre, train_rec, train_fscore,train_auc]
      a6[i] = outcome
      print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_AUC %.3f'
            % (train_acc, train_pre, train_rec, train_fscore,train_auc))
    elif i ==2:
      clf = linear_model.LogisticRegression()
      print("=====================lr-cross====================")
      b = np.zeros(5)
      accuracy = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy') 
      b[0] = accuracy.mean()
      #calculate the precision
      precision = cross_val_score(clf, x_train, y_train, cv=5, scoring='precision_macro')
      b[1] = precision.mean()
      #calculate the recall score
      recall = cross_val_score(clf, x_train, y_train, cv=5, scoring='recall_macro')
      b[2] = recall.mean()
      #calculate the f_measure
      f_measure = cross_val_score(clf, x_train, y_train, cv=5, scoring='f1_macro')
      b[3] = f_measure.mean()
      auc = cross_val_score(clf, x_train, y_train, cv=5, scoring='roc_auc')
      b[4] = auc.mean()
      print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_AUC %.3f'
            % (accuracy.mean(), precision.mean(), recall.mean(), f_measure.mean(), auc.mean()))
      a[i] = b
      print("=====================lr-test====================")

      clf = clf.fit(x_train,y_train)
      predicted = clf.predict(x_test)
      predicted = np.array(predicted)
      train_auc = roc_auc_score(y_test, predicted)
      train_acc = accuracy_score(y_test, predicted)
      train_pre = precision_score(y_test, predicted)
      train_rec = recall_score(y_test, predicted)
      train_fscore = f1_score(y_test, predicted)
      #train_mcc = matthews_corrcoef(y_test, predicted)
      outcome = [train_acc, train_pre, train_rec, train_fscore,train_auc]
      a6[i] = outcome
      print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_AUC %.3f'
            % (train_acc, train_pre, train_rec, train_fscore,train_auc))
    elif i==3:
      clf = GaussianNB()
      print("=================Gaussian====bayes-cross====================")
      b = np.zeros(5)
      accuracy = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy') 
      b[0] = accuracy.mean()
      #calculate the precision
      precision = cross_val_score(clf, x_train, y_train, cv=5, scoring='precision_macro')
      b[1] = precision.mean()
      #calculate the recall score
      recall = cross_val_score(clf, x_train, y_train, cv=5, scoring='recall_macro')
      b[2] = recall.mean()
      #calculate the f_measure
      f_measure = cross_val_score(clf, x_train, y_train, cv=5, scoring='f1_macro')
      b[3] = f_measure.mean()
      auc = cross_val_score(clf, x_train, y_train, cv=5, scoring='roc_auc')
      b[4] = auc.mean()
      print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_AUC %.3f'
            % (accuracy.mean(), precision.mean(), recall.mean(), f_measure.mean(), auc.mean()))
      a[i] = b
      print("================Gaussian=====bayes-test====================")

      clf = clf.fit(x_train,y_train)
      predicted = clf.predict(x_test)
      predicted = np.array(predicted)
      train_auc = roc_auc_score(y_test, predicted)
      train_acc = accuracy_score(y_test, predicted)
      train_pre = precision_score(y_test, predicted)
      train_rec = recall_score(y_test, predicted)
      train_fscore = f1_score(y_test, predicted)
      #train_mcc = matthews_corrcoef(y_test, predicted)
      outcome = [train_acc, train_pre, train_rec, train_fscore,train_auc]
      a6[i] = outcome
      print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_AUC %.3f'
            % (train_acc, train_pre, train_rec, train_fscore,train_auc))
      
    elif i==4:
      clf = ensemble.RandomForestClassifier()
      print("=====================rf-cross====================")
      b = np.zeros(5)
      accuracy = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy') 
      b[0] = accuracy.mean()
      #calculate the prec.mean()ision
      precision = cross_val_score(clf, x_train, y_train, cv=5, scoring='precision_macro')
      b[1] = precision.mean()
      #calculate the recall score
      recall = cross_val_score(clf, x_train, y_train, cv=5, scoring='recall_macro')
      b[2] = recall.mean()
      #calculate the f_measure
      f_measure = cross_val_score(clf, x_train, y_train, cv=5, scoring='f1_macro')
      b[3] = f_measure.mean()
      auc = cross_val_score(clf, x_train, y_train, cv=5, scoring='roc_auc')
      b[4] = auc.mean()
      print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_AUC %.3f'
            % (accuracy.mean(), precision.mean(), recall.mean(), f_measure.mean(), auc.mean()))
      a[i] = b
      print("=====================rf-test====================")

      clf = clf.fit(x_train,y_train)
      predicted = clf.predict(x_test)
      predicted = np.array(predicted)
      train_auc = roc_auc_score(y_test, predicted)
      train_acc = accuracy_score(y_test, predicted)
      train_pre = precision_score(y_test, predicted)
      train_rec = recall_score(y_test, predicted)
      train_fscore = f1_score(y_test, predicted)
      #train_mcc = matthews_corrcoef(y_test, predicted)
      outcome = [train_acc, train_pre, train_rec, train_fscore,train_auc]
      a6[i] = outcome
      print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_AUC %.3f'
            % (train_acc, train_pre, train_rec, train_fscore,train_auc))
    elif i==5:
      clf = MLPClassifier(hidden_layer_sizes = (50,50,50),activation = 'relu',solver = 'adam',
                            alpha = 0.001,learning_rate = 'adaptive')
      print("=====================nn-cross====================")
      b = np.zeros(5)
      accuracy = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy') 
      b[0] = accuracy.mean()
      #calculate the precision
      precision = cross_val_score(clf, x_train, y_train, cv=5, scoring='precision_macro')
      b[1] = precision.mean()
      #calculate the recall score
      recall = cross_val_score(clf, x_train, y_train, cv=5, scoring='recall_macro')
      b[2] = recall.mean()
      #calculate the f_measure
      f_measure = cross_val_score(clf, x_train, y_train, cv=5, scoring='f1_macro')
      b[3] = f_measure.mean()
      auc = cross_val_score(clf, x_train, y_train, cv=5, scoring='roc_auc')
      b[4] = auc.mean()
      print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_AUC %.3f'
            % (accuracy.mean(), precision.mean(), recall.mean(), f_measure.mean(), auc.mean()))
      a[i] = b
      print("=====================nn-test====================")

      clf = clf.fit(x_train,y_train)
      predicted = clf.predict(x_test)
      predicted = np.array(predicted)
      train_auc = roc_auc_score(y_test, predicted)
      train_acc = accuracy_score(y_test, predicted)
      train_pre = precision_score(y_test, predicted)
      train_rec = recall_score(y_test, predicted)
      train_fscore = f1_score(y_test, predicted)
      #train_mcc = matthews_corrcoef(y_test, predicted)
      outcome = [train_acc, train_pre, train_rec, train_fscore,train_auc]
      a6[i] = outcome
      print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_AUC %.3f'
            % (train_acc, train_pre, train_rec, train_fscore,train_auc))
    elif i ==6:
        if x_train.shape[1]== 10147:
            clf = XGBClassifier(base_score=0.5,
                                booster=None, 
                                colsample_bylevel=1,
                                colsample_bynode=1, 
                                colsample_bytree=0.85, 
                                gamma=0.2, 
                                gpu_id=None,
                                importance_type='gain', 
                                interaction_constraints=None,
                                learning_rate=0.05, 
                                max_delta_step=0, 
                                max_depth=9,
                                min_child_weight=3, 
                                #missing=nan, 
                                monotone_constraints=None,
                                n_estimators=180, 
                                n_jobs=None, 
                                num_parallel_tree=1,
                                random_state=283, 
                                reg_alpha=0.05, 
                                reg_lambda=1, 
                                scale_pos_weight=1,
                                subsample=0.75, 
                                #;;u   tree_method=auto, 
                                validate_parameters=True,
                                verbosity=1)
        else:
            clf = XGBClassifier()
        print("=====================XGboost-cross====================")
        b = np.zeros(5)
        accuracy = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy') 
        b[0] = accuracy.mean()
      #calculate the precision
        precision = cross_val_score(clf, x_train, y_train, cv=5, scoring='precision_macro')
        b[1] = precision.mean()
      #calculate the recall score
        recall = cross_val_score(clf, x_train, y_train, cv=5, scoring='recall_macro')
        b[2] = recall.mean()
      #calculate the f_measure
        f_measure = cross_val_score(clf, x_train, y_train, cv=5, scoring='f1_macro')
        b[3] = f_measure.mean()
        auc = cross_val_score(clf, x_train, y_train, cv=5, scoring='roc_auc')
        b[4] = auc.mean()
        print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_AUC %.3f'
              % (accuracy.mean(), precision.mean(), recall.mean(), f_measure.mean(), auc.mean()))
        a[i] = b
        print("=====================xgboost-test====================")

        clf = clf.fit(x_train,y_train)
        predicted = clf.predict(x_test)
        predicted = np.array(predicted)
        train_auc = roc_auc_score(y_test, predicted)
        train_acc = accuracy_score(y_test, predicted)
        train_pre = precision_score(y_test, predicted)
        train_rec = recall_score(y_test, predicted)
        train_fscore = f1_score(y_test, predicted)
        #train_mcc = matthews_corrcoef(y_test, predicted)
        outcome = [train_acc, train_pre, train_rec, train_fscore,train_auc]
        a6[i] = outcome
        print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_AUC %.3f'
              % (train_acc, train_pre, train_rec, train_fscore,train_auc))
  return a,a6

    

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        self.dropout = nn.Dropout(p=0.1)
        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        
        # Readout layer
        #self.fc1 = nn.Linear(, 8192)
       # self.fc3 = nn.Linear(8192, 256)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        x = self.dropout(x) 
        # One time step
        #x_packed = pack_padded_sequence(x, x_lens,batch_first=True,enforce_sorted=False)
        out, hn = self.rnn(x, h0)
        #out, lens = pad_packed_sequence(out, batch_first=True)
        out = self.fc2(out[:, -1, :]) 
        #out = self.fc3(out) 
        #out = self.fc2(out) 
        return out


class LSTM1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        #self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(GRUModel, self).__init__()
        
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        self.dropout = nn.Dropout(p=0.1)
        # RNN
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)#, nonlinearity='relu'), layer_dim
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        
        # Initialize hidden state with zeros 
        h0 = Variable(torch.zeros(self.layer_dim,x.size(0), self.hidden_dim))
            
        # One time step
        x = self.dropout(x)
        out, hn = self.gru(x, h0)
        #out, lens = pad_packed_sequence(out, batch_first=True)
        out = self.fc(out[:, -1, :]) 
        return out
      

class VGG(nn.Module):
    def __init__(self, model,m):
        super(VGG, self).__init__()
        self.conv_layer = nn.Conv2d(1, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.resnext_layer = nn.Sequential(*list(model.children())[1:-1])
        #self.Linear_layer1 = nn.Linear(3136, 256)  # need edition during different segments' training !!!
        self.Linear_layer2 = nn.Linear(6272, 2)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.resnext_layer(x)
        x = x.view(x.size(0), -1)
        #x = self.Linear_layer1(x)
        out = self.Linear_layer2(x)
        out = self.dropout(out)
        return out
    
class AlexNet(nn.Module):
    def __init__(self, model,m):
        super(AlexNet, self).__init__()
        self.conv_layer = nn.Conv2d(1, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.resnext_layer = nn.Sequential(*list(model.children())[1:-1])
        #self.Linear_layer1 = nn.Linear(2304, 256)
        self.Linear_layer2 = nn.Linear(4608, 2)
        self.dropout = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.resnext_layer(x)
        x = x.view(x.size(0), -1)
        #x = self.Linear_layer1(x)
        out = self.Linear_layer2(x)
        out = self.dropout(out)
        out = self.softmax(out)
        return out


class SqueezeNet(nn.Module):
    def __init__(self, model, prior, m):
        super(SqueezeNet, self).__init__()
        self.conv_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.squeeze_layer = nn.Sequential(*list(model.children())[1:-1])

        self.Linear_layer1 = nn.Linear(m,2)
        #self.Linear_layer2 = nn.Linear(20480, 2048)
        #self.Linear_layer3 = nn.Linear(2048^2, 2)
        self.dropout = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.squeeze_layer(x)
        # x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        #x = self.Linear_layer1(x)
        #x = self.Linear_layer2(x)
        out = self.Linear_layer1(x)
        out = self.dropout(out)
        out = self.softmax(out)
        return out