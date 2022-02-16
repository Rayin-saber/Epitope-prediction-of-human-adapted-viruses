# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 10:49:05 2021

@author: xiang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 16:57:30 2021

@author: rayin
"""
import warnings
warnings.filterwarnings("ignore")
import torchvision.models as models
import torch
import pandas as pd
import numpy as np
import warnings
from models import setup_seed
from validation import evaluate
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from models import cross_val

from models import RNNModel
from models import LSTM1
from models import VGG
from models import AlexNet
from models import SqueezeNet
from models import GRUModel
from train import train_cnn
from train import train_rnn
from train import predictions_from_output
from train import calculate_prob

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(mode, features):

    
                
    if mode == 'TL':
            if features =='ProtVec':
                        features_numpy0 = pd.read_csv('x_train0.csv', header=None)
                        train_feature = np.array(features_numpy0)
                        targets_numpy = pd.read_csv('y_train.csv', header=None)
                        train_label = np.array(targets_numpy)
                        features_numpy0 = pd.read_csv('x_test0.csv', header=None)
                        test_feature = np.array(features_numpy0)
                        targets_numpy = pd.read_csv('y_test.csv', header=None)
                        test_label = np.array(targets_numpy)
                        cross_val(train_feature, train_label, test_feature, test_label)
                        
            elif features =='ProtVec + ctd':
                        
                        features_numpy0 = pd.read_csv('x_train1.csv', header=None)
                        train_feature = np.array(features_numpy0)
                        targets_numpy = pd.read_csv('y_train.csv', header=None)
                        train_label = np.array(targets_numpy)
                        features_numpy0 = pd.read_csv('x_test1.csv', header=None)
                        test_feature = np.array(features_numpy0)
                        targets_numpy = pd.read_csv('y_test.csv', header=None)
                        test_label = np.array(targets_numpy)
                        cross_val(train_feature, train_label, test_feature, test_label)
                        
            elif features =='ctd':
                        features_numpy0 = pd.read_csv('x_train2.csv', header=None)
                        train_feature = np.array(features_numpy0)
                        targets_numpy = pd.read_csv('y_train.csv', header=None)
                        train_label = np.array(targets_numpy)
                        features_numpy0 = pd.read_csv('x_test2.csv', header=None)
                        test_feature = np.array(features_numpy0)
                        targets_numpy = pd.read_csv('y_test.csv', header=None)
                        test_label = np.array(targets_numpy)
                        cross_val(train_feature, train_label, test_feature, test_label)
                        
            
            
    else:
        if features =='ProtVec':
                
                features_numpy0 = pd.read_csv('x_train.csv', header=None)
                train_feature = np.array(features_numpy0)
                targets_numpy = pd.read_csv('y_train.csv', header=None)
                train_label = np.array(targets_numpy)
                features_numpy0 = pd.read_csv('x_test.csv', header=None)
                test_feature = np.array(features_numpy0)
                targets_numpy = pd.read_csv('y_test.csv', header=None)
                test_label = np.array(targets_numpy)
                m = 100
                n = 100
                k = 160000
                
        elif features =='ProtVec + ctd':
                    
                    features_numpy0 = pd.read_csv('x_train1.csv', header=None)
                    train_feature = np.array(features_numpy0)
    
                    train_feature = np.append(train_feature,np.zeros((features_numpy0.shape[0],54)),axis=1)
                    targets_numpy = pd.read_csv('y_train.csv', header=None)
                    train_label = np.array(targets_numpy)
                    features_numpy0 = pd.read_csv('x_test1.csv', header=None)
                    test_feature = np.array(features_numpy0)
                    test_feature = np.append(test_feature,np.zeros((features_numpy0.shape[0],54)),axis=1)
                    targets_numpy = pd.read_csv('y_test.csv', header=None)
                    test_label = np.array(targets_numpy)
                    m = 101
                    n = 101
                    k = 166464
                    
        elif features =='ctd':
                    features_numpy0 = pd.read_csv('x_train2.csv', header=None)
                    train_feature = np.array(features_numpy0)
                    train_feature = np.append(train_feature,np.zeros((features_numpy0.shape[0],22)),axis=1)
                    targets_numpy = pd.read_csv('y_train.csv', header=None)
                    train_label = np.array(targets_numpy)
                    features_numpy0 = pd.read_csv('x_test2.csv', header=None)
                    test_feature = np.array(features_numpy0)
                    test_feature = np.append(test_feature,np.zeros((features_numpy0.shape[0],22)),axis=1)
                    targets_numpy = pd.read_csv('y_test.csv', header=None)
                    test_label = np.array(targets_numpy)
                    k = 3136
                    m = 13
                    n = 13
                    
        if mode == 'RNN':
              setup_seed(18)
              train_results = []
              test_results = []
              train_label = train_label.reshape((len(train_label),))
              test_label = test_label.reshape((len(test_label),))
              stratified_folder = KFold(n_splits=5,random_state=2, shuffle=True) 
              best_acc = 0
              
              parameters = {
    
                  'learning_rate': 0.0005,  
            
                  'batch_size': 4,
            
                  'num_of_epochs': 100
                  }
              input_dim = n   # input dimension
              hidden_dim = 25  # hidden layer dimension
              layer_dim = 1     # number of hidden layers
              output_dim = 8
              vacc = []
              for method in ['RNN','LSTM','GRU']:
            
            
                  for idx, (train_index, test_index) in enumerate(stratified_folder.split(train_feature, train_label)):
                      if method == 'GRU':
                          net = GRUModel(input_dim, hidden_dim, layer_dim, output_dim)
                          print("Using GRU...")
                      elif method == 'RNN':
                          net = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
                          print("Using RNN...")
                      elif method == 'LSTM':
                          net = LSTM1(input_dim, hidden_dim, layer_dim, output_dim)
                          print("Using LSTM...")  
      
                      train_x = np.array(train_feature)[train_index]
                      test_x = np.array(train_feature)[test_index]
                      train_y = np.array(train_label)[train_index]
                      test_y = np.array(train_label)[test_index]
            
                      train_x = np.reshape(train_x, (np.array(train_x).shape[0],m,n))
                      test_x = np.reshape(test_x, (np.array(test_x).shape[0], m,n))
            
                      train_x = torch.tensor(train_x, dtype=torch.float32)
                      train_y = torch.tensor(train_y, dtype=torch.int64)
                      test_x = torch.tensor(test_x, dtype=torch.float32)
                      test_y = torch.tensor(test_y, dtype=torch.int64)
     
                      train_result, test_result, best_acc = train_rnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x, train_y, test_x, test_y, False, best_acc)
                      train_results.append(train_result)
                      test_results.append(test_result)
                  T_results = np.array(train_results).mean(axis=0)
                  V_results = np.array(test_results).mean(axis=0)
                        
                  print("##########################################################################")
                  print("After 5 folders, the results are:")
                  print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f\tAUC %.3f' % (T_results[0], T_results[1], T_results[2], T_results[3], T_results[4], T_results[5]))
                  print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f\tAUC %.3f' % (V_results[0], V_results[1], V_results[2], V_results[3], V_results[4],V_results[5]))
                  if method == 'GRU':
                      net = GRUModel(input_dim, hidden_dim, layer_dim, output_dim)
                      print("Using GRU...")
                  elif method == 'RNN':
                      net = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
                      print("Using RNN...")
                  elif method == 'LSTM':
                      net = LSTM1(input_dim, hidden_dim, layer_dim, output_dim)
                      print("Using LSTM...") 
                  val_x = np.reshape(test_feature, (np.array(test_feature).shape[0],m,n))
                  val_x = torch.tensor(val_x, dtype=torch.float32)
                  val_y = torch.tensor(test_label, dtype=torch.int64)
                  net.load_state_dict(torch.load('rnn_no_pr.pkl'))
                  net.eval()
                  val_scores = net(val_x)
                  prediction = predictions_from_output(val_scores)
                  pred_prob = calculate_prob(val_scores)
                  fpr_cnn, tpr_cnn, _ = roc_curve(val_y.cpu().detach().numpy(), pred_prob.cpu().detach().numpy())
                  AUC = auc(fpr_cnn, tpr_cnn)
                  prediction = prediction.view_as(val_y)
                  precision, recall, fscore, mcc, val_acc = evaluate(val_y, prediction)
                  print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f\tAUC %.3f' % (val_acc, precision, recall, fscore, mcc,AUC))
                  vv = np.array([val_acc, precision, recall, fscore, mcc,AUC])
                  vacc.append(V_results)
                  vacc.append(vv)
              return np.array(vacc)
          
        elif mode == 'CNN':
            train_label = train_label.reshape((len(train_label),))
            test_label = test_label.reshape((len(test_label),))
            stratified_folder = KFold(n_splits=5, random_state=2, shuffle=True) 
            best_acc1 = 0
            train_results = []
            test_results = []
            vacc = []
            
            parameters = {
    
                        'learning_rate': 0.0001,  
    
                        'batch_size': 2,
    
                        # Number of training iterations
                        'num_of_epochs': 100
                    }
            for method in ['VGG','AlexNet','SqueezeNet']:
                acac = []
                if os.path.exists("cnn_no_pr.pkl"):
                      os.remove("cnn_no_pr.pkl")
                for idx, (train_index, test_index) in enumerate(stratified_folder.split(train_feature, train_label)):
                    
                    print(idx+1, "th folder...")
        
                   
                    if method == 'AlexNet':
                        net = AlexNet(models.alexnet(pretrained=False),m=k)
                        print("Using AlexNet...")
                    elif method == 'SqueezeNet':
                        net = SqueezeNet(models.squeezenet1_0(pretrained=True), prior=False,m = 166464)
                        print("Using SqueezeNet...")
                        parameters = {
                              # Note, no learning rate decay implemented
                              'learning_rate': 0.00005, 
    
                              # Size of mini batch
                              'batch_size': 4,
    
                              # Number of training iterations
                              'num_of_epochs': 100
                        }
                    elif method == 'VGG':
                        net = VGG(models.vgg16(pretrained=True),m=k)
                        print("Using VGG-16...")
                    if torch.cuda.is_available():
                        print('running with GPU')
                        net.cuda()
                    
        
            
        
                    train_x = np.array(train_feature)[train_index]
                    test_x = np.array(train_feature)[test_index]
                    train_y = np.array(train_label)[train_index]
                    test_y = np.array(train_label)[test_index]
        
    
                    train_x = np.reshape(train_x, (np.array(train_x).shape[0], 1,m,n))
                    test_x = np.reshape(test_x, (np.array(test_x).shape[0], 1, m,n))
        
                    train_x = torch.tensor(train_x, dtype=torch.float32)
                    train_y = torch.tensor(train_y, dtype=torch.int64)
                    test_x = torch.tensor(test_x, dtype=torch.float32)
                    test_y = torch.tensor(test_y, dtype=torch.int64)
                    
                    # using GPU...
                    if torch.cuda.is_available():
                        train_x = train_x.cuda()
                        train_y = train_y.cuda()
                        test_x = test_x.cuda()
                        test_y = test_y.cuda()
                    
                    train_result, test_result, best_acc = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'],  parameters['batch_size'], train_x, train_y, test_x, test_y, False, best_acc1)
                    train_results.append(train_result)
                    test_results.append(test_result)
                T_results = np.array(train_results).mean(axis=0)
                V_results = np.array(test_results).mean(axis=0)
                acac.append(V_results[0])
                print("##########################################################################")
                print("After 5 folders, the results are:")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f\tAUC %.3f' % (T_results[0], T_results[1], T_results[2], T_results[3], T_results[4], T_results[5]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f\tAUC %.3f' % (V_results[0], V_results[1], V_results[2], V_results[3], V_results[4],V_results[5]))
                if method == 'AlexNet':
                    model = AlexNet(models.alexnet(pretrained=False),m=k)
                elif method == 'SqueezeNet':
                    model = SqueezeNet(models.squeezenet1_0(pretrained=True), prior=False,m = k)
                elif method == 'VGG':
                    model = VGG(models.vgg16(pretrained=True),m=k)
                val_x = np.reshape(test_feature, (np.array(test_feature).shape[0],1, m,n))
                val_x = torch.tensor(val_x, dtype=torch.float32)
                val_y = torch.tensor(test_label, dtype=torch.int64)
                model.load_state_dict(torch.load('cnn_no_pr.pkl'))
                model.eval()
                val_scores = model(val_x)
                prediction = predictions_from_output(val_scores)
                pred_prob = calculate_prob(val_scores)
                fpr_cnn, tpr_cnn, _ = roc_curve(val_y.cpu().detach().numpy(), pred_prob.cpu().detach().numpy())
                AUC = auc(fpr_cnn, tpr_cnn)
                prediction = prediction.view_as(val_y)
                precision, recall, fscore, mcc, val_acc = evaluate(val_y, prediction)
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f\tAUC %.3f' % (val_acc, precision, recall, fscore, mcc,AUC))
                vv = np.array([val_acc, precision, recall, fscore, mcc,AUC])
                #vacc.append(T_results)
                vacc.append(V_results)
                vacc.append(vv)
                print(acac)
            return np.array(vacc)



if __name__ == "__main__":
    features ='ProtVec + ctd'
    mode = 'TL'
    main(mode, features)


































