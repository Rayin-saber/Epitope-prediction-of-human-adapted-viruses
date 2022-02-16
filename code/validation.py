# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 10:38:17 2021

@author: Xianghe Zhu
"""

import math
import numpy as np
import random
import torch
import torch.nn.functional as F
def get_confusion_matrix(y_true, y_pred):
    """
    Calculates the confusion matrix from given labels and predictions.
    Expects tensors or numpy arrays of same shape.
    """

    ## 2 classes
    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range(y_true.shape[0]):
        if y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
        elif y_true[i] == 1 and y_pred[i] == 1:
            TP += 1

    conf_matrix = [
        [TP, FP],
        [FN, TN]
    ]

    return conf_matrix


def get_accuracy(conf_matrix):
    """
    Calculates accuracy metric from the given confusion matrix.
    """
    TP, FP, FN, TN = conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]
    if TP + FP + FN + TN > 0:
        return (TP + TN) / (TP + FP + FN + TN)
    else:
        return 0



def get_precision(conf_matrix):
    """
    Calculates precision metric from the given confusion matrix.
    """
    TP, FP = conf_matrix[0][0], conf_matrix[0][1]

    if TP + FP > 0:
        return TP / (TP + FP)
    else:
        return 0


def get_recall(conf_matrix):
    """
    Calculates recall metric from the given confusion matrix.
    """
    TP, FN = conf_matrix[0][0], conf_matrix[1][0]

    if TP + FN > 0:
        return TP / (TP + FN)
    else:
        return 0


def get_f1score(conf_matrix):
    """
    Calculates f1-score metric from the given confusion matrix.
    """
    p = get_precision(conf_matrix)
    r = get_recall(conf_matrix)

    if p + r > 0:
        return 2 * p * r / (p + r)
    else:
        return 0


def get_mcc(conf_matrix):
    """
    Calculates Matthew's Correlation Coefficient metric from the given confusion matrix.
    """
    TP, FP, FN, TN = conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]
    if TP + FP > 0 and TP + FN > 0 and TN + FP > 0 and TN + FN > 0:
        return (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    else:
        return 0

def get_AUC(labels,preds,n_bins=100):
    m = sum(labels)
    n = len(labels) - m
    total_case = m * n
    pos = [0 for _ in range(n_bins)]
    neg = [0 for _ in range(n_bins)]
    bin_width = 1.0 / n_bins
    for i in range(len(labels)):
        nth_bin = int(preds[i]/bin_width)
        if labels[i]==1:
            pos[nth_bin] += 1
        else:
            neg[nth_bin] += 1
    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(n_bins):
        satisfied_pair += (pos[i]*accumulated_neg + pos[i]*neg[i]*0.5)
        accumulated_neg += neg[i]
    return satisfied_pair / total_case

def evaluate(Y_real, Y_pred):
    conf_matrix = get_confusion_matrix(Y_real, Y_pred)
    #print(conf_matrix)
    precision = get_precision(conf_matrix)
    recall = get_recall(conf_matrix)
    fscore = get_f1score(conf_matrix)
    mcc = get_mcc(conf_matrix)
    val_acc = get_accuracy(conf_matrix)
    #auc = get_AUC(Y_real, Y_pred,n_bins=100)
    return precision, recall, fscore, mcc, val_acc#,auc

def train_test_split_data(feature, label, split_ratio, shuffled_flag):
    #setup_seed(18)
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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def predictions_from_output(scores):
    """
    Maps logits to class predictions.
    """
    prob = F.softmax(scores, dim=1)
    _, predictions = prob.topk(1)
    return predictions
def calculate_prob(scores):
    """
    Maps logits to class predictions.
    """
    prob = F.softmax(scores, dim=1)
    pred_probe, _ = prob.topk(1)
    return pred_probe

def get_time_string(time):
    """
    Creates a string representation of minutes and seconds from the given time.
    """
    mins = time // 60
    secs = time % 60
    time_string = ''

    if mins < 10:
        time_string += '  '
    elif mins < 100:
        time_string += ' '

    time_string += '%dm ' % mins

    if secs < 10:
        time_string += ' '

    time_string += '%ds' % secs

    return time_string