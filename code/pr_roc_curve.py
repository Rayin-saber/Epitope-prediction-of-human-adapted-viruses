#cross-validation train + independent test data + external validation data to draw roc curve
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from models import train_test_split_data
def draw_roc_syn_test():
  
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
                            monotone_constraints=None,
                            n_estimators=180, 
                            n_jobs=None, 
                            num_parallel_tree=1,
                            random_state=283, 
                            reg_alpha=0.05, 
                            reg_lambda=1, 
                            scale_pos_weight=1,
                            subsample=0.75, 
                            validate_parameters=True,
                            verbosity=1)
    test_order = pd.read_csv('test_order.csv')
    test_order = np.array(test_order)
    train_order = pd.read_csv('train_order.csv')
    train_order = np.array(train_order)
    ddd = pd.read_csv('virus_realddd.csv')
   
    test_label = np.array(ddd['Label'])[test_order]
    train_label = np.array(ddd['Label'])[train_order]

    plt.figure(figsize=(8,8))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', alpha=.8)
    
    #plot the roc curve for synthetic test set

    for name in ['DPC','AAP15','CTD','BepiPred2', 'LBtope','Accessibility','Flexibility',
                 'Beta-Turn', 'Antigenicity', 'Hydrophilicity', 'ProtVec', 'ProtVec + CTD']:
        if name == 'aac':
            train_data = pd.read_csv('aac.csv')
            train_x = np.array(train_data)[train_order].reshape((len(train_order),train_data.shape[1]))
            test_x = np.array(train_data)[test_order].reshape((len(test_order),train_data.shape[1]))
            probas_syn = clf.fit(train_x, train_label).predict_proba(test_x)
            fpr, tpr, thresholds = roc_curve(test_label, probas_syn[:, 1])
        
        elif name == 'AAP15':
            train_data = np.array(pd.read_csv('aap15.csv'))
            train_label1 = np.array(pd.read_csv('aap15_label.csv'))
            train_x, test_x, train_y, test_y = train_test_split_data(train_data, train_label1, 0.1, shuffled_flag=True)
            probas_syn = clf.fit(train_x, train_y).predict_proba(test_x)
            fpr, tpr, thresholds = roc_curve(test_y, probas_syn[:, 1])
            
        elif name == 'AAP20':
            train_data = np.array(pd.read_csv('aap20.csv'))
            train_label1 = np.array(pd.read_csv('aap20_label.csv'))
            train_x, test_x, train_y, test_y = train_test_split_data(train_data, train_label1, 0.1, shuffled_flag=True)
            probas_syn = clf.fit(train_x, train_y).predict_proba(test_x)
            fpr, tpr, thresholds = roc_curve(test_y, probas_syn[:, 1])
            
        elif name == 'CTD':
            train_data = pd.read_csv('feature.csv')
            train_x = np.array(train_data)[train_order].reshape((len(train_order),train_data.shape[1]))
            test_x = np.array(train_data)[test_order].reshape((len(test_order),train_data.shape[1]))
            probas_syn = clf.fit(train_x, train_label).predict_proba(test_x)
            fpr, tpr, thresholds = roc_curve(test_label, probas_syn[:, 1])
            
        elif name == 'DPC':
                train_data = pd.read_csv('dpc.csv')
                train_x = np.array(train_data)[train_order].reshape((len(train_order),train_data.shape[1]))
                test_x = np.array(train_data)[test_order].reshape((len(test_order),train_data.shape[1]))
                probas_syn = clf.fit(train_x, train_label).predict_proba(test_x)
                fpr, tpr, thresholds = roc_curve(test_label, probas_syn[:, 1])
                
        elif name == 'BepiPred2':
            test_label = pd.read_csv('BepiPred2.csv')['real_y']
            predicted_y = pd.read_csv('BepiPred2.csv')['predicted']
            fpr, tpr, thresholds = roc_curve(test_label, predicted_y)
            
        elif name =='LBtope':
            pred_label = pd.read_csv('lbtope.csv')['result']
            for i in range(len(pred_label)):
                pred_label[i] = pred_label[i]/100
                if pred_label[i]<0.6:
                    pred_label[i] = pred_label[i]*0.5/0.6
                else:
                    pred_label[i] = pred_label[i]*1.25-0.25
            fpr, tpr, thresholds = roc_curve(np.array(ddd['Label']), pred_label)
        
        elif name =='Accessibility':
            pred_label = pd.read_csv('Emini_result.csv')['Emini']
            real_label = pd.read_csv('Emini_result.csv')['Label']
            fpr, tpr, thresholds = roc_curve(real_label, pred_label)
        
        elif name =='Flexibility':
            pred_label = pd.read_csv('Karplus_result.csv')['Karplus']
            real_label = pd.read_csv('Karplus_result.csv')['Label']
            fpr, tpr, thresholds = roc_curve(real_label, pred_label)
            
        elif name =='Beta-Turn':
                pred_label = pd.read_csv('Chou_result.csv')['Chou']
                real_label = pd.read_csv('Chou_result.csv')['Label']
                fpr, tpr, thresholds = roc_curve(real_label, pred_label)
                
        elif name =='Antigenicity':
                    pred_label = pd.read_csv('Kolaskar_result.csv')['Kolaskar']
                    real_label = pd.read_csv('Kolaskar_result.csv')['Label']
                    fpr, tpr, thresholds = roc_curve(real_label, pred_label)
        
        elif name =='Hydrophilicity':
            pred_label = pd.read_csv('Parker_result.csv')['Parker']
            real_label = pd.read_csv('Parker_result.csv')['Label']
            fpr, tpr, thresholds = roc_curve(real_label, pred_label)
            
        elif name == 'ProtVec':
            features_numpy0 = pd.read_csv('x_train.csv')
            train_feature = np.array(features_numpy0)
            targets_numpy = pd.read_csv('y_train.csv')
            train_label = np.array(targets_numpy)
            features_numpy0 = pd.read_csv('x_test.csv')
            test_feature = np.array(features_numpy0)
            targets_numpy = pd.read_csv('y_test.csv')
            test_label = np.array(targets_numpy)
            probas_syn = clf.fit(train_feature, train_label).predict_proba(test_feature)
            fpr, tpr, thresholds = roc_curve(test_label, probas_syn[:, 1])
            
        elif name == 'ProtVec + CTD':
            features_numpy0 = pd.read_csv('x_train1.csv')
            train_feature = np.array(features_numpy0)
            targets_numpy = pd.read_csv('y_train.csv')
            train_label = np.array(targets_numpy)
            features_numpy0 = pd.read_csv('x_test1.csv')
            test_feature = np.array(features_numpy0)
            targets_numpy = pd.read_csv('y_test.csv')
            test_label = np.array(targets_numpy)
            probas_syn = clf.fit(train_feature, train_label).predict_proba(test_feature)
            fpr, tpr, thresholds = roc_curve(test_label, probas_syn[:, 1])
        
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=  name + ' ' + r'(AUC = %0.3f)' % (roc_auc), lw=2, alpha=.8)
    
    
    plt.xlim([-0.00, 1.05])
    plt.ylim([-0.00, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right",  prop={'size': 8})
    plt.savefig('roc.eps', dpi=300)

def draw_pr_curve():    
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
                            monotone_constraints=None,
                            n_estimators=180, 
                            n_jobs=None, 
                            num_parallel_tree=1,
                            random_state=283, 
                            reg_alpha=0.05, 
                            reg_lambda=1, 
                            scale_pos_weight=1,
                            subsample=0.75, 
                            validate_parameters=True,
                            verbosity=1)
    test_order = pd.read_csv('test_order.csv')
    test_order = np.array(test_order)
    train_order = pd.read_csv('train_order.csv')
    train_order = np.array(train_order)
    ddd = pd.read_csv('virus_realddd.csv')
   
    test_label = np.array(ddd['Label'])[test_order]
    train_label = np.array(ddd['Label'])[train_order]
    
    plt.figure(figsize=(8,8))
    plt.plot(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=1, y1=0)
    for name in ['DPC','AAP15','CTD','BepiPred2', 'LBtope','Accessibility','Flexibility',
                 'Beta-Turn', 'Antigenicity', 'Hydrophilicity', 'ProtVec', 'ProtVec + CTD']:
        print(name)
        if name == 'aac':
            train_data = pd.read_csv('aac.csv')
            train_x = np.array(train_data)[train_order].reshape((len(train_order),train_data.shape[1]))
            test_x = np.array(train_data)[test_order].reshape((len(test_order),train_data.shape[1]))
            probas_syn = clf.fit(train_x, train_label).predict_proba(test_x)
            precision, recall, thresholds = precision_recall_curve(test_label, probas_syn[:, 1])
        
        elif name == 'aap15':
            train_data = np.array(pd.read_csv('aap15.csv'))
            train_label1 = np.array(pd.read_csv('aap15_label.csv'))
            train_x, test_x, train_y, test_y = train_test_split_data(train_data, train_label1, 0.1, shuffled_flag=True)
            probas_syn = clf.fit(train_x, train_y).predict_proba(test_x)
            precision, recall, thresholds = precision_recall_curve(test_y, probas_syn[:, 1])
            
        elif name == 'AAP20':
            train_data = np.array(pd.read_csv('aap20.csv'))
            train_label1 = np.array(pd.read_csv('aap20_label.csv'))
            train_x, test_x, train_y, test_y = train_test_split_data(train_data, train_label1, 0.1, shuffled_flag=True)
            probas_syn = clf.fit(train_x, train_y).predict_proba(test_x)
            precision, recall, thresholds = precision_recall_curve(test_y, probas_syn[:, 1])
            
        elif name == 'CTD':
            train_data = pd.read_csv('feature.csv')
            train_x = np.array(train_data)[train_order].reshape((len(train_order),train_data.shape[1]))
            test_x = np.array(train_data)[test_order].reshape((len(test_order),train_data.shape[1]))
            probas_syn = clf.fit(train_x, train_label).predict_proba(test_x)
            precision, recall, thresholds = precision_recall_curve(test_label, probas_syn[:, 1])
            
        elif name == 'DPC':
                train_data = pd.read_csv('dpc.csv')
                train_x = np.array(train_data)[train_order].reshape((len(train_order),train_data.shape[1]))
                test_x = np.array(train_data)[test_order].reshape((len(test_order),train_data.shape[1]))
                probas_syn = clf.fit(train_x, train_label).predict_proba(test_x)
                precision, recall, thresholds = precision_recall_curve(test_label, probas_syn[:, 1])
                
        elif name == 'BepiPred2':
            test_label = pd.read_csv('BepiPred2.csv')['real_y']
            predicted_y = pd.read_csv('BepiPred2.csv')['predicted']
            precision, recall, thresholds = precision_recall_curve(test_label, predicted_y)
            
        elif name =='Accessibility':
            pred_label = pd.read_csv('Emini_result.csv')['Emini']
            real_label = pd.read_csv('Emini_result.csv')['Label']
            precision, recall, thresholds = precision_recall_curve(real_label, pred_label)
        
        elif name =='Flexibility':
            pred_label = pd.read_csv('Karplus_result.csv')['Karplus']
            real_label = pd.read_csv('Karplus_result.csv')['Label']
            precision, recall, thresholds = precision_recall_curve(real_label, pred_label)
            
        elif name =='Beta-Turn':
                pred_label = pd.read_csv('Chou_result.csv')['Chou']
                real_label = pd.read_csv('Chou_result.csv')['Label']
                precision, recall, thresholds = precision_recall_curve(real_label, pred_label)
                
        elif name =='Antigenicity':
                    pred_label = pd.read_csv('Kolaskar_result.csv')['Kolaskar']
                    real_label = pd.read_csv('Kolaskar_result.csv')['Label']
                    precision, recall, thresholds = precision_recall_curve(real_label, pred_label)
        
        elif name =='Hydrophilicity':
            pred_label = pd.read_csv('Parker_result.csv')['Parker']
            real_label = pd.read_csv('Parker_result.csv')['Label']
            precision, recall, thresholds = precision_recall_curve(real_label, pred_label)
            
        elif name == 'ProtVec':
            features_numpy0 = pd.read_csv('x_train.csv')
            train_feature = np.array(features_numpy0)
            targets_numpy = pd.read_csv('y_train.csv')
            train_label = np.array(targets_numpy)
            features_numpy0 = pd.read_csv('x_test.csv')
            test_feature = np.array(features_numpy0)
            targets_numpy = pd.read_csv('y_test.csv')
            test_label = np.array(targets_numpy)
            probas_syn = clf.fit(train_feature, train_label).predict_proba(test_feature)
            precision, recall, thresholds = precision_recall_curve(test_label, probas_syn[:, 1])
            
        elif name == 'ProtVec + CTD':
            features_numpy0 = pd.read_csv('x_train1.csv')
            train_feature = np.array(features_numpy0)
            targets_numpy = pd.read_csv('y_train.csv')
            train_label = np.array(targets_numpy)
            features_numpy0 = pd.read_csv('x_test1.csv')
            test_feature = np.array(features_numpy0)
            targets_numpy = pd.read_csv('y_test.csv')
            test_label = np.array(targets_numpy)
            probas_syn = clf.fit(train_feature, train_label).predict_proba(test_feature)
            precision, recall, thresholds = precision_recall_curve(test_label, probas_syn[:, 1])
        
        elif name =='LBtope':
            pred_label = pd.read_csv('lbtope.csv')['result']
            precision, recall, thresholds = precision_recall_curve(np.array(ddd['Label']), pred_label)
            
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, label=  name + ' ' + r'(PRAUC = %0.3f)' % (pr_auc), lw=2, alpha=.8)

    plt.xlim([-0.00, 1.05])
    plt.ylim([-0.00, 1.05])
    plt.title('PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower right',  prop={'size':8})
    plt.savefig('pr.eps', dpi=300)