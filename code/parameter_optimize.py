import numpy as np
import pandas as pd
import warnings

from models import setup_seed
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import xgboost as xgb

warnings.filterwarnings("ignore")

features_numpy0 = pd.read_csv('x_train1.csv', header=None)
train_feature = np.array(features_numpy0)
targets_numpy = pd.read_csv('y_train.csv', header=None)
train_label = np.array(targets_numpy)
features_numpy0 = pd.read_csv('x_test1.csv', header=None)
test_feature = np.array(features_numpy0)
targets_numpy = pd.read_csv('y_test.csv', header=None)
test_label = np.array(targets_numpy)
################################################################################################################################################
#parameter optimization for xgboost
setup_seed(2)
clf = xgb.XGBClassifier(base_score=0.5,
                                booster=None, 
                                colsample_bylevel=1,
                                colsample_bynode=1, 
                                colsample_bytree=0.85, 
                                gamma=0, 
                                gpu_id=None,
                                importance_type='gain', 
                                interaction_constraints=None,
                                learning_rate=0.05, 
                                max_delta_step=0, 
                                max_depth=9,
                                min_child_weight=1, 
                                #missing=nan, 
                                monotone_constraints=None,
                                n_estimators=180, 
                                n_jobs=None, 
                                num_parallel_tree=1,
                                random_state=15, 
                                reg_alpha=0.25, 
                                reg_lambda=1, 
                                scale_pos_weight=1,
                                subsample=0.8, 
                                #tree_method=auto, 
                                validate_parameters=True,
                                verbosity=1)
warnings.filterwarnings("ignore")
kfold = StratifiedKFold(n_splits=5)
accuracy = cross_val_score(clf, train_feature, train_label, cv=kfold, scoring='accuracy')
print(accuracy)
print(accuracy.mean())

y_pred = clf.fit(train_feature, train_label).predict(test_feature)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(test_label, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#parameter test 1 ('n_estimators': 150)
param_test1 = {
 'n_estimators':list(range(0,201,10)),
}
gsearch1 = GridSearchCV(estimator=clf,param_grid = param_test1, scoring='accuracy',n_jobs=4, cv=5)
gsearch1.fit(train_feature, train_label)
gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_


#parameter test 2 ('max_depth': 9 and 'min_child_weight': 1)
param_test2 = {
 'max_depth':list(range(3,10,1)),
 'min_child_weight':list(range(1,6,1))
}
gsearch2 = GridSearchCV(estimator=clf,param_grid = param_test2, scoring='accuracy',n_jobs=4, cv=5)
gsearch2.fit(train_feature, train_label)
gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_


#parameter test 3 ('gamma': 0)
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator=clf, 
 param_grid = param_test3, scoring='accuracy',n_jobs=4, cv=5)
gsearch3.fit(train_feature, train_label)
gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_


#parameter test 4 ('subsample': 0.8, 'colsample_bytree': 0.85)
param_test4 = {
 'subsample':[i/100.0 for i in range(75,90,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}
gsearch4 = GridSearchCV(estimator=clf, 
 param_grid = param_test4, scoring='accuracy',n_jobs=4,cv=5)
gsearch4.fit(train_feature, train_label)
gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_


#parameter test 5 ('reg_alpha': 0.25)
param_test5 = {
 'reg_alpha':[1e-5, 1e-2, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 1, 100]
}
gsearch5 = GridSearchCV(estimator=clf, 
 param_grid = param_test5, scoring='accuracy',n_jobs=4, cv=5)
gsearch5.fit(train_feature, train_label)
gsearch5.cv_results_, gsearch5.best_params_, gsearch5.best_score_

#parameter test 6 ('learning_rate': 0.05)
param_test6 = {
 'learning_rate':[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
}
gsearch6 = GridSearchCV(estimator=clf, 
 param_grid = param_test6, scoring='accuracy',n_jobs=4, cv=5)
gsearch6.fit(train_feature, train_label)
gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_





