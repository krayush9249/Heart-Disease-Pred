
import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import base

# Loading the train & test data -
train = pd.read_csv(r'/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/train2.csv')
test = pd.read_csv(r'/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/test2.csv')

# Splitting the data into independent & dependent variables -
X_train, y_train =  base.splitter(train, y_var='DISEASE')
X_test, y_test =  base.splitter(test, y_var='DISEASE')

y = np.concatenate([y_train, y_test], axis=0)

# Standardizing the data -
X_scaled, X_train_scaled, X_test_scaled = base.standardizer(X_train, X_test)

# Training the model -

from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

# XGBoost Classifier 
xgb_params = {}
xgb = XGBClassifier(**xgb_params) 

# AdaBoost Classifier 
ada_params = {}
ada = AdaBoostClassifier(**ada_params) 

# Gradient Boosting Classifier
gbc_params = {}
gbc = GradientBoostingClassifier(**gbc_params)

# Soft Voting Classifier 

clfs = [('XGBoost', xgb), ('AdaBoost', ada), ('GradientBoosting', gbc)]
vc = VotingClassifier(estimators=clfs, voting='soft')


model = base.model_train(vc, X_train_scaled, y_train)

# Checking the model's performance -
y_pred, y_pred_proba = base.model_eval(model, X_train_scaled, X_test_scaled, y_train, y_test)
base.cross_val(model, X_scaled, y, scoring='recall')


base.show_pred(y_pred, y_pred_proba)


base.precision_recall_curve_plot(model, X_test_scaled, y_test)

base.roc_auc_curve_plot(model, X_test_scaled, y_test)
