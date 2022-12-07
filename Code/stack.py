
import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import base

# Loading the train & test data -
train = pd.read_csv('/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/train.csv')
test = pd.read_csv('/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/test.csv')

# Splitting the data into independent & dependent variables -
X_train, y_train =  base.splitter(train, y_var='target')
X_test, y_test =  base.splitter(test, y_var='target')

y = np.concatenate([y_train, y_test], axis=0)

# Standardizing the data -
X_scaled, X_train_scaled, X_test_scaled = base.standardizer(X_train, X_test)

# Training the model -

from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier

# XGBoost Classifier 
xgb_params = {}
xgb = XGBClassifier(**xgb_params) 

# AdaBoost Classifier 
ada_params = {}
ada = AdaBoostClassifier(**ada_params) 

# Gradient Boosting Classifier
gbc_params = {}
gbc = GradientBoostingClassifier(**gbc_params)


# Stacking Classifier -
from sklearn.neighbors import KNeighborsClassifier

estimator = [('XGBoost', xgb),
             ('AdaBoost', ada),
             ('GradientBoosting', gbc)]

stack = StackingClassifier(estimators=estimator, 
                           final_estimator=KNeighborsClassifier(n_neighbors=11))

model = base.model_train(stack, X_train_scaled, y_train)

# Checking the model's performance -
base.model_eval(model, X_train_scaled, X_test_scaled, y_train, y_test)
base.cross_val(model, X_scaled, y, scoring='recall')
