
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
from sklearn.ensemble import AdaBoostClassifier

params = {}
clf = AdaBoostClassifier(**params)

model = base.model_train(clf, X_train_scaled, y_train)

# Checking the model's performance -
base.model_eval(model, X_train_scaled, X_test_scaled, y_train, y_test)
base.cross_val(model, X_scaled, y, scoring='recall')

