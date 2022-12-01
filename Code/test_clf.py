
import pandas as pd
import base

# Loading the train & test data -
train = pd.read_csv('/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/train.csv')
test = pd.read_csv('/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/test.csv')

# Splitting the data into independent & dependent variables -
X_train, y_train =  base.splitter(train, y_var='target')
X_test, y_test =  base.splitter(test, y_var='target')

# Standardizing the data -
X_train_scaled, X_test_scaled = base.standardizer(X_train, X_test)

# Training the model -
from sklearn.linear_model import LogisticRegression

params = {}
clf = LogisticRegression(**params)

base.model_train(clf, X_train_scaled, y_train)

# Checking the model's performance -
base.model_eval(clf, X_train_scaled, X_test_scaled, y_train, y_test)