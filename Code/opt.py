
import pandas as pd
import numpy as np
import base

# Loading the train & test data -
train = pd.read_csv('/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/train.csv')
test = pd.read_csv('/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/test.csv')

# Splitting the data into independent & dependent variables -
X_train, y_train =  base.splitter(train, y_var='target')
X_test, y_test =  base.splitter(test, y_var='target')

# Standardizing the data -
X_scaled, X_train_scaled, X_test_scaled = base.standardizer(X_train, X_test)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=0.1)
logreg.fit(X_train_scaled, y_train)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train_scaled, y_train)

from sklearn.model_selection import GridSearchCV

# Creating the parameter grid -
params = {
          'C' : [0.1, 0.3, 0.5, 0.65, 0.7, 0.85, 1]
         }

n_folds = 5

# Instantiating the grid search -
grid_search = GridSearchCV(estimator=logreg, param_grid=params, 
                          cv=n_folds, verbose=1, n_jobs=-1, scoring='f1')

# Fitting the grid search -
grid_search.fit(X_train_scaled, y_train)

print(grid_search.best_score_)

print(grid_search.best_params_)


base.roc_auc_curve_plot(knn, X_test_scaled, y_test)

    
base.precision_recall_curve_plot(knn, X_test_scaled, y_test)


# Let's choose a different threshold value
threshold = 0.65
preds = np.where(knn.predict_proba(X_test_scaled)[:, 1] > threshold, 1, 0)


from sklearn.metrics import classification_report
print(classification_report(y_test, preds))

