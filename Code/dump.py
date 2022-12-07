
# Importing required libraries -
import pandas as pd
import pickle
import base
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier


# Function to save our model -
def model_dump(model_obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model_obj, f) 


# Loading the train & test dataset -
train = pd.read_csv('/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/train.csv')
test = pd.read_csv('/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/test.csv')

# Combining the train & test dataset -
data = pd.concat([train, test], axis=0)

# Splitting the data -
X, y = base.splitter(data)

# Standardizing the data - 
X_scaled = base.standardize(X)


if __name__ == '__main__':
    
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

    clfs = [('XGBoost', xgb), ('AdaBoost', ada), ('GradientBossting', gbc)]
    vc = VotingClassifier(estimators=clfs, voting='soft')

    model_obj = base.model_train(vc, X_scaled, y)

    model_file = '/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Model/VotingClassifier.pkl'
    model_dump(model_obj, model_file)