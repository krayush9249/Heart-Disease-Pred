
# Importing required libraries -
import pandas as pd
import pickle
import base
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier


# Function to save our model -
def model_dump(model_obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model_obj, f) 


# Loading the training dataset -
train = pd.read_csv(r'..\Files\train.csv')

X_train, y_train = base.splitter(train)

X_scaled, X_train_scaled, X_test_scaled = base.standardizer(X_train, y_train)

if __name__ == '__main__':
    
    # XGBoost Classifier -

    # Final parameters -
    
    xgb_params = {'base_score': 0.5,
                  'colsample_bytree': 0.6,
                  'gamma': 0.1,
                  'learning_rate': 0.15,
                  'max_depth': 9,
                  'min_child_weight': 3,
                  'n_estimators': 100,
                  'reg_alpha': 0.1,
                  'reg_lambda': 10,
                  'scale_pos_weight': 5,
                  'subsample': 0.5}
    xgb = XGBClassifier(**xgb_params) 

    # CatBoost Classifier -

    # Final parameters -
    cat_params = {'iterations': 10,
                  'learning_rate': 0.65,
                  'depth': 8,
                  'loss_function': 'Logloss',
                  'eval_metric': 'Recall',
                  'subsample': 0.6,
                  'l2_leaf_reg': 0.001,
                  'scale_pos_weight': 2}
    cat = CatBoostClassifier(**cat_params) 

    # Soft Voting Classifier -

    clfs = [('XGBoost', xgb), ('CatBoost', cat)]
    vc = VotingClassifier(estimators=clfs, voting='soft')

    model_obj = base.model_train(vc, X_train_scaled, y_train)

    model_file = r'VotingClassifier.pkl'
    model_dump(model_obj, model_file)