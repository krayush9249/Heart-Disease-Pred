
import pandas as pd
filepath = '/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/final_data.csv'
data = pd.read_csv(filepath)

data.shape

X = data.drop('target', axis=1)
y = data['target']

from sklearn.model_selection import train_test_split
train_ratio = 0.70
test_ratio = 0.25
val_ratio = 0.05

X_train, x_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=1-train_ratio, shuffle=True, stratify=y, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(x_test, y_test, 
                                                test_size=test_ratio/(test_ratio + val_ratio), shuffle=True, stratify=y_test, random_state=42) 

print(X_train.shape[0], X_test.shape[0], X_val.shape[0])

train = pd.concat([X_train, y_train], axis=1)
train.head(3)

test = pd.concat([X_test, y_test], axis=1)
test.head(3)

valid = pd.concat([X_val, y_val], axis=1)
valid.head(3)

train.to_csv('/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/train.csv', index=False)
test.to_csv('/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/test.csv', index=False)
valid.to_csv('/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/val.csv', index=False)

df = pd.concat([train, valid, test], axis=0)
df.shape[0]

