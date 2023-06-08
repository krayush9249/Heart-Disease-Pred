
import pandas as pd
filepath = r'/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/final_data_2.csv'
data = pd.read_csv(filepath)

data.shape

X = data.drop('DISEASE', axis=1)
y = data['DISEASE']

from sklearn.model_selection import train_test_split
train_ratio = 0.75
test_ratio = 0.24
val_ratio = 0.01

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

train.to_csv(r'/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/train2.csv', index=False)
test.to_csv(r'/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/test2.csv', index=False)
valid.to_csv(r'/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/val2.csv', index=False)

df = pd.concat([train, valid, test], axis=0)
df.shape[0]

