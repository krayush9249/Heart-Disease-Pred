### LIME - Local Interpretable Model-Agnostic Explanations

import base
import numpy as np
import pandas as pd 
import tensorflow as tf
import lime


with open(r'/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Model/ann_model.json','r') as json_file:
    loaded_model_json = json_file.read()

loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.load_weights(r'/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Model/ann_model.h5')


train = pd.read_csv(r'/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/final_data_2.csv')
val = pd.read_csv(r'/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/val2.csv')

X_train = train.drop(columns=['DISEASE'], axis=1)
X_val = val.drop(columns=['DISEASE'], axis=1)

X_scaled = base.standardize(X_train)


interpretor = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_scaled),
    feature_names=X_train.columns.values,
    class_names=['No', 'Yes'],
    mode='classification',
    verbose=True)


exp = interpretor.explain_instance(
    data_row=X_val.iloc[0].values,
    predict_fn=loaded_model.predict,
    labels=(0,))  # Specify the label index as 0


ls = exp.as_list(label=0)  # Use label 0 here as well
feat_imp = pd.DataFrame(ls, columns=['Feature', 'Explanation'])
print("Feature Importance\n", feat_imp)


plt = exp.as_pyplot_figure(label=0) # Specify label 0 here as well
plt.tight_layout()
plt.show()

exp.save_to_file(r'/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Model/exp.html',
                 labels=(0,))

