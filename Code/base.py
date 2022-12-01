
def splitter(data, y_var):

   # Splitting the data into dependent & independent variables -
    X = data.drop(columns=y_var, axis=1).values
    y = data[y_var].values

    return X, y

from sklearn.preprocessing import StandardScaler

def standardizer(X_train, X_test):
    
    # Standardizing the data -
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # X_scaled = np.concatenate([X_train_scaled, X_test_scaled,], axis=0)
    return X_train_scaled, X_test_scaled

def model_train(model_obj, X_train, y_train, **kwargs): 

    model_obj.fit(X_train, y_train, **kwargs)
    return model_obj

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

def model_eval(model_obj, X_train, X_test, y_train, y_test):

    y_pred_test = model_obj.predict(X_test)
    y_pred_test_proba = model_obj.predict_proba(X_test)[:, 1]
    print("Train accuracy: {:.2f}%".format(accuracy_score(y_train, model_obj.predict(X_train)) * 100))
    print("Test accuracy: {:.2f}%".format(accuracy_score(y_test, model_obj.predict(X_test)) * 100))
    print("F1 Score: {:.2f}".format(f1_score(y_test, y_pred_test)))
    print("Precision: {:.2f}".format(precision_score(y_test, y_pred_test)))
    print("Recall: {:.2f}".format(recall_score(y_test, y_pred_test)))
    print("ROC AUC Score: {:.2f}".format(roc_auc_score(y_test, y_pred_test_proba)))
    conmat = confusion_matrix(y_test, y_pred_test)
    tp = conmat[0][0]
    fp = conmat[0][1]
    fn = conmat[1][0]
    tn = conmat[1][1]
    tpr = tp/(tp+fn)
    tnr = tn/(fp+tn)
    fpr = fp/(tp+fn)
    fnr = fn/(fp+tn)
    print("Type 1 Error: {:.2f}".format(fpr))
    print("Type 2 Error: {:.2f}".format(fnr))
    print("Sensitivity: {:.2f}".format(tpr))
    print("Specificity: {:.2f}\n".format(1-fpr))