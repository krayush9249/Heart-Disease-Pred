
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base
import tensorflow as tf
from tensorflow import keras

# Loading the train & test data -
train = pd.read_csv('/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/train.csv')
test = pd.read_csv('/Users/kumarpersonal/Downloads/Heart-Disease-Pred/Data/test.csv')

# Splitting the data into independent & dependent variables -
X_train, y_train =  base.splitter(train, y_var='target')
X_test, y_test =  base.splitter(test, y_var='target')

# Standardizing the data -
X_scaled, X_train_scaled, X_test_scaled = base.standardizer(X_train, X_test)

model = keras.Sequential([
                          keras.layers.Dense(units=128, input_shape=(11,), activation='relu'),
                          keras.layers.BatchNormalization(axis=1),
                          keras.layers.Dense(units=64, activation='relu'),
                          keras.layers.BatchNormalization(axis=1),
                          keras.layers.Dense(units=32, activation='relu'),
                          keras.layers.BatchNormalization(axis=1),
                          keras.layers.Dense(units=16, activation='relu'),
                          keras.layers.BatchNormalization(axis=1),
                          keras.layers.Dense(units=1, activation='sigmoid')
                         ])

adam=keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=1e-3, patience=10, mode='max', verbose=0)
mc = tf.keras.callbacks.ModelCheckpoint(filepath='model.h5', save_best_only=True, save_weights_only=True)

hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                 epochs=100, batch_size=32, callbacks=[es, mc], verbose=1)

_, train_acc = model.evaluate(X_train, y_train, batch_size=32, verbose=0)
_, test_acc = model.evaluate(X_test, y_test, batch_size=32, verbose=0)

print('Train Accuracy: {:.3f}'.format(train_acc))
print('Test Accuracy: {:.3f}'.format(test_acc))


y_pred = model.predict(X_test, batch_size=32, verbose=0)
y_pred_class = np.argmax(y_pred, axis=-1)


from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


plt.figure(figsize=(10, 5))
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='test')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(hist.history['accuracy'], label='train')
plt.plot(hist.history['val_accuracy'], label='test')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_class)

plt.figure(figsize = (10, 7))
sns.heatmap(cm, annot=True, fmt='d', cbar=False, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.show()


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_class))

