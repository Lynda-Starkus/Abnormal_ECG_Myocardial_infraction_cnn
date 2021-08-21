import pandas as pd
import numpy as np
import tensorflow

from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

normal_data = pd.read_csv("ptbdb_normal.csv", header=None)
abnormal_data = pd.read_csv("ptbdb_abnormal.csv", header=None)
dataset = pd.concat([normal_data, abnormal_data])

dataset_train, dataset_test = train_test_split(dataset, test_size=0.2, random_state=1337, stratify=dataset[187])


Y = np.array(dataset_train[187].values).astype(np.int8)
X = np.array(dataset_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(dataset_test[187].values).astype(np.int8)
X_test = np.array(dataset_test[list(range(187))].values)[..., np.newaxis]


def get_model():
    nclass = 1
    inp = Input(shape=(187, 1))
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(inp)
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(nclass, activation=activations.sigmoid, name="dense_3_ptbdb")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = tensorflow.keras.optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
    model.summary()
    return model

model = get_model()
path = "ptbdb.h5"
model_checkpoint = ModelCheckpoint(path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
reduced = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
callbacks_list = [model_checkpoint, early_stopping, reduced]  # early_stopping

model.fit(X, Y, epochs=1000, verbose=2, callbacks=callbacks_list, validation_split=0.1)
model.load_weights(path)

predicted = model.predict(X_test)
predicted = (predicted>0.5).astype(np.int8)

resultats = open("résultats_obtenus.txt", "w")

for row in predicted :

    np.savetxt(resultats, (row, ecg))

resultats.close()


f1 = f1_score(Y_test, predicted)

print("Test f1 score : %s "% f1)

acc = accuracy_score(Y_test, predicted)

print("Test accuracy score : %s "% acc)

