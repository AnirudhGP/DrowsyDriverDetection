from __future__ import absolute_import
from __future__ import print_function
import numpy as np

np.random.seed(1337)  # for reproducibility
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop, Adam

from sklearn import svm

from six.moves import cPickle as pickle

pickle_files = ['yawn_mouths.pickle']
i = 0
for pickle_file in pickle_files:
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        if i == 0:
            train_dataset = save['train_dataset']
            train_labels = save['train_labels']
            test_dataset = save['test_dataset']
            test_labels = save['test_labels']
        else:
            print("here")
            train_dataset = np.concatenate((train_dataset, save['train_dataset']))
            train_labels = np.concatenate((train_labels, save['train_labels']))
            test_dataset = np.concatenate((test_dataset, save['test_dataset']))
            test_labels = np.concatenate((test_labels, save['test_labels']))
        del save  # hint to help gc free up memory
    i += 1

print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

batch_size = 1
nb_classes = 1
epochs = 20

X_train = train_dataset
X_train = X_train.reshape((X_train.shape[0], X_train.shape[3]) + X_train.shape[1:3])
# X_train = X_train.reshape(X_train.shape[0:3])
# X_train = X_train.reshape(len(X_train), -1)
Y_train = train_labels

X_test = test_dataset
X_test = X_test.reshape((X_test.shape[0], X_test.shape[3]) + X_test.shape[1:3])
# X_test = X_test.reshape(X_test.shape[0:3])
# X_test = X_test.reshape(len(X_test), -1)
Y_test = test_labels

# input image dimensions
_, img_channels, img_rows, img_cols = X_train.shape

# convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(32, (3, 3), padding='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))

model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0005), metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=[X_test, Y_test])
# model.save('yawnModel.h5')

score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

'''cls = svm.LinearSVC()
cls.fit(X_train, Y_train)
score = cls.score(X_test, Y_test)'''
