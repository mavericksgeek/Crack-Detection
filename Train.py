from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils import plot_model
from pylab import *

import matplotlib.pyplot as plt

def on_epoch_end(self, epoch, logs=None):
    print(eval(self.model.optimizer.lr))

# 4. Load data into train and test sets
mat = scipy.io.loadmat('train_X.mat')
X_train = mat['train_X']
mat = scipy.io.loadmat('test_X.mat')
X_test = mat['test_X']

mat = scipy.io.loadmat('train_Y.mat')
y_train = mat['train_Y']
mat = scipy.io.loadmat('test_Y.mat')
y_test = mat['test_Y']

# 5. Preprocess input data
X_train = X_train.reshape(15128, 1, 64, 64)
X_test = X_test.reshape(6580, 1, 64, 64)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# 6. Preprocess class labels
Y_train = np_utils.to_categorical(y_train, 2)
Y_test = np_utils.to_categorical(y_test, 2)

# 7. Define model architecture
model = Sequential()

model.add(Convolution2D(64, 5, 5, activation='tanh', input_shape=(1,64,64),
                        dim_ordering='th'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Convolution2D(64, 5, 5, activation='tanh'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 9. Fit model on training data
history = model.fit(X_train, Y_train,
          batch_size=64, nb_epoch=1, verbose=1)
y_prob = model.predict(X_test)
print(y_prob)

scipy.io.savemat('y_prob.mat',y_prob)

grid = np.array(y_prob, dtype='f')
print(grid[1:5,0:3])

#print(model.history.params)
# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
#score
#plot_model(model, to_file='model.png')
#plot(score)
print(history.history.keys())
print(score)

#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
# summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
