import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, GRU, Dropout
from keras import backend as K
import time
import sounddevice as sd
from matplotlib import pyplot as plt

# Load and Prep Data ###########################################################
a = np.load('/home/kac487/repo/nnAvacadoz/trainingDat/kyleData.npy')
b = np.load('/home/kac487/repo/nnAvacadoz/trainingDat/richData.npy')
c = np.load('/home/kac487/repo/nnAvacadoz/trainingDat/ambikaData.npy')
kyle = 0
rich = 1
ambika = 2

a_labels = np.full(a.shape[0], kyle)
b_labels = np.full(b.shape[0], rich)
c_labels = np.full(c.shape[0], ambika)

# Append labels to last column of data matrices
a = np.c_[a, a_labels]
b = np.c_[b, b_labels]
c = np.c_[c, c_labels]

dataSet = np.concatenate((a,b,c))
np.random.shuffle(dataSet)
dataSet.shape
data = dataSet[:, :-1] + 1
labels = dataSet[:, -1]


idx = 55
plt.plot(data[idx,:])
plt.show()
sd.play(data[idx-4,:]/64,48000)
# for i in range(x_train.shape[0]):
    # sd.play(x_train[i,:]/64,48000)
    # time.sleep(1.5)
    # print('Sample %d' % i)
input('wait here')


cutoff = int(len(data) * 0.8)
x_train = data[:cutoff]
y_train = labels[:cutoff]
x_test = data[cutoff:]
y_test = labels[cutoff:]

input_shape = (x_train.shape[1],1)
batch_size = 128
n_classes = 3
epochs = 4

if K.image_data_format() == 'channels_first':
    x_train = np.expand_dims(x_train, axis=0)
    x_test = np.expand_dims(x_test, axis=0)
else:
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)

del(a,b,c,dataSet,data,labels)

y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)


# Define Model #################################################################
input_shape
x_train.shape
y_train.shape
x_test.shape
y_test.shape
mdl = Sequential()
mdl.add(Conv1D(32,(8),activation='relu',input_shape=input_shape))
mdl.add(MaxPooling1D(pool_size=(40)))
# mdl.add(Conv1D(64,(6),activation='relu',input_shape=input_shape))
# mdl.add(MaxPooling1D(pool_size=(40)))
mdl.add(Dropout(0.25))
mdl.add(Conv1D(128,(4),activation='relu',input_shape=input_shape))
mdl.add(MaxPooling1D(pool_size=(20)))
mdl.add(GRU(units=512))
# mdl.add(Flatten())
# mdl.add(Dense(128,activation='relu'))
mdl.add(Dense(n_classes,activation='softmax'))
mdl.summary()



#tensorboard = keras.callbacks.TensorBoard(log_dir='../logs',
#    histogram_freq=0,
#    batch_size=batch_size,
#    write_graph=True,
#    write_grads=False,
#    write_images=True)

mdl.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

mdl.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test)#,
          )#callbacks=[tensorboard])
