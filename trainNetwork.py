import numpy as np

# Keras Imports
import keras
from keras.models import Sequential
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint

# `Load` and Prep Data ###########################################################
# Load matrices containing each of the speakers
a = np.load('./data/kyleData.npy')
b = np.load('./data/richData.npy')
c = np.load('./data/ambikaData.npy')

# Assign training lables to each speaker
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

# Create a single large data matrix
dataSet = np.concatenate((a,b,c))
# Shuffle data set
np.random.shuffle(dataSet)

# Shuffled Data Matrix
data = dataSet[:, :-1] + 1

# Shuffled Label Matrix
labels = dataSet[:, -1]

# Segment data into training and test sets 80%/20% split
cutoff = int(len(data) * 0.8)
x_train = data[:cutoff]
y_train = labels[:cutoff]
x_test = data[cutoff:]
y_test = labels[cutoff:]
if K.image_data_format() == 'channels_first':
   x_train = np.expand_dims(x_train, axis=0)
   x_test = np.expand_dims(x_test, axis=0)
else:
   x_train = np.expand_dims(x_train, axis=2)
   x_test = np.expand_dims(x_test, axis=2)

# Clear un-needed variables from memory
del(a,b,c,dataSet,data,labels)

# Define Training Parameters
input_shape = (x_train.shape[1],1)
batch_size = 32
n_classes = 3
epochs = 55

# Save label vectors for later network performance eval.
y_trainP = y_train
y_testP = y_test

# Convert labels to cagetegorical vectors
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)

# Define Model #################################################################

mdl = Sequential()
mdl.add(Conv1D(32,(8),activation='relu',input_shape=input_shape))
mdl.add(MaxPooling1D(pool_size=(40)))

mdl.add(Conv1D(64,(6),activation='relu',input_shape=input_shape))
mdl.add(MaxPooling1D(pool_size=(40)))

mdl.add(Conv1D(128,(3),activation='relu',input_shape=input_shape))
mdl.add(MaxPooling1D(pool_size=(40)))

mdl.add(Flatten())
mdl.add(Dense(64,activation='relu'))
mdl.add(Dense(n_classes,activation='softmax'))

mdl.summary()

# Compile Model
mdl.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])

# Load weights if not training on current run
# mdl.load_weights('./modelCheckpoint/weights-improvement-40-0.88.hdf5')

# Define Callbacks #############################################################

# Send logging data to TensorBoard
tensorboard = TensorBoard(log_dir='./logs',
   histogram_freq=1,
   batch_size=batch_size,
   write_graph=True,
   write_grads=False,
   write_images=True)

# Checkpoints to save weights
checkpointPath = './modelCheckpoint/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(checkpointPath,
    monitor='val_acc',
    verbose=1,
    save_best_only=True,
    mode='max')

# Train the model ##############################################################

# Call Fit to train the model
mdl.fit(x_train, y_train,
          batch_size=batch_size,
          shuffle=True,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard,checkpoint])

# Compute Prediction Statistics from entire dataset
data = np.r_[x_train,x_test]
pred = mdl.predict_classes(data,batch_size=5)
true = np.r_[y_trainP,y_testP]

# Compoute and save confusion matrix
cnf_matrix = confusion_matrix(true, pred)
np.save('confusionMat.npy',conf_matrix)
