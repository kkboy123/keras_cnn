# python CNN_preprocess.py <train data> <test data> <submission file> <loop> <validation size> <sample size> <dropout one> <dropout two> <rotation> <shear> <zoom> <width> <height> <denoise>
# python CNN_preprocess.py train_data.csv validation_data.csv validation_predicted_cnn.csv 10 1000 1 0.5 0.5 20 0.3 0.2 0.1 0.1 192
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
import pandas as pd
import numpy as np
import sys

train_data_csv = sys.argv[1]
test_data_csv = sys.argv[2]
predicted_csv = sys.argv[3]
loop = int(sys.argv[4])
validation_size = int(sys.argv[5])
sample = int(sys.argv[6])
drop_one = float(sys.argv[7])
drop_two = float(sys.argv[8])
rotation = int(sys.argv[9])
shear = float(sys.argv[10])
zoom = float(sys.argv[11])
width = float(sys.argv[12])
height = float(sys.argv[13])
denoise = int(sys.argv[14])

# Options
VALIDATION_SIZE = validation_size
# input image dimensions
img_rows, img_cols = 28, 28
# the CIFAR10 images are RGB
img_channels = 1

bins = np.array([192.0,256.0])

# read training data from CSV file
data = pd.read_csv(train_data_csv, header=None)
labels = data.ix[:,1].values.astype('int32')
X_train = (data.iloc[:,2:].values).astype('float32')
X_train[X_train<denoise] = 0

# read test data from CSV file
data = pd.read_csv(test_data_csv, header=None)
ids = data.ix[:,0].values.astype(np.str)
X_test = data.iloc[:,1:].values.astype('float32')
X_test[X_test<denoise] = 0

# convert list of labels to binary class matrix
y_train = np_utils.to_categorical(labels) 


input_dim = X_train.shape[1]
nb_classes = y_train.shape[1]

# pre-processing: divide by max and substract mean
scale = np.max(X_test)
X_test /= scale

mean = np.std(X_test)
X_test -= mean

# pre-processing: divide by max and substract mean
scale = np.max(X_train)
X_train /= scale

mean = np.std(X_train)
X_train -= mean

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)


model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(img_channels, img_rows, img_cols)))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(Convolution2D(32, 3, 3))
# model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(drop_one))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(Convolution2D(64, 3, 3))
# model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(drop_two))

model.add(Flatten())
model.add(Dense(input_dim=64*7*7, output_dim=128, W_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(input_dim=128, output_dim=nb_classes, W_regularizer=l2(0.01)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# load weight
if predicted_csv.endswith('.hdf5'):
    model.load_weights(predicted_csv)
    predicted_csv = predicted_csv + ".csv"
# this will do preprocessing and realtime data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=rotation,  # randomly rotate images in the range (degrees, 0 to 180)
    shear_range=shear,  # Float. Shear Intensity (Shear angle in counter-clockwise direction as radians)
    zoom_range=zoom, # Float or [lower, upper]. Range for random zoom. If a float
    width_shift_range=width,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=height,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images


# model.load_weights('/Users/cynthia_lee/Desktop/eric/ep2/predicted_cnn.csv.049-0.1052-0.9730.hdf5')
print("Training...")
if (VALIDATION_SIZE):
    checkpointer = ModelCheckpoint(filepath=predicted_csv+".{epoch:03d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}-{val_acc:.4f}.hdf5", verbose=0, save_best_only=True, monitor='val_loss')
    stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    X_validation = X_train[:VALIDATION_SIZE]
    y_validation = y_train[:VALIDATION_SIZE]
    X_train = X_train[VALIDATION_SIZE:]
    y_train = y_train[VALIDATION_SIZE:]
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)
    print('X_train => {0}, y_train => {1}').format(X_train.shape, y_train.shape)
    #nn.fit(X_train, y_train, validation_data=(X_validation, y_validation), nb_epoch=loop, batch_size=16, verbose=1)
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                        samples_per_epoch=X_train.shape[0]*sample,
                        nb_epoch=loop,
                        validation_data=(X_validation, y_validation),
                        verbose=1,
                        callbacks=[checkpointer, stopping])
else:
    checkpointer = ModelCheckpoint(filepath=predicted_csv+".{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5", verbose=0, save_best_only=True, monitor='loss')
    stopping = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='auto')
    #nn.fit(X_train, y_train, validation_split=VALIDATION_SPLIT, nb_epoch=loop, batch_size=16, verbose=1)
    # fit the model on the batches generated by datagen.flow()
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                        samples_per_epoch=X_train.shape[0]*sample,
                        nb_epoch=loop,
                        verbose=1,
                        callbacks=[checkpointer, stopping])

print("Generating test predictions...")
preds = model.predict_proba(X_test, verbose=0)
print('preds => \n{0}').format(preds)


def write_preds(preds, fname):
    pd.DataFrame({"ImageId": ids, "prob0": preds[:,0], "prob1": preds[:,1], "prob2": preds[:,2], "prob3": preds[:,3], "prob4": preds[:,4], "prob5": preds[:,5], "prob6": preds[:,6], "prob7": preds[:,7], "prob8": preds[:,8], "prob9": preds[:,9]}).to_csv(fname, index=False, header=False)

write_preds(preds, predicted_csv)