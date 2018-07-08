import keras
import numpy as np
np.random.seed(123)

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import callbacks
from keras.optimizers import SGD, Adam, RMSprop
import keras.backend as K
from keras.layers import Dropout, Activation, Convolution2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dense
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard

from prepare_data import prepare_data
from visualization import visualizer



# Metrics
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


# def jacard_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

# Losses, based on metrics
# def jacard_coef_loss(y_true, y_pred):
#     return -jacard_coef(y_true, y_pred)
#
#
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)




def build_model(raws):
    model = Sequential()
    model.add(Convolution2D(filters=32,
                            kernel_size=(2, 2),
                            activation='relu',
                            input_shape=raws.shape[1:],
                            padding='same'
                            ))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(BatchNormalization(axis=3))
    model.add(Convolution2D(filters=32,
                            kernel_size=(3, 3),
                            activation='relu',
                            padding='same'
                            ))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Convolution2D(filters=64,
                            kernel_size=(3, 3),
                            activation='relu',
                            padding='same'
                            ))
    model.add(UpSampling2D(size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Convolution2D(filters=32,
                            kernel_size=(3, 3),
                            activation='relu',
                            padding='same'
                            ))
    model.add(UpSampling2D(size=(2,2)))
    model.add(BatchNormalization(axis=3))

    model.add(Convolution2D(filters=1,
                            kernel_size=(1, 1),
                            input_shape=raws.shape[1:],
                            padding='same'
                            ))
    model.add(Activation("sigmoid"))




    learning_rate = 0.001
    # opt_choosing = "Adam" # SGD or RMS or Adam
    # if opt_choosing == "SGD":
    #     optimizer     = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    # elif opt_choosing == "Adam":
    #     optimizer    = Adam(lr=learning_rate)
    # elif opt_choosing == "RMS":
    #     optimizer = RMSprop(lr=learning_rate)
    # optimizer = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = Adam()

    model.compile(loss="binary_crossentropy",
                  # loss=dice_coef_loss,
                  optimizer=optimizer,
                  metrics=[dice_coef])
    print(model.summary())

    return model

def train_model(X_train, y_train, p_X_train, p_y_train ):

    pretrain = False
    epochs = 500
    batch_size = 32
    patience = 30
    filepath = "checkpoints/best_model-{epoch:02d}-{val_loss:.2f}.hdf5"

    callbacks = [
        # ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-9, epsilon=0.00001, verbose=1, mode='min'),
        ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto'),
        # EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        TensorBoard("tensorboard/", write_images=True)
    ]

    model = build_model(X_train)

    if pretrain:
        model.fit(      p_X_train,
                        p_y_train,
                        epochs=epochs//2,
                        batch_size=batch_size)
    model.fit(  X_train,
                y_train,
                validation_split=0.20,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
                )



def test_visualization(X_test, y_test):
    model = keras.models.load_model("best_models/best_model-462-0.01.hdf5", custom_objects={"dice_coef": dice_coef})
    visualizer(model, X_test, y_test)




if __name__ == '__main__':

    # Prepare and split data
    raws_th, labels_th, raws, labels = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(raws, labels, test_size=0)
    p_X_train, p_X_test, p_y_train,p_y_test  = train_test_split(raws_th, labels_th, test_size=0)

    # train_model(X_train, y_train, p_X_train, p_y_train )
    test_visualization(X_test, y_test)




