import os

import keras.losses
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping


# DATA SOURCE --------------------------------------------------

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_data_dir = './ball/train'
validation_data_dir =  './ball/validation'

num_of_classes = len(os.listdir(train_data_dir))


batch_size = 12
epochs = 200
size = (224, 224)
class_mode = 'categorical'

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    # validation_split=0.1,
    horizontal_flip=True,
    vertical_flip=True,
)

valid_datagen = ImageDataGenerator(
    rescale=1. / 255
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    shuffle=True,
    target_size=size,
    # subset='training',
    batch_size=batch_size,
    class_mode=class_mode)

validation_generator = valid_datagen.flow_from_directory(
    validation_data_dir,
    shuffle=False,
    # subset='validation',
    target_size=size,
    class_mode=class_mode)



# MODEL --------------------------------------------------

model = Sequential(
    [
        Conv2D(128, kernel_size=3, activation='relu', input_shape=(size[0], size[1], 3)),
        MaxPooling2D(pool_size=2),

        Conv2D(256, kernel_size=3, activation='relu'),
        MaxPooling2D(pool_size=2),

        Conv2D(256, kernel_size=3, activation='relu'),
        MaxPooling2D(pool_size=2),

        Conv2D(512, kernel_size=3, activation='relu'),
        MaxPooling2D(pool_size=2),

        Conv2D(512, kernel_size=3, activation='relu'),
        MaxPooling2D(pool_size=2),

        Conv2D(1024, kernel_size=3, activation='relu'),
        MaxPooling2D(pool_size=2),

        Flatten(),
        Dropout(0.2),
        Dense(4096, activation='relu'),
        Dropout(0.6),
        Dense(2048, activation='relu'),
        Dropout(0.5),

        Dense(num_of_classes, activation='softmax')
    ]
)

model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.5),
            metrics=['accuracy'])


# TRAINING --------------------------------------------------

early_stop = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=50, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, min_delta=1e-4)



with tf.device('/GPU:0'):

    model.fit(
        train_generator,
        validation_data=validation_generator,
        # callbacks=[early_stop, reduce_lr],
        epochs=epochs
    )


# model.save("model.h5")
