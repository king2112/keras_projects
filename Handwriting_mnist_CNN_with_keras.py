

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.optimizers import Adam




batch_size = 128
num_classes = 10
epochs = 10

img_rows , img_cols=28,28

(x_train , y_train),(x_test , y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols).astype('float32')
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1).astype('float32')
    input_shape = (img_rows, img_cols, 1)

x_train /= 255
x_test /= 255


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)




cnn_model=Sequential([
    Conv2D(filters=32 , kernel_size=3 , activation='relu' , input_shape=input_shape),
    MaxPooling2D(pool_size=2),
    Dropout(0.2),
    Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=2),
    Dropout(0.2),
    Flatten(),
    Dense(128 ,activation='relu'),
    Dropout(0.5),
    Dense(10 , activation='softmax')
])

tensorboard = TensorBoard(
    log_dir=r'logs\{}'.format('cnn_1layer'),
    write_graph=True,
    write_grads=True,
    histogram_freq=1,
    write_images=True,
)

cnn_model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(0.01),
    metrics=['accuracy']
)


cnn_model.fit(
    x_train ,y_train,
    batch_size = batch_size,
    epochs = epochs,
    verbose=1,
    validation_data=(x_test , y_test)

)
score = cnn_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


