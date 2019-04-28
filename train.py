import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD 
from alexnet_cifar10 import *


batch_size = 128 
num_classes = 10
epochs = 100 
image_size = 32
channel = 3 

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = AlexNet(image_size, channel, num_classes)
model.compile(optimizer=SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

train_gen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True, 
                width_shift_range=4.0/32.0, height_shift_range=4.0/32.0)
test_gen = ImageDataGenerator(rescale=1.0/255)

model.fit_generator(train_gen.flow(x_train, y_train, batch_size, shuffle=True),
                        steps_per_epoch=x_train.shape[0]//batch_size,
                        validation_data=test_gen.flow(x_test, y_test, batch_size, shuffle=False),
                        validation_steps=x_test.shape[0]//batch_size,
                        max_queue_size=5, epochs=epochs)
