"""
Exercise 3
In the videos you looked at how you would improve Fashion MNIST using Convolutions.
For your exercise see if you can improve MNIST to 99.8% accuracy or more using only a single convolutional layer and a single MaxPooling 2D.
You should stop training once the accuracy goes above this amount. It should happen in less than 20 epochs,
so it's ok to hard code the number of epochs for training, but your training must end once it hits the above metric.
If it doesn't, then you'll need to redesign your layers.

I've started the code for you -- you need to finish it!

When 99.8% accuracy has been hit, you should print out the string "Reached 99.8% accuracy so cancelling training!"
"""
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize Features
x_train = x_train/255.0
x_test  = x_test/255.0

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

class MyEarlyStopping(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.998):
            print("Reached 99.8% accuracy so cancelling training!")
            self.model.stop_training = True

callback = MyEarlyStopping()

callback = MyEarlyStopping()

EPOCHS = 10

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),input_shape=(28,28,1),activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation="swish"),
    tf.keras.layers.Dense(10,activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),epochs=EPOCHS,callbacks=[callback])