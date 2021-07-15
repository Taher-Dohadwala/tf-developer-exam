"""
Below is code with a link to a happy or sad dataset which contains 80 images, 40 happy and 40 sad.
Create a convolutional neural network that trains to 100% accuracy on these images,  which cancels training upon hitting training accuracy of >.999

Hint -- it will work best with 3 convolutional layers.+
"""

"""
Results:
Epoch 80/100
4/4 [==============================] - 0s 106ms/step - loss: 0.0746 - acc: 1.0000
Reached max training accuracy. Stopping training!!!
"""
import tensorflow as tf


DESIRED_ACCURACY = 0.999

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),input_shape=(150,150,3),activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(16,(3,3),activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation="relu"),
    tf.keras.layers.Dense(1,activation="sigmoid")
])
# Load data
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        'h-or-s/',
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
# Callbacks
class MyEarlyStopping(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("acc") > 0.999:
            print("Reached max training accuracy. Stopping training!!!")
            self.model.stop_training = True

mycallback = MyEarlyStopping()
# Compile
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss="binary_crossentropy",metrics='acc')

# Fit
model.fit(
    train_generator,
    epochs=100,
    callbacks=[mycallback]
)

