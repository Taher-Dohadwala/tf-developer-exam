# In this exercise you will train a CNN on the FULL Cats-v-dogs dataset
# This will require you doing a lot of data preprocessing because
# the dataset isn't split into training and validation for you
#visit https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765

# The expectation here is that the model will train, and that accuracy will be > 95% on both training and validation

# Command line preprocessing trick to remove all files with zero bytes
# find . -type f -size 0b -print -delete
import tensorflow as tf

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),input_shape=(150,150,3),activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(16,(3,3),activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dense(1,activation="sigmoid")
])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss="binary_crossentropy",metrics=['acc'])

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.1)

train_generator = train_datagen.flow_from_directory(
    "cats_dogs/PetImages",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=True,
    subset="training"
)


validation_generator = train_datagen.flow_from_directory(
    "cats_dogs/PetImages",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset="validation"
)

# Callbacks
class MyEarlyStopping(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("acc") and logs.get("val_acc") > 0.95:
            print("Reached 95% accuracy on training and validation!!!")
            self.model.stop_training = True


mycallback = MyEarlyStopping()

model.fit(
    train_generator,
    validation_data = validation_generator,
    epochs = 100,
    callbacks=[mycallback])